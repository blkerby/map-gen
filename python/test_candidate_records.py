from types import SimpleNamespace
from threading import Lock
import unittest
from unittest.mock import Mock, patch

import torch

from env import Actions, CandidateBatch, ProposalData
from generate import (
    compute_group_proposal_shortlist,
    empty_proposal_data,
    score_staged_candidate_request,
    select_candidate_actions,
)
from learn import compute_candidate_diagnostics, proposal_batch_loss


class CandidateRecordsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.candidates = Actions(
            room_idx=torch.tensor([[0, 0], [1, 1]]),
            room_x=torch.zeros(2, 2),
            room_y=torch.zeros(2, 2),
            room_area=torch.zeros(2, 2, dtype=torch.int64),
        )
        self.group = SimpleNamespace(
            config=SimpleNamespace(
                autocast=False, recommended_candidates=0, temperature=torch.tensor([0.5, 1.0])
            ),
            balance_score_tables=SimpleNamespace(
                room_area=Mock(), toilet_crossed_room=Mock(), toilet_failure=Mock()
            ),
            area_balance_dual_mask=torch.zeros(2, 1, dtype=torch.bool),
        )
        self.features = SimpleNamespace(
            global_features=SimpleNamespace(
                room_placed=torch.zeros(4, 1, dtype=torch.bool),
                toilet_crossed_room_idx=torch.full((4,), -1),
            )
        )
        self.outcomes = SimpleNamespace(
            door_invalid=torch.zeros(2, 2, 1), toilet_invalid=torch.zeros(2, 2)
        )
        self.profiler = Mock(enabled=False)

    def sample_candidates(self):
        fields = {
            name: torch.zeros(4)
            for name in (
                "door_invalid",
                "connection_invalid",
                "toilet_invalid",
                "phantoon_pair_invalid",
                "phantoon_area_invalid",
                "balance_score",
                "area_balance_score",
                "toilet_balance_score",
                "avg_frontiers",
                "graph_diameter",
                "save_to_room_utility",
                "save_from_room_utility",
                "refill_to_room_utility",
                "refill_from_room_utility",
                "missing_connect_utility",
                "area_crossings",
                "area_x",
                "area_y",
            )
        }
        predictions = SimpleNamespace(
            **fields,
            vanilla_area_invalid=torch.zeros(4, 6),
            area_size=torch.zeros(4, 3),
            area_map_station_count=torch.zeros(4, 3),
            proposal_state=None,
            proposal_row_snapshot_idx=None,
            proposal_row_frontier_idx=None,
        )
        predictions.balance_score = torch.tensor([-1.0, -3.0, 0.0, 0.0])
        with (
            patch(
                "generate.compute_step_balance_score_targets",
                return_value=(torch.zeros(2, 2, 1), torch.zeros(2, 2, 1, dtype=torch.bool)),
            ),
            patch(
                "generate.apply_candidate_area_balance_scores", return_value=torch.zeros(2, 2, 1)
            ),
            patch(
                "generate.apply_candidate_toilet_balance_score",
                return_value=torch.tensor([[2.0, -1.0], [0.0, 0.0]]),
            ),
            patch(
                "generate.compute_expected_reward",
                return_value=torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
            ),
            patch(
                "generate.area_balance_reward",
                return_value=torch.tensor([[2.0, -1.0], [0.0, 0.0]]),
            ),
            patch("generate.rand_choice", return_value=torch.tensor([1, 0])) as sampler,
        ):
            selection = select_candidate_actions(
                self.group,
                Mock(return_value=predictions),
                self.candidates,
                self.outcomes,
                SimpleNamespace(
                    door_match=torch.zeros(2, 2, 1), toilet_invalid=torch.zeros(2, 2)
                ),
                self.features,
                self.device,
                1,
                self.profiler,
            )
        expected_logits = torch.tensor(
            [[1.0, 4.0], [float("-inf"), float("-inf")]]
        ) / self.group.config.temperature.unsqueeze(1)
        torch.testing.assert_close(selection.sampling_logits, expected_logits)
        expected_probs = torch.stack(
            [torch.softmax(expected_logits[0], dim=0), torch.tensor([1.0, 0.0])]
        )
        torch.testing.assert_close(sampler.call_args.args[0], expected_probs)
        return selection

    def test_recorded_logits_match_sampler_with_all_balance_terms(self) -> None:
        self.sample_candidates()

    def test_high_temperature_flattens_all_final_sampling_preferences(self) -> None:
        self.group.config.temperature = torch.full((2,), 1_000_000.0)
        selection = self.sample_candidates()
        torch.testing.assert_close(
            selection.sampling_logits[0].softmax(0), torch.full((2,), 0.5), atol=1e-6, rtol=0
        )

    def test_proposal_sampling_scales_scores_and_learned_prices_together(self) -> None:
        group = SimpleNamespace(
            previous_proposal_scores=Mock(),
            config=SimpleNamespace(
                temperature=torch.ones(1),
                proposal_temperature=torch.tensor([2.0]),
                shortlist_candidates=2,
            ),
            proposal_balance_score_table=Mock(),
            proposal_area_balance_score_table=Mock(),
        )
        shared = SimpleNamespace(
            profiler=self.profiler,
            gpu_lock=Lock(),
            door_variant_compatibility=torch.ones(1, 1, dtype=torch.bool),
            door_variant_connection_variant_idx=torch.zeros(1, dtype=torch.int64),
        )
        model = Mock()
        model.proposal_output.output_dtype = torch.float32
        model.proposal_output.return_value = torch.tensor([[1.0, 2.0]])
        with (
            patch("generate.prepare_proposal_inputs", return_value=SimpleNamespace(features=None)),
            patch(
                "generate.selected_proposal_rows_from_cache",
                return_value=(
                    torch.zeros(1, 1),
                    torch.zeros(1, dtype=torch.int64),
                    torch.ones(1, 1),
                    torch.zeros(1, dtype=torch.int64),
                    torch.zeros(1, dtype=torch.int64),
                ),
            ),
            patch(
                "generate.compute_proposal_balance_score_residual",
                return_value=torch.tensor([[2.0, 4.0]]),
            ),
            patch(
                "generate.compute_proposal_area_balance_score_residual",
                return_value=torch.tensor([[3.0, -2.0]]),
            ),
            patch(
                "generate.sample_proposal_shortlist",
                return_value=(
                    torch.tensor([[0, 0]]),
                    torch.tensor([[0, 1]]),
                    torch.tensor([2]),
                    torch.tensor([2]),
                ),
            ) as sampler,
            patch("generate.add_stat_totals"),
        ):
            compute_group_proposal_shortlist(group, model, self.device, shared)
        torch.testing.assert_close(sampler.call_args.args[0], torch.tensor([[3.0, 2.0]]))

    def test_empty_candidate_records_have_zero_diagnostics(self) -> None:
        for candidate_count in (0, 3):
            data = empty_proposal_data(2, candidate_count, self.device)
            diagnostics = compute_candidate_diagnostics(data, proposal_target_temperature=1.0)
            for value in (
                diagnostics.selected_probability,
                diagnostics.target_entropy,
                diagnostics.uniform_kl,
            ):
                torch.testing.assert_close(value, torch.tensor(0.0))

    def test_rejected_negative_and_logits_reach_diagnostics_and_training(self) -> None:
        selection = self.sample_candidates()
        self.group.config.recommended_candidates = 2
        batch = CandidateBatch(
            candidates=self.candidates,
            proposal_frontier_idx=torch.tensor([[0, 0], [-1, -1]]),
            proposal_action_idx=torch.tensor([[0, 1], [-1, -1]]),
            proposal_rejected=torch.zeros(2, 2, dtype=torch.bool),
            scored_invalid_frontier_idx=torch.tensor([[0], [-1]]),
            scored_invalid_proposal_action_idx=torch.tensor([[2], [-1]]),
            scored_invalid_rejected=torch.tensor([[True], [False]]),
            reward_outcomes=self.outcomes,
            post_candidate_outcomes=SimpleNamespace(door_match=torch.zeros(2, 2, 1)),
            feature_requirements=Mock(),
            stats=Mock(),
        )
        prepared = SimpleNamespace(
            proposal_balance_residual=torch.zeros(2, 2),
            scored_invalid_proposal_balance_residual=torch.zeros(2, 1),
        )
        staged = SimpleNamespace(
            ready_event=None,
            features=self.features,
            candidate_batch=batch,
            request=SimpleNamespace(group=self.group, prepared_step=prepared),
        )
        with (
            patch("generate.select_candidate_actions", return_value=selection),
            patch("generate.select_outcomes", return_value=Mock()),
        ):
            result = score_staged_candidate_request(staged, Mock(), self.device, 1, self.profiler)
        data = (
            ProposalData(
                frontier_idx=result.proposal_frontier_idx.unsqueeze(1),
                action_idx=result.proposal_action_idx.unsqueeze(1),
                invalid=result.proposal_invalid.unsqueeze(1),
                rejected=result.proposal_rejected.unsqueeze(1),
                sampling_logits=result.sampling_logits.unsqueeze(1),
                selected_candidate=result.selected_candidate.unsqueeze(1),
                target_reward=result.target_reward.unsqueeze(1),
                balance_residual=result.balance_residual.unsqueeze(1),
            )
            .slice(0, 1)
            .to(self.device)
        )
        torch.testing.assert_close(data.invalid, torch.tensor([[[False, False, True]]]))
        torch.testing.assert_close(data.rejected, data.invalid)
        torch.testing.assert_close(
            data.sampling_logits, torch.tensor([[[2.0, 8.0, float("-inf")]]])
        )
        diagnostics = compute_candidate_diagnostics(data, proposal_target_temperature=1.0)
        torch.testing.assert_close(
            diagnostics.selected_probability, torch.sigmoid(torch.tensor(6.0))
        )
        scores = torch.tensor([[0.0, 0.0, 5.0]], requires_grad=True)
        loss = proposal_batch_loss(
            scores,
            data.target_reward[:, 0],
            data.invalid[:, 0],
            1.0,
            self.device,
        )
        loss.backward()
        self.assertGreater(scores.grad[0, 2].item(), 0.0)

    def test_fallback_probability_is_retained_and_unselected_rows_are_ignored(self) -> None:
        data = ProposalData(
            frontier_idx=torch.tensor([[[0, 0]], [[-1, -1]]]),
            action_idx=torch.tensor([[[0, 1]], [[-1, -1]]]),
            invalid=torch.zeros(2, 1, 2, dtype=torch.bool),
            rejected=torch.tensor([[[True, True]], [[False, False]]]),
            sampling_logits=torch.tensor([[[0.0, 2.0]], [[float("-inf"), float("-inf")]]]),
            selected_candidate=torch.tensor([[1], [0]]),
            target_reward=torch.zeros(2, 1, 2),
            balance_residual=torch.zeros(2, 1, 2),
        )
        diagnostics = compute_candidate_diagnostics(data, proposal_target_temperature=1.0)
        torch.testing.assert_close(
            diagnostics.selected_probability, torch.sigmoid(torch.tensor(2.0))
        )
        data.selected_candidate.fill_(-1)
        torch.testing.assert_close(
            compute_candidate_diagnostics(data, 1.0).selected_probability, torch.tensor(0.0)
        )


if __name__ == "__main__":
    unittest.main()
