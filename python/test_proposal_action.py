import torch

from env import (
    AREA_COUNT,
    ProposalData,
    proposal_action_door_variant_idx,
    proposal_action_idx,
    proposal_action_room_area,
)
from generate import sample_proposal_shortlist
from learn import (
    compute_candidate_diagnostics,
    proposal_batch_loss,
    proposal_scores_for_candidates,
)
from model import ProposalOutput


def test_proposal_action_helpers_flatten_area_variants() -> None:
    door_variant_idx = torch.tensor([0, 0, 1, 2], dtype=torch.int16)
    room_area = torch.tensor([0, 5, 3, 4], dtype=torch.int16)
    action_idx = proposal_action_idx(door_variant_idx, room_area)

    assert AREA_COUNT == 6
    assert action_idx.tolist() == [0, 5, 9, 16]
    assert proposal_action_door_variant_idx(action_idx).tolist() == [0, 0, 1, 2]
    assert proposal_action_room_area(action_idx).tolist() == [0, 5, 3, 4]


def test_proposal_loss_compares_candidate_scores() -> None:
    device = torch.device("cpu")
    target_reward = torch.tensor([[0.0, 10.0]], dtype=torch.float32)

    aligned_score = torch.tensor([[0.0, 10.0]], dtype=torch.float32)
    reversed_score = torch.tensor([[10.0, 0.0]], dtype=torch.float32)

    aligned_loss = proposal_batch_loss(
        aligned_score,
        target_reward,
        torch.zeros_like(target_reward, dtype=torch.bool),
        1.0,
        device,
    )
    reversed_loss = proposal_batch_loss(
        reversed_score,
        target_reward,
        torch.zeros_like(target_reward, dtype=torch.bool),
        1.0,
        device,
    )

    assert torch.isfinite(aligned_loss)
    assert torch.isfinite(reversed_loss)
    assert aligned_loss < reversed_loss


def test_invalid_candidate_receives_downward_gradient() -> None:
    candidate_score = torch.tensor([[0.0, 5.0]], requires_grad=True)
    loss = proposal_batch_loss(
        candidate_score,
        torch.zeros((1, 2)),
        torch.tensor([[False, True]]),
        1.0,
        torch.device("cpu"),
    )

    loss.backward()
    assert candidate_score.grad is not None
    assert candidate_score.grad[0, 1] > 0


def test_all_invalid_row_has_zero_proposal_loss() -> None:
    candidate_score = torch.tensor([[1.0, 2.0]], requires_grad=True)
    loss = proposal_batch_loss(
        candidate_score,
        torch.zeros((1, 2)),
        torch.ones((1, 2), dtype=torch.bool),
        1.0,
        torch.device("cpu"),
    )

    assert loss.item() == 0.0


def test_proposal_target_temperature_controls_target_sharpness() -> None:
    candidate_score = torch.tensor([[0.0, 2.0]])
    target_reward = torch.tensor([[0.0, 1.0]])
    invalid = torch.zeros((1, 2), dtype=torch.bool)

    matching_sharp_loss = proposal_batch_loss(
        candidate_score,
        target_reward,
        invalid,
        0.5,
        torch.device("cpu"),
    )
    softer_loss = proposal_batch_loss(
        candidate_score,
        target_reward,
        invalid,
        1.0,
        torch.device("cpu"),
    )

    assert matching_sharp_loss < softer_loss


def test_selected_probability_uses_generation_temperature() -> None:
    proposal_data = ProposalData(
        frontier_idx=torch.tensor([[[0, 0]]], dtype=torch.int16),
        action_idx=torch.tensor([[[0, 1]]], dtype=torch.int16),
        invalid=torch.zeros((1, 1, 2), dtype=torch.bool),
        selected_candidate=torch.ones((1, 1), dtype=torch.int64),
        target_reward=torch.tensor([[[0.0, 1.0]]]),
    )
    soft_target = compute_candidate_diagnostics(
        proposal_data,
        proposal_target_temperature=1.0,
        generation_temperature=torch.tensor([0.5]),
    )
    sharp_target = compute_candidate_diagnostics(
        proposal_data,
        proposal_target_temperature=0.1,
        generation_temperature=torch.tensor([0.5]),
    )

    expected_selected_probability = torch.softmax(torch.tensor([0.0, 2.0]), dim=0)[1]
    torch.testing.assert_close(
        soft_target.selected_probability,
        expected_selected_probability,
    )
    torch.testing.assert_close(
        sharp_target.selected_probability,
        expected_selected_probability,
    )
    assert sharp_target.target_entropy < soft_target.target_entropy


def test_proposal_scores_gather_candidates_across_frontiers() -> None:
    output = ProposalOutput(input_width=1, hidden_widths=[], output_width=3)
    with torch.no_grad():
        output.layers[-1].weight[:, 0] = torch.tensor([1.0, 2.0, 3.0])
    scores = proposal_scores_for_candidates(
        output,
        proposal_state=torch.tensor([[1.0], [2.0]]),
        row_snapshot_idx=torch.tensor([0, 0]),
        row_frontier_idx=torch.tensor([0, 1]),
        frontier_idx=torch.tensor([[0, 1]], dtype=torch.int16),
        action_idx=torch.tensor([[2, 0]], dtype=torch.int16),
        device=torch.device("cpu"),
    )

    assert torch.equal(scores, torch.tensor([[3.0, 2.0]]))


def test_proposal_shortlist_ranks_all_frontiers_per_environment() -> None:
    torch.manual_seed(0)
    frontier_idx, action_idx, pair_counts, possible_counts = sample_proposal_shortlist(
        proposal_scores=torch.tensor(
            [
                [1000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [500.0, 400.0, 0.0, 0.0, 0.0, 0.0],
                [300.0, 200.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        frontier_door_variant=torch.tensor([0, 0, 0]),
        inventory=torch.ones((2, 1), dtype=torch.int16),
        door_variant_compatibility=torch.ones((1, 1), dtype=torch.bool),
        door_variant_connection_variant_idx=torch.tensor([0]),
        row_snapshot_idx=torch.tensor([0, 0, 1]),
        row_frontier_idx=torch.tensor([0, 1, 0]),
        environment_count=2,
        shortlist_candidates=2,
        proposal_temperature=torch.ones(2),
        device=torch.device("cpu"),
    )

    assert frontier_idx.tolist() == [[0, 1], [0, 0]]
    assert action_idx.tolist() == [[0, 0], [0, 1]]
    assert pair_counts.tolist() == [12, 6]
    assert possible_counts.tolist() == [12, 6]


def test_proposal_shortlist_pads_environment_without_frontiers() -> None:
    frontier_idx, action_idx, pair_counts, possible_counts = sample_proposal_shortlist(
        proposal_scores=torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        frontier_door_variant=torch.tensor([0]),
        inventory=torch.ones((2, 1), dtype=torch.int16),
        door_variant_compatibility=torch.ones((1, 1), dtype=torch.bool),
        door_variant_connection_variant_idx=torch.tensor([0]),
        row_snapshot_idx=torch.tensor([0]),
        row_frontier_idx=torch.tensor([0]),
        environment_count=2,
        shortlist_candidates=2,
        proposal_temperature=torch.ones(2),
        device=torch.device("cpu"),
    )

    assert frontier_idx[1].tolist() == [-1, -1]
    assert action_idx[1].tolist() == [-1, -1]
    assert pair_counts.tolist() == [6, 0]
    assert possible_counts.tolist() == [6, 0]


def test_proposal_shortlist_masks_incompatible_door_variants() -> None:
    frontier_idx, action_idx, _, possible_counts = sample_proposal_shortlist(
        proposal_scores=torch.tensor([[0.0] * AREA_COUNT + [1000.0] * AREA_COUNT]),
        frontier_door_variant=torch.tensor([0]),
        inventory=torch.ones((1, 2), dtype=torch.int16),
        door_variant_compatibility=torch.tensor([[True, False], [False, True]]),
        door_variant_connection_variant_idx=torch.tensor([0, 1]),
        row_snapshot_idx=torch.tensor([0]),
        row_frontier_idx=torch.tensor([0]),
        environment_count=1,
        shortlist_candidates=AREA_COUNT,
        proposal_temperature=torch.ones(1),
        device=torch.device("cpu"),
    )

    assert frontier_idx.tolist() == [[0] * AREA_COUNT]
    assert sorted(action_idx[0].tolist()) == list(range(AREA_COUNT))
    assert possible_counts.tolist() == [AREA_COUNT]


def test_proposal_shortlist_pads_fully_incompatible_frontier() -> None:
    frontier_idx, action_idx, _, possible_counts = sample_proposal_shortlist(
        proposal_scores=torch.zeros((1, AREA_COUNT)),
        frontier_door_variant=torch.tensor([0]),
        inventory=torch.ones((1, 1), dtype=torch.int16),
        door_variant_compatibility=torch.tensor([[False]]),
        door_variant_connection_variant_idx=torch.tensor([0]),
        row_snapshot_idx=torch.tensor([0]),
        row_frontier_idx=torch.tensor([0]),
        environment_count=1,
        shortlist_candidates=2,
        proposal_temperature=torch.ones(1),
        device=torch.device("cpu"),
    )

    assert frontier_idx.tolist() == [[-1, -1]]
    assert action_idx.tolist() == [[-1, -1]]
    assert possible_counts.tolist() == [0]


def test_proposal_shortlist_masks_door_variants_without_inventory() -> None:
    frontier_idx, action_idx, _, possible_counts = sample_proposal_shortlist(
        proposal_scores=torch.tensor([[0.0] * AREA_COUNT + [1000.0] * AREA_COUNT]),
        frontier_door_variant=torch.tensor([0]),
        inventory=torch.tensor([[1, 0]], dtype=torch.int16),
        door_variant_compatibility=torch.ones((2, 2), dtype=torch.bool),
        door_variant_connection_variant_idx=torch.tensor([0, 1]),
        row_snapshot_idx=torch.tensor([0]),
        row_frontier_idx=torch.tensor([0]),
        environment_count=1,
        shortlist_candidates=AREA_COUNT,
        proposal_temperature=torch.ones(1),
        device=torch.device("cpu"),
    )

    assert frontier_idx.tolist() == [[0] * AREA_COUNT]
    assert sorted(action_idx[0].tolist()) == list(range(AREA_COUNT))
    assert possible_counts.tolist() == [AREA_COUNT]


def test_proposal_possible_count_is_computed_without_a_shortlist() -> None:
    frontier_idx, action_idx, pair_counts, possible_counts = sample_proposal_shortlist(
        proposal_scores=torch.zeros((1, AREA_COUNT * 2)),
        frontier_door_variant=torch.tensor([0]),
        inventory=torch.tensor([[1, 0]], dtype=torch.int16),
        door_variant_compatibility=torch.ones((2, 2), dtype=torch.bool),
        door_variant_connection_variant_idx=torch.tensor([0, 1]),
        row_snapshot_idx=torch.tensor([0]),
        row_frontier_idx=torch.tensor([0]),
        environment_count=1,
        shortlist_candidates=0,
        proposal_temperature=torch.ones(1),
        device=torch.device("cpu"),
    )

    assert frontier_idx.shape == (1, 0)
    assert action_idx.shape == (1, 0)
    assert pair_counts.tolist() == [AREA_COUNT * 2]
    assert possible_counts.tolist() == [AREA_COUNT]


def main() -> None:
    test_proposal_action_helpers_flatten_area_variants()
    test_proposal_loss_compares_candidate_scores()
    test_invalid_candidate_receives_downward_gradient()
    test_all_invalid_row_has_zero_proposal_loss()
    test_proposal_target_temperature_controls_target_sharpness()
    test_selected_probability_uses_generation_temperature()
    test_proposal_scores_gather_candidates_across_frontiers()
    test_proposal_shortlist_ranks_all_frontiers_per_environment()
    test_proposal_shortlist_pads_environment_without_frontiers()
    test_proposal_shortlist_masks_incompatible_door_variants()
    test_proposal_shortlist_pads_fully_incompatible_frontier()
    test_proposal_shortlist_masks_door_variants_without_inventory()
    test_proposal_possible_count_is_computed_without_a_shortlist()


if __name__ == "__main__":
    main()
