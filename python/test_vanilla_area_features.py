from dataclasses import fields, replace
import json
from pathlib import Path
import unittest

import torch

from env import Engine, FeatureSlot, mask_unforced_vanilla_area_features
from train_config import Config, GENERATION_VARIABLE_FLOAT_FIELDS, VANILLA_AREA_CONDITION_FIELDS


class VanillaAreaFeaturesTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for mixed-device inputs")
    def test_cuda_configuration_with_cpu_outcomes(self) -> None:
        raw = torch.tensor([[0, 1, -1, 0, 1, -1]], dtype=torch.int8)
        variables = torch.zeros(1, len(GENERATION_VARIABLE_FLOAT_FIELDS), device="cuda")
        force_columns = [
            GENERATION_VARIABLE_FLOAT_FIELDS.index(name) for name in VANILLA_AREA_CONDITION_FIELDS
        ]
        variables[:, force_columns[3:]] = 1
        expected = torch.tensor([[-1, -1, -1, 0, 1, -1]], dtype=torch.int8)
        for candidate_count in (1, 3):
            with self.subTest(candidate_count=candidate_count):
                result = mask_unforced_vanilla_area_features(
                    raw.unsqueeze(1).expand(-1, candidate_count, -1),
                    variables.unsqueeze(1).expand(-1, candidate_count, -1),
                )
                self.assertEqual(result.device, variables.device)
                torch.testing.assert_close(
                    result.cpu(), expected.unsqueeze(1).expand(-1, candidate_count, -1)
                )
        torch.testing.assert_close(raw, torch.tensor([[0, 1, -1, 0, 1, -1]], dtype=torch.int8))

    def test_state_and_candidate_builders_mask_only_unforced_features(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = Config.model_validate_json((root / "configs/debug.json").read_text())
        engine = Engine(
            json.loads((root / config.room_set).read_text()),
            config.features,
            config.generation.min_area_size,
            config.generation.max_area_size,
        )
        env = engine.create_environment_group(
            map_size=config.map_size,
            num_envs=2,
            candidate_spatial_cell_size=config.generation.candidate_spatial_cell_size,
            area_bounding_box_width=config.generation.area_bounding_box_width,
            area_bounding_box_height=config.generation.area_bounding_box_height,
            seed=0,
            num_threads=1,
        )
        slot = FeatureSlot(env, pin_memory=False)
        slot.ensure(4, 0, 0, 0)
        raw = torch.tensor([[0, 1, -1, 0, 1, -1], [0, 1, -1, 0, 1, -1]], dtype=torch.int8)
        outcomes = replace(
            env.get_current_feature_outcomes(torch.device("cpu"), 0, 2),
            vanilla_area_invalid=raw,
        )
        variables = torch.zeros(2, len(GENERATION_VARIABLE_FLOAT_FIELDS))
        force_columns = [
            GENERATION_VARIABLE_FLOAT_FIELDS.index(name) for name in VANILLA_AREA_CONDITION_FIELDS
        ]
        variables[:, force_columns] = torch.tensor(
            [[0, 0, 0, 1, 1, 1], [1, 1, 1, 0, 0, 0]], dtype=torch.float32
        )
        expected = torch.tensor([[-1, -1, -1, 0, 1, -1], [0, 1, -1, -1, -1, -1]])
        candidate_outcomes = replace(
            outcomes,
            **{
                field.name: getattr(outcomes, field.name).unsqueeze(1).expand(
                    -1, 2, *getattr(outcomes, field.name).shape[1:]
                )
                for field in fields(outcomes)
            },
        )
        # Swap configurations across candidate slots to catch broadcasting mistakes.
        candidate_variables = torch.stack([variables, variables.flip(0)], dim=1)
        candidate_expected = torch.stack([expected, expected.flip(0)], dim=1)
        for include_variables in (False, True):
            for include_lookahead in (False, True):
                with self.subTest(variables=include_variables, lookahead=include_lookahead):
                    shared = dict(
                        environment_count=2,
                        include_temperature=False,
                        include_recommended_candidates=False,
                        include_generation_variable_floats=include_variables,
                        include_lookahead_outcomes=include_lookahead,
                        frontier_row_count=0,
                        missing_connect_query_row_count=0,
                        save_refill_utility_query_row_count=0,
                    )
                    state = slot.state_features(
                        **shared,
                        log_temperature=torch.zeros(2),
                        log_recommended_candidates=torch.zeros(2),
                        generation_variable_floats=variables,
                        lookahead_outcomes=outcomes,
                    ).global_features
                    candidate = slot.features(
                        **shared,
                        candidate_count=2,
                        log_temperature=torch.zeros(2, 2),
                        log_recommended_candidates=torch.zeros(2, 2),
                        generation_variable_floats=candidate_variables,
                        lookahead_outcomes=candidate_outcomes,
                    ).global_features
                    if include_lookahead:
                        torch.testing.assert_close(
                            state.lookahead_vanilla_area_invalid, expected.to(raw.dtype)
                        )
                        torch.testing.assert_close(
                            candidate.lookahead_vanilla_area_invalid,
                            candidate_expected.to(raw.dtype),
                        )
                        torch.testing.assert_close(
                            state.lookahead_connection_invalid, outcomes.connection_invalid
                        )
                    else:
                        assert state.lookahead_vanilla_area_invalid.shape == (2, 0)
                        assert candidate.lookahead_vanilla_area_invalid.shape == (2, 2, 0)
        torch.testing.assert_close(outcomes.vanilla_area_invalid, raw)
        torch.testing.assert_close(
            raw, torch.tensor([[0, 1, -1, 0, 1, -1]] * 2, dtype=torch.int8)
        )


if __name__ == "__main__":
    unittest.main()
