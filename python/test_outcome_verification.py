import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from env import Actions, Engine
from generate import verify_and_step
from train_config import Config, instantiate_scheduleable_config


class OutcomeVerificationTest(unittest.TestCase):
    def test_unfinished_zebes_verification_and_final_export(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs/zebes.json"
        config = instantiate_scheduleable_config(
            Config.model_validate_json(config_path.read_text()), 0
        )
        rooms = json.loads(config.room_set.read_text())
        engine = Engine(
            rooms,
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
            seed=1234,
            frontier_neighbor_count=config.generation.frontier_neighbor_count,
            frontier_window_size=config.generation.frontier_window_size,
            num_threads=2,
            frontier_neighbor_algorithm=config.generation.frontier_neighbor_algorithm,
        )
        for _ in range(2):
            env.clear()
            actions = Actions(
                room_idx=torch.zeros(2, dtype=torch.uint8),
                room_x=torch.full((2,), 32, dtype=torch.int8),
                room_y=torch.full((2,), 32, dtype=torch.int8),
                room_area=torch.zeros(2, dtype=torch.uint8),
            )
            verify_and_step(SimpleNamespace(env=env), actions, True)
            partial = env.get_current_feature_outcomes(torch.device("cpu"), 0, 2)
            assert (partial.maridia_water == -1).any()
            assert (partial.norfair_heat == -1).any()
            env.verify_outcome_consistency()
            env.finish()
            final = env.get_outcomes(torch.device("cpu"), verify_consistency=True)
            assert (final.step_outcomes.maridia_water >= 0).all()
            assert (final.step_outcomes.norfair_heat >= 0).all()

    def test_generation_verifies_without_exporting_and_propagates_errors(self) -> None:
        env = Mock()
        actions = Mock(spec=Actions)
        env.verify_outcome_consistency.side_effect = RuntimeError("known outcome changed")
        with self.assertRaisesRegex(RuntimeError, "known outcome changed"):
            verify_and_step(SimpleNamespace(env=env), actions, True)
        env.step.assert_called_once_with(actions)
        env.verify_outcome_consistency.assert_called_once_with()
        env.get_outcomes.assert_not_called()

    def test_disabled_verification_only_steps(self) -> None:
        env = Mock()
        actions = Mock(spec=Actions)
        verify_and_step(SimpleNamespace(env=env), actions, False)
        env.step.assert_called_once_with(actions)
        env.verify_outcome_consistency.assert_not_called()
        env.get_outcomes.assert_not_called()


if __name__ == "__main__":
    unittest.main()
