import unittest
from dataclasses import fields, replace
from pathlib import Path

import torch
from pydantic import ValidationError

from env import count_invalid_outcomes
from generate import compute_expected_reward
from learn import accumulate_main_loss, average_main_loss, empty_main_loss_breakdown
from test_area_rewards import area_predictions, unknown_outcomes, zero_generate_config
from train_config import Config, instantiate_scheduleable_config, validate_config


class SuccessTest(unittest.TestCase):
    def test_terminal_validity_includes_each_required_outcome(self) -> None:
        unknown = unknown_outcomes()
        valid = replace(
            unknown,
            **{field.name: torch.zeros_like(getattr(unknown, field.name)) for field in fields(unknown)},
        )
        areas_valid = torch.ones((1, 2, 6), dtype=torch.bool)
        enabled = torch.ones_like(areas_valid)
        torch.testing.assert_close(
            count_invalid_outcomes(valid, enabled, areas_valid, areas_valid),
            torch.zeros((1, 2), dtype=torch.int64),
        )
        for name in (
            "door_invalid", "connection_invalid", "toilet_invalid",
            "phantoon_pair_invalid", "phantoon_area_invalid", "vanilla_area_invalid",
        ):
            for value in (1, -1):
                with self.subTest(outcome=name, value=value):
                    failed = getattr(valid, name).clone()
                    failed.reshape(-1)[0] = value
                    outcomes = replace(valid, **{name: failed})
                    torch.testing.assert_close(
                        count_invalid_outcomes(outcomes, enabled, areas_valid, areas_valid),
                        torch.tensor([[1, 0]]),
                    )
        failed_areas = areas_valid.clone()
        failed_areas[0, 0, 0] = False
        for size_valid, station_valid in (
            (failed_areas, areas_valid), (areas_valid, failed_areas),
        ):
            torch.testing.assert_close(
                count_invalid_outcomes(valid, enabled, size_valid, station_valid),
                torch.tensor([[1, 0]]),
            )
        disabled = replace(valid, vanilla_area_invalid=torch.ones_like(valid.vanilla_area_invalid))
        assert not count_invalid_outcomes(
            disabled, torch.zeros_like(enabled), areas_valid, areas_valid
        ).any()

    def test_success_reward_is_weighted_log_probability(self) -> None:
        predictions = replace(area_predictions(), success=torch.tensor([[-100.0, 2.0]]))
        outcomes = unknown_outcomes()
        config = zero_generate_config()
        baseline = compute_expected_reward(predictions, outcomes, config)
        torch.testing.assert_close(baseline, torch.zeros_like(baseline))
        for weight in (3.0, torch.tensor([3.0])):
            reward = compute_expected_reward(
                predictions, outcomes, replace(config, reward_success=weight)
            )
            torch.testing.assert_close(
                reward - baseline, 3.0 * torch.nn.functional.logsigmoid(predictions.success)
            )
            assert torch.isfinite(reward).all()
            assert reward[0, 1] > reward[0, 0]

    def test_success_fields_are_required_and_reward_can_be_scheduled(self) -> None:
        config = Config.model_validate_json(Path("configs/debug.json").read_text())
        assert config.train.success_weight == 1.0
        assert config.generation.reward_success == 0.0
        for section, field in (("generation", "reward_success"), ("train", "success_weight")):
            data = config.model_dump(mode="json")
            del data[section][field]
            with self.assertRaises(ValidationError):
                Config.model_validate(data)
            data[section][field] = -1.0
            with self.assertRaisesRegex(ValueError, field):
                validate_config(Config.model_validate(data))
        data = config.model_dump(mode="json")
        data["generation"]["reward_success"] = {"linear": [0.0, 2.0]}
        scheduled = Config.model_validate(data)
        validate_config(scheduled)
        assert instantiate_scheduleable_config(scheduled, 320).generation.reward_success == 1.0

    def test_success_loss_survives_aggregation(self) -> None:
        total = empty_main_loss_breakdown()
        source = replace(total, success=0.75, success_contribution=0.03)
        accumulate_main_loss(total, source)
        accumulate_main_loss(total, source)
        averaged = average_main_loss(total, 2)
        assert averaged.success == source.success
        assert averaged.success_contribution == source.success_contribution


if __name__ == "__main__":
    unittest.main()
