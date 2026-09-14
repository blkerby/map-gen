import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import safetensors.torch
import torch

from env import Actions, EpisodeData
from experience import ExperienceStorage, load_balance_experience
from features import GenerationVariableFloatsFeature
from learn import generation_area_balance_targets
from model import BalanceModel
from train_config import GENERATION_VARIABLE_FLOAT_FIELDS


def generation_values() -> dict[str, torch.Tensor]:
    values = {}
    for name in GENERATION_VARIABLE_FLOAT_FIELDS:
        if name.startswith("log_"):
            continue
        if name.startswith("target_area_probability_"):
            continue
        values[name] = torch.full((2,), 0.25)
    values["temperature"] = torch.tensor([300.0, 0.03])
    values["proposal_temperature"] = torch.tensor([100.0, 0.01])
    for area in range(6):
        values[f"target_area_rooms_{area}"] = torch.full((2,), 12.0 * (area + 1))
    values["force_ship_in_crateria"] = torch.tensor([0.0, 1.0])
    return values


def test_construction_defines_every_field_and_preserves_inputs() -> None:
    values = generation_values()
    original = {name: value.clone() for name, value in values.items()}
    encoded = GenerationVariableFloatsFeature.construct_tensor(values, num_rooms=252)
    assert encoded.shape == (2, len(GENERATION_VARIABLE_FLOAT_FIELDS))
    assert encoded.dtype == torch.float32
    for index, name in enumerate(GENERATION_VARIABLE_FLOAT_FIELDS):
        if name.startswith("log_"):
            expected = values[name.removeprefix("log_")].log()
        elif name.startswith("target_area_probability_"):
            expected = values[name.replace("probability", "rooms")] / 252
        else:
            expected = values[name]
        torch.testing.assert_close(encoded[:, index], expected)
    for name in values:
        torch.testing.assert_close(values[name], original[name])
    large = {name: value.clone() for name, value in values.items()}
    for area in range(6):
        large[f"target_area_rooms_{area}"] *= 2
    torch.testing.assert_close(
        encoded, GenerationVariableFloatsFeature.construct_tensor(large, num_rooms=504)
    )


def test_main_and_balance_models_consume_stored_features_directly() -> None:
    encoded = GenerationVariableFloatsFeature.construct_tensor(generation_values(), num_rooms=252)
    main_feature = GenerationVariableFloatsFeature()
    features = SimpleNamespace(global_features=SimpleNamespace(generation_variable_floats=encoded))
    balance = BalanceModel(
        left_count=1,
        right_count=1,
        up_count=0,
        down_count=0,
        door_output_variant_idx=torch.tensor([0, 1]),
        door_room_idx=torch.tensor([0, 1]),
        door_variant_compatibility=torch.ones((2, 2), dtype=torch.bool),
        room_connection_variant_idx=torch.tensor([0, 1]),
        num_room_connection_variants=2,
        toilet_compatibility=torch.zeros(2, dtype=torch.bool),
        hidden_width=4,
        num_layers=1,
    )
    capture = Mock(return_value=None)
    handles = [
        network.register_forward_pre_hook(capture)
        for network in (balance.door_net, balance.toilet_net, balance.area_net)
    ]
    try:
        predictions = balance(encoded)
    finally:
        for handle in handles:
            handle.remove()
    assert capture.call_count == 3
    for call in capture.call_args_list:
        torch.testing.assert_close(call.args[1][0], encoded)
    torch.testing.assert_close(main_feature(features, torch.float32), encoded)
    torch.testing.assert_close(main_feature(features, torch.bfloat16), encoded.bfloat16())
    assert torch.isfinite(predictions.room_area).all()


def test_missing_input_and_unhandled_schema_field_are_errors() -> None:
    values = generation_values()
    with unittest.TestCase().assertRaisesRegex(ValueError, "num_rooms"):
        GenerationVariableFloatsFeature.construct_tensor(values, num_rooms=0)
    del values["reward_door"]
    with unittest.TestCase().assertRaises(KeyError):
        GenerationVariableFloatsFeature.construct_tensor(values, num_rooms=252)
    with patch(
        "features.GENERATION_VARIABLE_FLOAT_FIELDS", (*GENERATION_VARIABLE_FLOAT_FIELDS, "new_field")
    ):
        with unittest.TestCase().assertRaisesRegex(RuntimeError, "every schema field"):
            GenerationVariableFloatsFeature.construct_tensor(generation_values(), num_rooms=252)


def test_experience_round_trip_and_balance_target_reconstruction() -> None:
    variables = GenerationVariableFloatsFeature.construct_tensor(
        generation_values(), num_rooms=252
    )
    data = EpisodeData(
        actions=Actions(
            room_idx=torch.zeros((2, 252), dtype=torch.uint8),
            room_x=torch.zeros((2, 252), dtype=torch.uint8),
            room_y=torch.zeros((2, 252), dtype=torch.uint8),
            room_area=torch.zeros((2, 252), dtype=torch.uint8),
        ),
        temperature=torch.tensor([300.0, 0.03]),
        recommended_candidates=torch.ones(2, dtype=torch.int64),
        generation_variable_floats=variables,
    )
    with tempfile.TemporaryDirectory() as directory:
        storage = ExperienceStorage(num_rooms=252, data_path=directory, episodes_per_file=2)
        storage.store(data)
        path = Path(directory) / "0.safetensors"
        _, restored = load_balance_experience(path, num_rooms=252)
        torch.testing.assert_close(restored, variables)
        targets = generation_area_balance_targets([{} for _ in range(252)], restored)
        expected_counts = torch.arange(1, 7).float().mul(12).expand(2, -1)
        torch.testing.assert_close(targets.effective_area_rooms, expected_counts)
        torch.testing.assert_close(targets.probability[:, 0], expected_counts / 252)
        legacy_path = Path(directory) / "legacy.safetensors"
        safetensors.torch.save_file({}, legacy_path, metadata={"format": "map-gen-experience-v2"})
        with unittest.TestCase().assertRaisesRegex(ValueError, "unsupported experience format"):
            load_balance_experience(legacy_path, num_rooms=252)


if __name__ == "__main__":
    test_construction_defines_every_field_and_preserves_inputs()
    test_main_and_balance_models_consume_stored_features_directly()
    test_missing_input_and_unhandled_schema_field_are_errors()
    test_experience_round_trip_and_balance_target_reconstruction()
