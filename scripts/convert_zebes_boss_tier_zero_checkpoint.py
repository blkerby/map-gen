#!/usr/bin/env python3
import argparse
import copy
import gc
import json
import os
import sys
from pathlib import Path

import safetensors.torch
import torch
from safetensors import safe_open


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from env import AREA_COUNT, VANILLA_AREA_CONSTRAINT_COUNT, Engine  # noqa: E402
from model import FrontierModel  # noqa: E402
from model_loading import frontier_model_kwargs, without_prefix  # noqa: E402
from train import (  # noqa: E402
    create_main_optimizer,
    load_named_optimizer_checkpoint_state,
    named_checkpoint_optimizers,
)
from train_config import (  # noqa: E402
    Config,
    GENERATION_VARIABLE_FLOAT_FIELDS,
    instantiate_scheduleable_config,
)


SOURCE_FORMAT = "map-gen-training-session-checkpoint-v8"
TARGET_FORMAT = "map-gen-training-session-checkpoint-v9"
MODEL_PREFIXES = ("main_model", "ema_model")
AFFECTED_PARAMETERS = {
    "global_mlp.weight",
    "maridia_water_output.weight",
    "maridia_water_output.bias",
    "norfair_heat_output.weight",
    "norfair_heat_output.bias",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a Zebes training checkpoint after changing Draygon water and Ridley heat "
            "from tier 1 to tier 0."
        ),
    )
    parser.add_argument("input", type=Path, help="Version 8 training checkpoint.")
    parser.add_argument("output", type=Path, help="New version 9 training checkpoint.")
    return parser.parse_args()


def special_room_idx(rooms: list[dict], special_type: str) -> int:
    matches = [i for i, room in enumerate(rooms) if room.get("special_type") == special_type]
    if len(matches) != 1:
        raise ValueError(f"expected one {special_type!r} room, found {len(matches)}")
    return matches[0]


def remove_indices(tensor: torch.Tensor, dimension: int, removed: list[int]) -> torch.Tensor:
    removed_set = set(removed)
    kept = [i for i in range(tensor.shape[dimension]) if i not in removed_set]
    return torch.index_select(tensor, dimension, torch.tensor(kept, dtype=torch.int64))


def optimizer_parameter_ids(
    model: FrontierModel,
    config: Config,
) -> dict[str, tuple[str, int]]:
    initial_config = instantiate_scheduleable_config(config, 0)
    optimizer = create_main_optimizer(model, config.optimizer, initial_config.optimizer)
    name_by_identity = {id(parameter): name for name, parameter in model.named_parameters()}
    result = {}
    for optimizer_name, part in named_checkpoint_optimizers(optimizer).items():
        state_dict = part.state_dict()
        for live_group, saved_group in zip(
            part.param_groups,
            state_dict["param_groups"],
            strict=True,
        ):
            for parameter, parameter_id in zip(
                live_group["params"],
                saved_group["params"],
                strict=True,
            ):
                parameter_name = name_by_identity[id(parameter)]
                if parameter_name in AFFECTED_PARAMETERS:
                    result[parameter_name] = (optimizer_name, parameter_id)
    if result.keys() != AFFECTED_PARAMETERS:
        raise ValueError(
            f"found optimizer IDs for {sorted(result)}, expected {sorted(AFFECTED_PARAMETERS)}"
        )
    return result


def lookahead_removed_columns(
    config: Config, metadata, water_slot: int, heat_slot: int
) -> list[int]:
    if config.features.lookahead_outcomes <= 0:
        return []
    lookahead_start = (
        metadata.num_room_connection_variants * int(config.features.inventory)
        + int(config.features.temperature)
        + int(config.features.recommended_candidates)
        + len(GENERATION_VARIABLE_FLOAT_FIELDS) * int(config.features.generation_variable_floats)
    )
    preferred_area_start = (
        lookahead_start
        + config.features.lookahead_outcomes
        + 2 * len(metadata.connection)
        + 6
        + 2 * VANILLA_AREA_CONSTRAINT_COUNT
        + 2 * AREA_COUNT * 3
    )
    water_columns = [preferred_area_start + 2 * water_slot + offset for offset in range(2)]
    heat_start = preferred_area_start + 2 * len(metadata.maridia_water_room_idx)
    heat_columns = [heat_start + 2 * heat_slot + offset for offset in range(2)]
    return water_columns + heat_columns


def transform_parameter(
    name: str,
    tensor: torch.Tensor,
    water_slot: int,
    heat_slot: int,
    global_columns: list[int],
) -> torch.Tensor:
    if name == "global_mlp.weight":
        return remove_indices(tensor, 1, global_columns)
    if name.startswith("maridia_water_output."):
        return remove_indices(tensor, 0, [water_slot])
    if name.startswith("norfair_heat_output."):
        return remove_indices(tensor, 0, [heat_slot])
    raise ValueError(f"no conversion defined for parameter {name!r}")


def transform_tensors(
    tensors: dict[str, torch.Tensor],
    optimizer_ids: dict[str, tuple[str, int]],
    water_slot: int,
    heat_slot: int,
    global_columns: list[int],
) -> dict[str, torch.Tensor]:
    result = dict(tensors)
    transformed = set()
    for prefix in MODEL_PREFIXES:
        for name in AFFECTED_PARAMETERS:
            key = f"{prefix}.{name}"
            result[key] = transform_parameter(
                name,
                tensors[key],
                water_slot,
                heat_slot,
                global_columns,
            )
            transformed.add(key)
        for family, slot in (("maridia_water", water_slot), ("norfair_heat", heat_slot)):
            key = f"{prefix}.{family}_tier_mask"
            result[key] = remove_indices(tensors[key], 0, [slot])
            transformed.add(key)

    for name, (optimizer_name, parameter_id) in optimizer_ids.items():
        prefix = f"optimizer.{optimizer_name}.state.{parameter_id}."
        state_keys = [key for key in tensors if key.startswith(prefix)]
        shaped_state_keys = [key for key in state_keys if tensors[key].ndim > 0]
        if not shaped_state_keys:
            raise ValueError(f"checkpoint has no shaped optimizer state for {name!r}")
        model_shape = tensors[f"main_model.{name}"].shape
        for key in shaped_state_keys:
            if tensors[key].shape != model_shape:
                raise ValueError(
                    f"optimizer tensor {key} has shape {tuple(tensors[key].shape)}, "
                    f"expected {tuple(model_shape)}"
                )
            result[key] = transform_parameter(
                name,
                tensors[key],
                water_slot,
                heat_slot,
                global_columns,
            )
            transformed.add(key)
    print(f"Converted {len(transformed)} model and optimizer tensors.")
    return result


def verify_converted_checkpoint(
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, str],
    config: Config,
    rooms: list[dict],
) -> None:
    engine = Engine(
        rooms,
        config.features,
        config.generation.min_area_size,
        config.generation.max_area_size,
    )
    model = FrontierModel(**frontier_model_kwargs(config, rooms, engine))
    model.load_state_dict(without_prefix(tensors, "main_model"))
    model.load_state_dict(without_prefix(tensors, "ema_model"))
    initial_config = instantiate_scheduleable_config(config, 0)
    optimizer = create_main_optimizer(model, config.optimizer, initial_config.optimizer)
    load_named_optimizer_checkpoint_state(optimizer, tensors, metadata, "optimizer")
    parameters = dict(model.named_parameters())
    affected_parameter_ids = {id(parameters[name]) for name in AFFECTED_PARAMETERS}
    for part in named_checkpoint_optimizers(optimizer).values():
        for parameter, state in part.state.items():
            if id(parameter) not in affected_parameter_ids:
                continue
            for value in state.values():
                if torch.is_tensor(value) and value.ndim > 0:
                    if value.shape != parameter.shape:
                        raise ValueError(
                            f"converted optimizer state shape {tuple(value.shape)} does not match "
                            f"parameter shape {tuple(parameter.shape)}"
                        )


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if input_path == output_path:
        raise ValueError("input and output checkpoint paths must differ")
    if output_path.exists():
        raise FileExistsError(f"output checkpoint already exists: {output_path}")

    with safe_open(input_path, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata()
        if metadata is None or metadata.get("format") != SOURCE_FORMAT:
            actual = None if metadata is None else metadata.get("format")
            raise ValueError(f"expected checkpoint format {SOURCE_FORMAT!r}, found {actual!r}")
        if "config" not in metadata:
            raise ValueError("checkpoint metadata field 'config' is missing")
        tensors = {name: checkpoint.get_tensor(name) for name in checkpoint.keys()}

    config = Config.model_validate_json(metadata["config"])
    room_path = REPO_ROOT / config.room_set
    rooms = json.loads(room_path.read_text())
    draygon_idx = special_room_idx(rooms, "draygon_boss")
    ridley_idx = special_room_idx(rooms, "ridley_boss")
    if rooms[draygon_idx].get("water") != 0 or rooms[ridley_idx].get("heat") != 0:
        raise ValueError("room definitions must have Draygon water=0 and Ridley heat=0")

    old_rooms = copy.deepcopy(rooms)
    old_rooms[draygon_idx]["water"] = 1
    old_rooms[ridley_idx]["heat"] = 1
    old_engine = Engine(
        old_rooms,
        config.features,
        config.generation.min_area_size,
        config.generation.max_area_size,
    )
    old_metadata = old_engine.output_metadata
    water_slot = old_metadata.maridia_water_room_idx.index(draygon_idx)
    heat_slot = old_metadata.norfair_heat_room_idx.index(ridley_idx)
    global_columns = lookahead_removed_columns(
        config,
        old_metadata,
        water_slot,
        heat_slot,
    )
    old_model = FrontierModel(**frontier_model_kwargs(config, old_rooms, old_engine))
    optimizer_ids = optimizer_parameter_ids(old_model, config)
    del old_model, old_engine
    gc.collect()

    converted = transform_tensors(
        tensors,
        optimizer_ids,
        water_slot,
        heat_slot,
        global_columns,
    )
    converted_metadata = dict(metadata)
    converted_metadata["format"] = TARGET_FORMAT
    converted_metadata["converted_from_format"] = SOURCE_FORMAT
    converted_metadata["tier_zero_rooms"] = json.dumps(
        ["Draygon's Room:water", "Ridley's Room:heat"]
    )
    verify_converted_checkpoint(converted, converted_metadata, config, rooms)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    if temporary_path.exists():
        raise FileExistsError(f"temporary checkpoint already exists: {temporary_path}")
    safetensors.torch.save_file(converted, temporary_path, metadata=converted_metadata)
    os.replace(temporary_path, output_path)
    print(f"Wrote converted checkpoint: {output_path}")


if __name__ == "__main__":
    main()
