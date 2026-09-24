#!/usr/bin/env python3
"""Upgrade a v21 checkpoint with a prediction-preserving per-room area input.

Usage: conda run -n map-gen python scripts/migrate_room_area.py SOURCE OUTPUT

Both main and EMA input weights gain six zero columns per room. Existing
optimizer IDs, moments, counters, and replay history are preserved; the new
moment columns start at zero. The source is never overwritten.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from env import AREA_COUNT, Engine
from features import GLOBAL_FEATURES, RoomAreaFeature
from model import FrontierModel
from model_loading import create_balance_model, frontier_model_kwargs, without_prefix
from train import (
    TRAINING_CHECKPOINT_FORMAT,
    create_adam_optimizer,
    create_main_optimizer,
    load_named_optimizer_checkpoint_state,
    named_checkpoint_optimizers,
    validate_checkpoint_metadata,
)
from train_config import Config, instantiate_scheduleable_config, validate_config

SOURCE_FORMAT = "map-gen-training-session-checkpoint-v21"
TARGET_FORMAT = "map-gen-training-session-checkpoint-v22"
INPUT_WEIGHT = "global_mlp.weight"


def append_zero_columns(value: torch.Tensor, count: int) -> torch.Tensor:
    if value.ndim != 2 or count <= 0:
        raise ValueError("expected a matrix and a positive number of columns")
    return torch.cat((value, value.new_zeros((value.shape[0], count))), dim=1)


def expand_model_input(
    tensors: dict[str, torch.Tensor], model: FrontierModel, prefix: str, width: int
) -> str:
    state = without_prefix(tensors, prefix)
    keys = [
        key
        for key in (f"{prefix}.{INPUT_WEIGHT}", f"{prefix}._orig_mod.{INPUT_WEIGHT}")
        if key in tensors
    ]
    if len(keys) != 1:
        raise ValueError(f"expected exactly one input weight for {prefix}")
    state[INPUT_WEIGHT] = append_zero_columns(state[INPUT_WEIGHT], width)
    model.load_state_dict(state, strict=True)
    tensors[keys[0]] = state[INPUT_WEIGHT]
    return keys[0]


def validate_optimizer_shapes(optimizer, prefix: str) -> None:
    for part in named_checkpoint_optimizers(optimizer).values():
        for parameter, state in part.state.items():
            for name, value in state.items():
                if torch.is_tensor(value) and value.ndim and value.shape != parameter.shape:
                    raise ValueError(f"incorrect optimizer state shape for {prefix}: {name}")


def expand_optimizer_input(
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, str],
    model: FrontierModel,
    optimizer,
    width: int,
) -> list[str]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    parts = named_checkpoint_optimizers(optimizer)
    if set(json.loads(metadata["optimizer_names"])) != set(parts):
        raise ValueError("unexpected optimizer parts")
    expanded = []
    for part_name, part in parts.items():
        saved_groups = json.loads(metadata[f"optimizer_{part_name}_param_groups"])
        for saved, current in zip(saved_groups, part.param_groups, strict=True):
            for saved_id, parameter in zip(saved["params"], current["params"], strict=True):
                if names[id(parameter)] != INPUT_WEIGHT:
                    continue
                state_prefix = f"optimizer.{part_name}.state.{saved_id}."
                for key in list(tensors):
                    value = tensors[key]
                    if not key.startswith(state_prefix) or not value.ndim:
                        continue
                    if tuple(value.shape) != (parameter.shape[0], parameter.shape[1] - width):
                        raise ValueError(f"unexpected source optimizer moment shape: {key}")
                    tensors[key] = append_zero_columns(value, width)
                    expanded.append(key)
    load_named_optimizer_checkpoint_state(optimizer, tensors, metadata, "optimizer")
    validate_optimizer_shapes(optimizer, "optimizer")
    return expanded


def verify_preservation(source: Path, tensors: dict, expanded: list[str], width: int) -> int:
    expanded_keys = set(expanded)
    with safe_open(source, framework="pt", device="cpu") as checkpoint:
        if set(checkpoint.keys()) != set(tensors):
            raise ValueError("migration unexpectedly changed checkpoint tensor keys")
        for key in checkpoint.keys():
            expected = checkpoint.get_tensor(key)
            if key in expanded_keys:
                expected = append_zero_columns(expected, width)
            if not torch.equal(tensors[key], expected):
                raise ValueError(f"migration did not preserve tensor {key}")
    return len(tensors) - len(expanded_keys)


def migrate_checkpoint(source: Path, output: Path) -> dict:
    if TRAINING_CHECKPOINT_FORMAT != TARGET_FORMAT:
        raise ValueError("this migration requires the v22 model implementation")
    if GLOBAL_FEATURES[-1] is not RoomAreaFeature:
        raise ValueError("migration requires room-area inputs to be appended last")
    if output.exists():
        raise FileExistsError(output)
    with safe_open(source, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata()
        if metadata is None or metadata["format"] != SOURCE_FORMAT:
            raise ValueError(f"expected {SOURCE_FORMAT}")
        tensors = {key: checkpoint.get_tensor(key) for key in checkpoint.keys()}
    config_data = json.loads(metadata["config"])
    if "room_area" in config_data["features"]:
        raise ValueError("source already contains room-area feature configuration")
    config_data["features"]["room_area"] = True
    config = Config.model_validate(config_data)
    validate_config(config)
    step_config = instantiate_scheduleable_config(config, int(metadata["num_episodes"]))
    room_path = Path(config.room_set)
    if not room_path.is_absolute():
        room_path = Path(__file__).resolve().parents[1] / room_path
    rooms = json.loads(room_path.read_text())
    engine = Engine(
        rooms,
        step_config.features,
        step_config.generation.min_area_size,
        step_config.generation.max_area_size,
    )
    width = len(rooms) * AREA_COUNT
    model = FrontierModel(**frontier_model_kwargs(step_config, rooms, engine))
    expanded = [expand_model_input(tensors, model, "main_model", width)]
    optimizer = create_main_optimizer(model, step_config.optimizer)
    expanded.extend(expand_optimizer_input(tensors, metadata, model, optimizer, width))
    expanded.append(expand_model_input(tensors, model, "ema_model", width))
    balance = create_balance_model(step_config, rooms, engine, torch.device("cpu"))
    balance.load_state_dict(without_prefix(tensors, "balance_model"), strict=True)
    balance_optimizer = create_adam_optimizer(balance.parameters(), step_config.balance_optimizer)
    load_named_optimizer_checkpoint_state(
        balance_optimizer, tensors, metadata, "balance_optimizer"
    )
    validate_optimizer_shapes(balance_optimizer, "balance_optimizer")
    report = {
        "source": str(source),
        "source_format": SOURCE_FORMAT,
        "output_format": TARGET_FORMAT,
        "added_input_columns": width,
        "expanded_tensors": expanded,
        "unchanged_tensors": verify_preservation(source, tensors, expanded, width),
        "preserved_replay_files": int(metadata["experience_num_files"]),
        "replay_compatibility": "existing room_area actions reconstruct the new feature",
    }
    metadata["format"] = TARGET_FORMAT
    metadata["config"] = config.model_dump_json()
    metadata["room_area_migration"] = json.dumps(report)
    validate_checkpoint_metadata(output, metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent, suffix=".safetensors", delete=False
    ) as temp:
        temporary = Path(temp.name)
    try:
        save_file(tensors, temporary, metadata=metadata)
        os.link(temporary, output)
    finally:
        temporary.unlink()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(migrate_checkpoint(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
