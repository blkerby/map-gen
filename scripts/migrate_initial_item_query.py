#!/usr/bin/env python3
"""Upgrade a v27 checkpoint to the v28 initial-item frontier query.

  conda run -n map-gen python scripts/migrate_initial_item_query.py SOURCE OUTPUT

Existing heads, balance models, and optimizer moments are preserved. New feature
columns start at zero, and the replacement initial-item head starts untrained.
Loss/reward settings, run identity, counters, and the source file are preserved.
Re-export inference models after migration.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from safetensors import safe_open

from env import Engine
from model import Float32Linear, FrontierModel
from model_loading import frontier_model_kwargs, without_prefix
from scripts.checkpoint_migration import add_model_parameters
from scripts.migrate_area_connections import expand_model_inputs, expand_optimizer_inputs
from scripts.migrate_return_distance import write_output
from train import (
    TRAINING_CHECKPOINT_FORMAT,
    create_main_optimizer,
    named_checkpoint_optimizers,
    validate_checkpoint_metadata,
)
from train_config import Config, instantiate_scheduleable_config, validate_config

SOURCE_FORMAT = "map-gen-training-session-checkpoint-v27"
TARGET_FORMAT = "map-gen-training-session-checkpoint-v28"
OLD_HEAD = "initial_item_output."
NEW_HEAD = "initial_item_query_output."


def legacy_parameter_layout(model: FrontierModel) -> torch.nn.Module:
    """Reconstruct the old parameter order for decoding optimizer state IDs."""
    legacy = torch.nn.Module()
    for name, module in model.named_children():
        if name == "initial_item_query_output":
            legacy.add_module("initial_item_output", Float32Linear(model.embedding_width, 1))
        else:
            legacy.add_module(name, module)
    return legacy


def remove_old_head_optimizer_state(tensors, metadata, model, optimizer) -> None:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    parts = named_checkpoint_optimizers(optimizer)
    if set(json.loads(metadata["optimizer_names"])) != set(parts):
        raise ValueError("unexpected main optimizer parts")
    for part_name, part in parts.items():
        key = f"optimizer_{part_name}_param_groups"
        groups = json.loads(metadata[key])
        removed = set()
        for saved, current in zip(groups, part.param_groups, strict=True):
            kept = []
            for parameter_id, parameter in zip(saved["params"], current["params"], strict=True):
                if names[id(parameter)].startswith(OLD_HEAD):
                    removed.add(parameter_id)
                else:
                    kept.append(parameter_id)
            saved["params"] = kept
        metadata[key] = json.dumps(groups)
        scalar_key = f"optimizer_{part_name}_scalar_state"
        scalar = json.loads(metadata[scalar_key])
        for parameter_id in removed:
            scalar.pop(str(parameter_id), None)
            prefix = f"optimizer.{part_name}.state.{parameter_id}."
            for name in [name for name in tensors if name.startswith(prefix)]:
                del tensors[name]
        metadata[scalar_key] = json.dumps(scalar)


def migrate_checkpoint(source: Path, output: Path) -> dict:
    if TRAINING_CHECKPOINT_FORMAT != TARGET_FORMAT:
        raise ValueError("this migration requires the v28 model implementation")
    if output.exists():
        raise FileExistsError(output)
    with safe_open(source, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata()
        if metadata is None or metadata["format"] != SOURCE_FORMAT:
            raise ValueError(f"expected a {SOURCE_FORMAT} checkpoint")
        tensors = {name: checkpoint.get_tensor(name) for name in checkpoint.keys()}
    for prefix in ("main_model", "ema_model", "balance_model"):
        normalized = without_prefix(tensors, prefix)
        for key in [key for key in tensors if key.startswith(f"{prefix}.")]:
            del tensors[key]
        tensors.update({f"{prefix}.{key}": value for key, value in normalized.items()})
    config = Config.model_validate_json(metadata["config"])
    validate_config(config)
    current = instantiate_scheduleable_config(config, int(metadata["num_episodes"]))
    room_path = Path(config.room_set)
    if not room_path.is_absolute():
        room_path = Path(__file__).resolve().parents[1] / room_path
    rooms = json.loads(room_path.read_text())
    engine = Engine(
        rooms, current.features, current.generation.min_area_size, current.generation.max_area_size
    )
    model = FrontierModel(**frontier_model_kwargs(current, rooms, engine))
    legacy = legacy_parameter_layout(model)
    remove_old_head_optimizer_state(
        tensors, metadata, legacy, create_main_optimizer(legacy, current.optimizer)
    )
    columns = {}
    if current.features.generation_variable_floats:
        width = model.global_mlp.in_features
        columns["global_mlp.weight"] = list(range(width - 2, width))
        if current.features.frontier_mask:
            width = model.frontier_node_mlp.in_features
            columns["frontier_node_mlp.weight"] = list(range(width - 6, width))
    for prefix in ("main_model", "ema_model"):
        for name, shape in (("weight", (1, model.embedding_width)), ("bias", (1,))):
            key = f"{prefix}.{OLD_HEAD}{name}"
            if tensors[key].shape != shape:
                raise ValueError(f"unexpected {key} shape: {tensors[key].shape}")
            del tensors[key]
        expand_model_inputs(tensors, model, prefix, columns)
        added = add_model_parameters(tensors, model, prefix, NEW_HEAD)
    expand_optimizer_inputs(
        tensors,
        metadata,
        model,
        create_main_optimizer(model, current.optimizer),
        "optimizer",
        columns,
        added,
    )
    report = {
        "source": str(source),
        "source_format": SOURCE_FORMAT,
        "output_format": TARGET_FORMAT,
        "added_input_columns": columns,
        "reset_head": NEW_HEAD.removesuffix("."),
    }
    metadata["format"] = TARGET_FORMAT
    metadata["initial_item_query_migration"] = json.dumps(report)
    validate_checkpoint_metadata(output, metadata)
    write_output(tensors, metadata, output)
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
