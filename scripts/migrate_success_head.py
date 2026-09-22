#!/usr/bin/env python3
"""Migrate a v20 checkpoint to v21, preserving weights and optimizer history.

The success head starts with zero logits. The new reward_success input column
and its optimizer moments start at zero, preserving all existing predictions.
Existing v3 replay files are supported by the current replay loader.
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

from env import Engine
from features import FeatureContext, GLOBAL_FEATURES
from features import GenerationVariableFloatsFeature
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
from train_config import (
    Config, GENERATION_VARIABLE_FLOAT_FIELDS, instantiate_scheduleable_config, validate_config,
)

SOURCE_FORMAT = "map-gen-training-session-checkpoint-v20"
TARGET_FORMAT = "map-gen-training-session-checkpoint-v21"
SUCCESS_PARAMETERS = {"success_output.weight", "success_output.bias"}


def insert_zero_column(value: torch.Tensor, column: int) -> torch.Tensor:
    if value.ndim != 2 or not 0 <= column <= value.shape[1]:
        raise ValueError(f"cannot insert column {column} into shape {tuple(value.shape)}")
    return torch.cat(
        (value[:, :column], value.new_zeros((value.shape[0], 1)), value[:, column:]), dim=1
    )


def success_input_columns(config: Config, rooms: list, engine: Engine) -> tuple[dict, dict]:
    variable_column = GENERATION_VARIABLE_FLOAT_FIELDS.index("reward_success")
    main_columns = {}
    if config.features.generation_variable_floats:
        kwargs = frontier_model_kwargs(config, rooms, engine)
        context = FeatureContext(
            features=config.features,
            output_metadata=engine.output_metadata,
            num_rooms=len(rooms),
            num_room_parts=engine.output_metadata.num_room_parts,
            num_connection_outputs=len(engine.output_metadata.connection),
            door_counts=kwargs["door_counts"],
            frontier_window_area=config.generation.frontier_window_size**2,
            area_bounding_box_width=config.generation.area_bounding_box_width,
            area_bounding_box_height=config.generation.area_bounding_box_height,
            max_area_size=config.generation.max_area_size,
        )
        offset = 0
        for feature_class in GLOBAL_FEATURES:
            if feature_class is GenerationVariableFloatsFeature:
                break
            if feature_class.is_enabled(config.features):
                offset += feature_class.tensor_width(context)
        main_columns["global_mlp.weight"] = offset + variable_column
    balance_columns = {
        f"{family}_net.0.weight": variable_column
        for family in ("door", "toilet", "area", "order")
    }
    return main_columns, balance_columns


def migrate_model(tensors, model, prefix: str, added: set[str], columns: dict) -> None:
    old = without_prefix(tensors, prefix)
    expected = model.state_dict()
    if set(old) != set(expected) - added:
        raise ValueError(f"unexpected source parameter keys for {prefix}")
    for name, column in columns.items():
        old[name] = insert_zero_column(old[name], column)
        tensors[f"{prefix}.{name}"] = old[name]
    for name in added:
        old[name] = torch.zeros_like(expected[name])
        tensors[f"{prefix}.{name}"] = old[name]
    model.load_state_dict(old, strict=True)


def migrate_optimizer(tensors, metadata, model, optimizer, prefix, added, columns) -> dict:
    expanded_columns = {}
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    parts = named_checkpoint_optimizers(optimizer)
    if set(json.loads(metadata[f"{prefix}_names"])) != set(parts):
        raise ValueError(f"unexpected optimizer parts for {prefix}")
    for part_name, part in parts.items():
        key = f"{prefix}_{part_name}_param_groups"
        saved_groups = json.loads(metadata[key])
        next_id = max(i for group in saved_groups for i in group["params"]) + 1
        for saved, current in zip(saved_groups, part.param_groups, strict=True):
            old_parameters = [p for p in current["params"] if names[id(p)] not in added]
            if len(old_parameters) != len(saved["params"]):
                raise ValueError(f"unexpected parameter count for {prefix}.{part_name}")
            old_ids = iter(saved["params"])
            ids = []
            for parameter in current["params"]:
                name = names[id(parameter)]
                if name in added:
                    ids.append(next_id)
                    next_id += 1
                    continue
                saved_id = next(old_ids)
                ids.append(saved_id)
                if name in columns:
                    state_prefix = f"{prefix}.{part_name}.state.{saved_id}."
                    for tensor_name in list(tensors):
                        value = tensors[tensor_name]
                        if tensor_name.startswith(state_prefix) and value.ndim:
                            tensors[tensor_name] = insert_zero_column(value, columns[name])
                            expanded_columns[tensor_name] = columns[name]
            saved["params"] = ids
        metadata[key] = json.dumps(saved_groups)
    load_named_optimizer_checkpoint_state(optimizer, tensors, metadata, prefix)
    for part in parts.values():
        for parameter, state in part.state.items():
            for name, value in state.items():
                if torch.is_tensor(value) and value.ndim and value.shape != parameter.shape:
                    raise ValueError(f"wrong shape for {prefix} optimizer moment {name}")
        for group in part.param_groups:
            for parameter in group["params"]:
                if names[id(parameter)] in added and parameter in part.state:
                    raise ValueError("new success head unexpectedly has optimizer history")
    return expanded_columns


def verify_preserved_tensors(source: Path, tensors: dict, columns: dict) -> dict:
    unchanged = 0
    expanded = 0
    with safe_open(source, framework="pt", device="cpu") as checkpoint:
        for name in checkpoint.keys():
            old = checkpoint.get_tensor(name)
            new = tensors[name]
            if old.shape == new.shape:
                if not torch.equal(old, new):
                    raise ValueError(f"existing tensor changed: {name}")
                unchanged += 1
                continue
            column = columns[name]
            if not torch.equal(new, insert_zero_column(old, column)):
                raise ValueError(f"expanded tensor did not preserve existing values: {name}")
            expanded += 1
    return {"unchanged_tensors": unchanged, "expanded_tensors": expanded}


def migrate_checkpoint(source: Path, output: Path) -> dict:
    if TRAINING_CHECKPOINT_FORMAT != TARGET_FORMAT:
        raise ValueError("this migration requires the v21 model implementation")
    if output.exists():
        raise FileExistsError(output)
    with safe_open(source, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata()
        if metadata is None or metadata["format"] != SOURCE_FORMAT:
            raise ValueError(f"expected {SOURCE_FORMAT}")
        tensors = {name: checkpoint.get_tensor(name) for name in checkpoint.keys()}
    config_data = json.loads(metadata["config"])
    if "reward_success" in config_data["generation"] or "success_weight" in config_data["train"]:
        raise ValueError("source already contains success-head configuration")
    config_data["generation"]["reward_success"] = 0.0
    config_data["train"]["success_weight"] = 1.0
    config = Config.model_validate(config_data)
    validate_config(config)
    step_config = instantiate_scheduleable_config(config, int(metadata["num_episodes"]))
    room_path = Path(config.room_set)
    if not room_path.is_absolute():
        room_path = Path(__file__).resolve().parents[1] / room_path
    rooms = json.loads(room_path.read_text())
    engine = Engine(rooms, step_config.features, step_config.generation.min_area_size,
                    step_config.generation.max_area_size)
    main_columns, balance_columns = success_input_columns(step_config, rooms, engine)
    model = FrontierModel(**frontier_model_kwargs(step_config, rooms, engine))
    migrate_model(tensors, model, "main_model", SUCCESS_PARAMETERS, main_columns)
    optimizer = create_main_optimizer(model, step_config.optimizer)
    optimizer_columns = migrate_optimizer(
        tensors, metadata, model, optimizer, "optimizer", SUCCESS_PARAMETERS, main_columns
    )
    migrate_model(tensors, model, "ema_model", SUCCESS_PARAMETERS, main_columns)
    balance = create_balance_model(step_config, rooms, engine, torch.device("cpu"))
    migrate_model(tensors, balance, "balance_model", set(), balance_columns)
    balance_optimizer = create_adam_optimizer(balance.parameters(), step_config.balance_optimizer)
    optimizer_columns.update(migrate_optimizer(
        tensors, metadata, balance, balance_optimizer, "balance_optimizer", set(), balance_columns
    ))
    all_columns = {
        **{f"{prefix}.{key}": value for prefix in ("main_model", "ema_model")
           for key, value in main_columns.items()},
        **{f"balance_model.{key}": value for key, value in balance_columns.items()},
        **optimizer_columns,
    }
    report = {
        "source": str(source),
        "source_format": SOURCE_FORMAT,
        "output_format": TARGET_FORMAT,
        "reward_success": 0.0,
        "success_weight": 1.0,
        "initial_success_probability": 0.5,
        "inserted_columns": all_columns,
        "preserved_replay_files": int(metadata["experience_num_files"]),
        "replay_compatibility": "v3 files load with reward_success=0",
        **verify_preserved_tensors(source, tensors, all_columns),
    }
    metadata["format"] = TARGET_FORMAT
    metadata["config"] = config.model_dump_json()
    metadata["success_head_migration"] = json.dumps(report)
    validate_checkpoint_metadata(output, metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".safetensors", delete=False) as temp:
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
