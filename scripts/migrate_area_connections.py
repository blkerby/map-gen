#!/usr/bin/env python3
"""Migrate a v23 checkpoint to the 15 area-connection predictions (v24).

Usage:
  conda run -n map-gen python scripts/migrate_area_connections.py SOURCE OUTPUT

The new reward is zero and BCE weight is one. New input columns and the new
prediction head start at zero; old weights, optimizer history, counters, and Aim
identity are preserved. The source is never modified and output must not exist.
Existing v4 replay files can be reused; training imports their new reward as zero.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from env import AREA_PAIR_COUNT, Engine
from features import (
    GLOBAL_FEATURES,
    FeatureContext,
    GenerationVariableFloatsFeature,
    LookaheadFeature,
)
from model import FrontierModel
from model_loading import create_balance_model, frontier_model_kwargs, without_prefix
from scripts.checkpoint_migration import add_model_parameters, extend_optimizer_groups
from train import (
    TRAINING_CHECKPOINT_FORMAT,
    create_adam_optimizer,
    create_main_optimizer,
    named_checkpoint_optimizers,
    validate_checkpoint_metadata,
)
from train_config import (
    Config,
    GENERATION_VARIABLE_FLOAT_FIELDS,
    instantiate_scheduleable_config,
    validate_config,
)

SOURCE_FORMAT = "map-gen-training-session-checkpoint-v23"
TARGET_FORMAT = "map-gen-training-session-checkpoint-v24"
NEW_HEAD = "area_connection_output."
REWARD_FIELD = "reward_area_distinct_crossing"


def global_added_columns(config: Config, rooms: list[dict], engine: Engine) -> list[int]:
    kwargs = frontier_model_kwargs(config, rooms, engine)
    metadata = engine.get_output_metadata()
    context = FeatureContext(
        features=config.features,
        output_metadata=metadata,
        num_rooms=len(rooms),
        num_room_parts=metadata.num_room_parts,
        num_connection_outputs=len(metadata.connection),
        door_counts=kwargs["door_counts"],
        frontier_window_area=config.generation.frontier_window_size**2,
        area_bounding_box_width=config.generation.area_bounding_box_width,
        area_bounding_box_height=config.generation.area_bounding_box_height,
        max_area_size=config.generation.max_area_size,
    )
    columns = []
    offset = 0
    for feature in GLOBAL_FEATURES:
        if not feature.is_enabled(config.features):
            continue
        if feature is GenerationVariableFloatsFeature:
            columns.append(offset + GENERATION_VARIABLE_FLOAT_FIELDS.index(REWARD_FIELD))
        elif feature is LookaheadFeature:
            start = offset + config.features.lookahead_outcomes
            columns.extend(range(start, start + AREA_PAIR_COUNT))
        offset += feature.tensor_width(context)
    return columns


def expand_columns(
    value: torch.Tensor, shape: torch.Size, added_columns: list[int]
) -> torch.Tensor:
    if len(shape) != 2 or value.shape != (shape[0], shape[1] - len(added_columns)):
        raise ValueError(
            f"unexpected source shape {tuple(value.shape)} for destination {tuple(shape)}"
        )
    if len(set(added_columns)) != len(added_columns) or any(
        i < 0 or i >= shape[1] for i in added_columns
    ):
        raise ValueError("invalid added column indices")
    keep = torch.ones(shape[1], dtype=torch.bool)
    keep[added_columns] = False
    result = value.new_zeros(shape)
    result[:, keep] = value
    return result


def expand_model_inputs(tensors, model, prefix: str, columns: dict[str, list[int]]) -> None:
    expected = model.state_dict()
    for name, added_columns in columns.items():
        key = f"{prefix}.{name}"
        tensors[key] = expand_columns(tensors[key], expected[name].shape, added_columns)


def expand_optimizer_inputs(
    tensors,
    metadata,
    model,
    optimizer,
    prefix: str,
    columns: dict[str, list[int]],
    added_parameters: set[str],
) -> None:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    for part_name, part in named_checkpoint_optimizers(optimizer).items():
        groups = json.loads(metadata[f"{prefix}_{part_name}_param_groups"])
        for saved, current in zip(groups, part.param_groups, strict=True):
            old_parameters = [p for p in current["params"] if names[id(p)] not in added_parameters]
            for parameter_id, parameter in zip(saved["params"], old_parameters, strict=True):
                name = names[id(parameter)]
                if name not in columns:
                    continue
                state_prefix = f"{prefix}.{part_name}.state.{parameter_id}."
                for key in list(tensors):
                    value = tensors[key]
                    if key.startswith(state_prefix) and value.ndim:
                        tensors[key] = expand_columns(value, parameter.shape, columns[name])
    extend_optimizer_groups(tensors, metadata, model, optimizer, prefix, added_parameters)


def migrate_checkpoint(source: Path, output: Path) -> dict:
    if TRAINING_CHECKPOINT_FORMAT != TARGET_FORMAT:
        raise ValueError("this migration requires the v24 model implementation")
    if output.exists():
        raise FileExistsError(output)
    with safe_open(source, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata()
        if metadata is None or metadata["format"] != SOURCE_FORMAT:
            raise ValueError(f"expected a {SOURCE_FORMAT} checkpoint")
        tensors = {name: checkpoint.get_tensor(name) for name in checkpoint.keys()}
    data = json.loads(metadata["config"])
    if REWARD_FIELD in data["generation"] or "area_distinct_crossing_weight" in data["train"]:
        raise ValueError("source already contains area-connection config fields")
    data["generation"][REWARD_FIELD] = 0.0
    data["train"]["area_distinct_crossing_weight"] = 1.0
    config = Config.model_validate(data)
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
    model = FrontierModel(**frontier_model_kwargs(step_config, rooms, engine))
    global_columns = global_added_columns(step_config, rooms, engine)
    main_columns = {"global_mlp.weight": global_columns} if global_columns else {}
    for prefix in ("main_model", "ema_model"):
        expand_model_inputs(tensors, model, prefix, main_columns)
        added = add_model_parameters(tensors, model, prefix, NEW_HEAD)
        for name in added:
            if torch.count_nonzero(tensors[f"{prefix}.{name}"]):
                raise ValueError("new connection head must initialize to zero")
    model.load_state_dict(without_prefix(tensors, "main_model"), strict=True)
    optimizer = create_main_optimizer(model, step_config.optimizer)
    expand_optimizer_inputs(tensors, metadata, model, optimizer, "optimizer", main_columns, added)
    balance = create_balance_model(step_config, rooms, engine, torch.device("cpu"))
    reward_column = GENERATION_VARIABLE_FLOAT_FIELDS.index(REWARD_FIELD)
    balance_columns = {
        f"{family}_net.0.weight": [reward_column]
        for family in ("door", "toilet", "area", "order", "step")
    }
    expand_model_inputs(tensors, balance, "balance_model", balance_columns)
    balance.load_state_dict(without_prefix(tensors, "balance_model"), strict=True)
    balance_optimizer = create_adam_optimizer(balance.parameters(), step_config.balance_optimizer)
    expand_optimizer_inputs(
        tensors, metadata, balance, balance_optimizer, "balance_optimizer", balance_columns, set()
    )
    # Existing columns and all unchanged tensors must survive bit-for-bit.
    with safe_open(source, framework="pt", device="cpu") as checkpoint:
        for name in checkpoint.keys():
            original = checkpoint.get_tensor(name)
            migrated = tensors[name]
            if original.shape == migrated.shape and not torch.equal(original, migrated):
                raise ValueError(f"migration changed an existing tensor: {name}")
    report = {
        "source": str(source),
        "output": str(output),
        "source_format": SOURCE_FORMAT,
        "output_format": TARGET_FORMAT,
        "reward_area_distinct_crossing": 0.0,
        "area_distinct_crossing_weight": 1.0,
        "global_added_columns": global_columns,
        "generation_variable_added_column": reward_column,
        "added_parameters": sorted(added),
        "experience_import_formats": [
            "map-gen-experience-v3", "map-gen-experience-v4", "map-gen-experience-v5"
        ],
    }
    metadata["format"] = TARGET_FORMAT
    metadata["config"] = config.model_dump_json()
    metadata["area_connections_migration"] = json.dumps(report)
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
