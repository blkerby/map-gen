#!/usr/bin/env python3
"""Add return-distance validity to a v24 training checkpoint.

Run with the map-gen environment active:
  python scripts/migrate_return_distance.py SOURCE OUTPUT

The checkpoint limit starts disabled. Added input columns and prediction heads
start at zero; existing weights, optimizer state, counters, and Aim identity are
preserved. Experience files do not need migration: older schemas import an
unrestricted limit. The source is preserved and the output must not exist.

Generation requests use max_return_distance: null to disable the cap, or a
positive limit up to 254. Internal room-part connections cost zero. Training
also accepts the existing schedule/range syntax and mixtures whose values may
be null. Distance outcomes share reward_connection and train.connection_weight.
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
from scripts.checkpoint_migration import add_model_parameters
from scripts.migrate_area_connections import expand_model_inputs, expand_optimizer_inputs
from train import (
    TRAINING_CHECKPOINT_FORMAT,
    create_adam_optimizer,
    create_main_optimizer,
    validate_checkpoint_metadata,
)
from train_config import (
    Config,
    GENERATION_VARIABLE_FLOAT_FIELDS,
    instantiate_scheduleable_config,
    validate_config,
)


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
            columns.append(offset + GENERATION_VARIABLE_FLOAT_FIELDS.index("max_return_distance"))
        elif feature is LookaheadFeature:
            start = (
                offset + config.features.lookahead_outcomes + AREA_PAIR_COUNT
                + 2 * len(metadata.connection)
            )
            columns.extend(range(start, start + 2 * len(metadata.connection)))
        offset += feature.tensor_width(context)
    return columns


def write_output(tensors, metadata, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent, suffix=".safetensors", delete=False
    ) as temporary:
        path = Path(temporary.name)
    try:
        save_file(tensors, path, metadata=metadata)
        os.link(path, output)
    finally:
        path.unlink()


def migrate_checkpoint(source: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    with safe_open(source, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata()
        if metadata is None or metadata["format"] != "map-gen-training-session-checkpoint-v24":
            raise ValueError("expected a v24 training checkpoint")
        tensors = {name: checkpoint.get_tensor(name) for name in checkpoint.keys()}
    data = json.loads(metadata["config"])
    if "max_return_distance" in data["generation"]:
        raise ValueError("source already contains max_return_distance")
    data["generation"]["max_return_distance"] = None
    config = Config.model_validate(data)
    validate_config(config)
    step = instantiate_scheduleable_config(config, int(metadata["num_episodes"]))
    room_path = Path(config.room_set)
    if not room_path.is_absolute():
        room_path = Path(__file__).resolve().parents[1] / room_path
    rooms = json.loads(room_path.read_text())
    engine = Engine(
        rooms, step.features, step.generation.min_area_size, step.generation.max_area_size
    )
    model = FrontierModel(**frontier_model_kwargs(step, rooms, engine))
    columns = global_added_columns(step, rooms, engine)
    main_columns = {"global_mlp.weight": columns} if columns else {}
    for prefix in ("main_model", "ema_model"):
        expand_model_inputs(tensors, model, prefix, main_columns)
        added = add_model_parameters(tensors, model, prefix, "return_distance_output.")
    model.load_state_dict(without_prefix(tensors, "main_model"), strict=True)
    optimizer = create_main_optimizer(model, step.optimizer)
    expand_optimizer_inputs(tensors, metadata, model, optimizer, "optimizer", main_columns, added)
    balance = create_balance_model(step, rooms, engine, torch.device("cpu"))
    limit_column = GENERATION_VARIABLE_FLOAT_FIELDS.index("max_return_distance")
    balance_columns = {
        f"{family}_net.0.weight": [limit_column]
        for family in ("door", "toilet", "area", "order", "step")
    }
    expand_model_inputs(tensors, balance, "balance_model", balance_columns)
    balance.load_state_dict(without_prefix(tensors, "balance_model"), strict=True)
    balance_optimizer = create_adam_optimizer(balance.parameters(), step.balance_optimizer)
    expand_optimizer_inputs(
        tensors, metadata, balance, balance_optimizer, "balance_optimizer", balance_columns, set()
    )
    report = {
        "source": str(source),
        "output": str(output),
        "format": TRAINING_CHECKPOINT_FORMAT,
        "global_added_columns": columns,
        "added_parameters": sorted(added),
    }
    metadata["format"] = TRAINING_CHECKPOINT_FORMAT
    metadata["config"] = config.model_dump_json()
    metadata["return_distance_migration"] = json.dumps(report)
    validate_checkpoint_metadata(output, metadata)
    write_output(tensors, metadata, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = migrate_checkpoint(args.source, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
