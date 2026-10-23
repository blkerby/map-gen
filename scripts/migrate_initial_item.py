#!/usr/bin/env python3
"""Upgrade a v26 checkpoint with zero-weight initial-item conditioning and prediction.

  conda run -n map-gen python scripts/migrate_initial_item.py SOURCE OUTPUT

Existing predictions and optimizer moments are preserved. The new reward and
loss coefficients are zero. Zebes uses 50/25/25 unconstrained/Ship/other starts;
room sets without Landing Site remain unconstrained. Re-export inference models.
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
from features import GLOBAL_FEATURES, FeatureContext, GenerationVariableFloatsFeature
from initial_item import NODE_COUNT, GRAPH_COUNT
from model import FrontierModel
from model_loading import create_balance_model, frontier_model_kwargs, without_prefix
from scripts.checkpoint_migration import add_model_parameters
from scripts.migrate_area_connections import expand_model_inputs, expand_optimizer_inputs
from scripts.migrate_return_distance import write_output
from train import (
    TRAINING_CHECKPOINT_FORMAT,
    create_main_optimizer,
    create_adam_optimizer,
    validate_checkpoint_metadata,
)
from train_config import (
    Config,
    GENERATION_VARIABLE_FLOAT_FIELDS,
    instantiate_scheduleable_config,
    validate_config,
)

SOURCE_FORMAT = "map-gen-training-session-checkpoint-v26"
TARGET_FORMAT = "map-gen-training-session-checkpoint-v27"


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
    offset = 0
    for feature in GLOBAL_FEATURES:
        if not feature.is_enabled(config.features):
            continue
        width = feature.tensor_width(context)
        if feature is GenerationVariableFloatsFeature:
            return list(range(offset + len(GENERATION_VARIABLE_FLOAT_FIELDS) - 3, offset + width))
        offset += width
    return []


def migrate_checkpoint(source: Path, output: Path) -> dict:
    if TRAINING_CHECKPOINT_FORMAT != TARGET_FORMAT:
        raise ValueError("this migration requires the v27 model implementation")
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
    data = json.loads(metadata["config"])
    room_path = Path(data["room_set"])
    if not room_path.is_absolute():
        room_path = Path(__file__).resolve().parents[1] / room_path
    rooms = json.loads(room_path.read_text())
    probabilities = (
        [0.5, 0.25, 0.25] if any(room.get("room_id") == 8 for room in rooms) else [1.0, 0.0, 0.0]
    )
    data["generation"].update(
        reward_initial_item=0.0,
        initial_item_start_probabilities=probabilities,
        initial_item_wall_jump_probability=0.5,
    )
    data["train"]["initial_item_weight"] = 0.0
    config = Config.model_validate(data)
    validate_config(config)
    current = instantiate_scheduleable_config(config, int(metadata["num_episodes"]))
    engine = Engine(
        rooms, current.features, current.generation.min_area_size, current.generation.max_area_size
    )
    model = FrontierModel(**frontier_model_kwargs(current, rooms, engine))
    columns = global_added_columns(current, rooms, engine)
    main_columns = {"global_mlp.weight": columns} if columns else {}
    for prefix in ("main_model", "ema_model"):
        expand_model_inputs(tensors, model, prefix, main_columns)
        added = add_model_parameters(tensors, model, prefix, "initial_item_output.")
    expand_optimizer_inputs(
        tensors,
        metadata,
        model,
        create_main_optimizer(model, current.optimizer),
        "optimizer",
        main_columns,
        added,
    )
    balance = create_balance_model(current, rooms, engine, torch.device("cpu"))
    old_width = len(GENERATION_VARIABLE_FLOAT_FIELDS) - 3
    new_width = len(GENERATION_VARIABLE_FLOAT_FIELDS) - 2 + NODE_COUNT + 1 + GRAPH_COUNT
    balance_columns = {
        f"{family}_net.0.weight": list(range(old_width, new_width))
        for family in ("door", "toilet", "area", "order", "step")
    }
    expand_model_inputs(tensors, balance, "balance_model", balance_columns)
    balance.load_state_dict(without_prefix(tensors, "balance_model"), strict=True)
    expand_optimizer_inputs(
        tensors,
        metadata,
        balance,
        create_adam_optimizer(balance.parameters(), current.balance_optimizer),
        "balance_optimizer",
        balance_columns,
        set(),
    )
    report = {
        "source": str(source),
        "source_format": SOURCE_FORMAT,
        "output_format": TARGET_FORMAT,
        "initial_item_start_probabilities": probabilities,
        "added_global_columns": len(columns),
    }
    metadata["format"] = TARGET_FORMAT
    metadata["config"] = config.model_dump_json()
    metadata["initial_item_migration"] = json.dumps(report)
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
