#!/usr/bin/env python3
"""Expand a v25 grouped-room training checkpoint into a v26 concrete-room checkpoint.

From the repository root:
  conda run -n map-gen python scripts/migrate_concrete_rooms.py SOURCE OUTPUT

Copies shared weights and optimizer moments to their former members. Training
counters and run identity are retained. Proposal scores are retained per door,
but sampling probabilities change because each concrete door is now a choice.
Re-export inference models from the migrated checkpoint.
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
from features import FrontierDoorVariantFeature, GlobalRoomPositionFeature
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

SOURCE_FORMAT = "map-gen-training-session-checkpoint-v25"
TARGET_FORMAT = "map-gen-training-session-checkpoint-v26"
DIRECTIONS = ("left", "right", "up", "down")


def legacy_indices(
    rooms: list[dict],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Reproduce v25's first-occurrence ordering of room, door, and connection classes."""
    room_classes = {}
    room_indices = []
    doors = [[] for _ in DIRECTIONS]
    for room in rooms:
        local_doors = [door for group in room["doors"] for door in group]
        geometry = (
            tuple(tuple(row) for row in room["map"]),
            tuple(sorted(room["toilet_crossing_x"])),
            room.get("special_type") == "toilet",
            tuple(
                sorted(
                    (DIRECTIONS.index(d["direction"]), d["x"], d["y"], d["kind"])
                    for d in local_doors
                )
            ),
        )
        key = (
            geometry,
            room.get("special_type"),
            room.get("save", False),
            room.get("refill", False),
            room.get("map_station", False),
            room.get("heat", 0),
            room.get("water", 0),
            tuple(sorted(tuple(pair) for pair in room["connections"])),
            tuple(sorted(tuple(pair) for pair in room["missing_connections"])),
        )
        variant = room_classes.setdefault(key, len(room_classes))
        room_indices.append(variant)
        for door in local_doors:
            direction = DIRECTIONS.index(door["direction"])
            doors[direction].append((variant, direction, door["x"], door["y"], door["kind"]))
    door_classes = {}
    door_indices = [
        door_classes.setdefault(key, len(door_classes)) for group in doors for key in group
    ]
    connection_classes = {}
    connection_indices = [
        connection_classes.setdefault((variant, *pair), len(connection_classes))
        for room, variant in zip(rooms, room_indices, strict=True)
        for pair in room["missing_connections"]
    ]
    return (
        torch.tensor(room_indices, dtype=torch.int64),
        torch.tensor(door_indices, dtype=torch.int64),
        torch.tensor(connection_indices, dtype=torch.int64),
        [len(group) for group in doors],
    )


def output_rows(indices: torch.Tensor, width: int) -> torch.Tensor:
    return (indices.unsqueeze(1) * width + torch.arange(width)).flatten()


def parameter_indices(
    main: FrontierModel,
    balance: torch.nn.Module,
    room: torch.Tensor,
    door: torch.Tensor,
    connection: torch.Tensor,
    counts: list[int],
) -> tuple[dict, dict]:
    main_indices = {}
    for head, indices in (
        ("door_output", door),
        ("balance_score_output", door),
        ("connection_output", connection),
        ("return_distance_output", connection),
    ):
        for suffix in ("weight", "bias"):
            main_indices[f"{head}.{suffix}"] = (0, indices)
    last = len(main.proposal_output.layers) - 1
    main_indices[f"proposal_output.layers.{last}.weight"] = (0, output_rows(door, AREA_COUNT))
    for name, module in main.named_modules():
        if isinstance(module, GlobalRoomPositionFeature):
            for axis in ("x", "y"):
                main_indices[f"{name}.embedding_{axis}"] = (0, room)
        if isinstance(module, FrontierDoorVariantFeature):
            main_indices[f"{name}.embedding.weight"] = (0, door)
    if main.features.inventory:
        # Inventory is the first global feature; the remaining columns retain their order.
        old_room_count = int(room.max()) + 1
        old_width = main.global_mlp.in_features - room.numel() + old_room_count
        columns = torch.cat((room, torch.arange(old_room_count, old_width)))
        main_indices["global_mlp.weight"] = (1, columns)
    direction_indices = []
    old_counts = []
    for group in torch.split(door, counts):
        unique, inverse = torch.unique(group, sorted=True, return_inverse=True)
        direction_indices.append(inverse)
        old_counts.append(unique.numel())
    door_rows = []
    offset = 0
    for source, target in ((0, 1), (1, 0), (2, 3), (3, 2)):
        door_rows.append(
            (
                direction_indices[source].unsqueeze(1) * old_counts[target]
                + direction_indices[target].unsqueeze(0)
                + offset
            ).flatten()
        )
        offset += old_counts[source] * old_counts[target]
    for indices, count in zip(direction_indices, old_counts, strict=True):
        door_rows.append(indices + offset)
        offset += count
    old_room_count = int(room.max()) + 1
    area_rows = torch.cat((output_rows(room, AREA_COUNT), room + old_room_count * AREA_COUNT))
    balance_indices = {}
    for network, rows in (("door_net", torch.cat(door_rows)), ("area_net", area_rows)):
        last = len(getattr(balance, network)) - 1
        for suffix in ("weight", "bias"):
            balance_indices[f"{network}.{last}.{suffix}"] = (0, rows)
    return main_indices, balance_indices


def expand_state(
    tensors: dict,
    model: torch.nn.Module,
    prefix: str,
    indices: dict,
    removed_buffers: dict[str, torch.Tensor],
) -> None:
    old = without_prefix(tensors, prefix)
    expected = model.state_dict()
    if set(old) != set(expected) | set(removed_buffers):
        raise ValueError(f"unexpected source keys for {prefix}")
    for name, value in removed_buffers.items():
        if not torch.equal(old[name], value):
            raise ValueError(f"source room set does not match {prefix}.{name}")
    expanded = {}
    parameter_names = dict(model.named_parameters())
    for name, target in expected.items():
        value = old[name]
        if name in indices:
            axis, mapping = indices[name]
            # Check the old extent as well as the expanded shape.
            old_extent = int(mapping.max()) + 1 if mapping.numel() else 0
            if value.shape[axis] != old_extent:
                raise ValueError(f"unexpected source shape for {prefix}.{name}: {value.shape}")
            value = value.index_select(axis, mapping)
        if value.shape != target.shape:
            raise ValueError(
                f"unexpected shape for {prefix}.{name}: {value.shape}, expected {target.shape}"
            )
        if name not in parameter_names and not torch.equal(value, target):
            raise ValueError(f"source buffer disagrees with room set: {prefix}.{name}")
        expanded[name] = value.contiguous().clone()
    model.load_state_dict(expanded, strict=True)
    for name in list(tensors):
        if name.startswith(f"{prefix}."):
            del tensors[name]
    tensors.update({f"{prefix}.{name}": value for name, value in expanded.items()})


def expand_optimizer(
    tensors: dict, metadata: dict, model, optimizer, prefix: str, indices: dict
) -> None:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    parts = named_checkpoint_optimizers(optimizer)
    if set(json.loads(metadata[f"{prefix}_names"])) != set(parts):
        raise ValueError(f"unexpected optimizer parts for {prefix}")
    for part_name, part in parts.items():
        groups = json.loads(metadata[f"{prefix}_{part_name}_param_groups"])
        for saved, current in zip(groups, part.param_groups, strict=True):
            for param_id, parameter in zip(saved["params"], current["params"], strict=True):
                name = names[id(parameter)]
                state_prefix = f"{prefix}.{part_name}.state.{param_id}."
                for key in [key for key in tensors if key.startswith(state_prefix)]:
                    value = tensors[key]
                    if value.ndim == 0:
                        continue
                    if name in indices:
                        axis, mapping = indices[name]
                        value = value.index_select(axis, mapping)
                    if value.shape != parameter.shape:
                        raise ValueError(f"unexpected optimizer moment shape: {key}")
                    tensors[key] = value.contiguous().clone()
    load_named_optimizer_checkpoint_state(optimizer, tensors, metadata, prefix)


def migrate_checkpoint(source: Path, output: Path) -> dict:
    if TRAINING_CHECKPOINT_FORMAT != TARGET_FORMAT:
        raise ValueError("this migration requires the v26 model implementation")
    if output.exists():
        raise FileExistsError(output)
    with safe_open(source, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata()
        if metadata is None or metadata["format"] != SOURCE_FORMAT:
            raise ValueError(f"expected a {SOURCE_FORMAT} checkpoint")
        tensors = {name: checkpoint.get_tensor(name) for name in checkpoint.keys()}
    config = Config.model_validate_json(metadata["config"])
    validate_config(config)
    current = instantiate_scheduleable_config(config, int(metadata["num_episodes"]))
    room_path = Path(config.room_set)
    if not room_path.is_absolute():
        room_path = Path(__file__).resolve().parents[1] / room_path
    rooms = json.loads(room_path.read_text())
    room, door, connection, counts = legacy_indices(rooms)
    engine = Engine(
        rooms, current.features, current.generation.min_area_size, current.generation.max_area_size
    )
    main = FrontierModel(**frontier_model_kwargs(current, rooms, engine))
    balance = create_balance_model(current, rooms, engine, torch.device("cpu"))
    main_indices, balance_indices = parameter_indices(
        main, balance, room, door, connection, counts
    )
    main_removed = {"door_variant_outcome_idx": door, "connection_variant_outcome_idx": connection}
    for name, module in main.named_modules():
        if isinstance(module, GlobalRoomPositionFeature):
            main_removed[f"{name}.room_connection_variant_idx"] = room
    for prefix in ("main_model", "ema_model"):
        expand_state(tensors, main, prefix, main_indices, main_removed)
    balance_removed = {"room_connection_variant_idx": room}
    for direction, group in zip(DIRECTIONS, torch.split(door, counts), strict=True):
        unique, inverse = torch.unique(group, sorted=True, return_inverse=True)
        balance_removed[f"{direction}_door_variant_idx"] = inverse
        balance_removed[f"{direction}_global_door_variant_idx"] = unique
    expand_state(tensors, balance, "balance_model", balance_indices, balance_removed)
    expand_optimizer(
        tensors,
        metadata,
        main,
        create_main_optimizer(main, current.optimizer),
        "optimizer",
        main_indices,
    )
    expand_optimizer(
        tensors,
        metadata,
        balance,
        create_adam_optimizer(balance.parameters(), current.balance_optimizer),
        "balance_optimizer",
        balance_indices,
    )
    report = {
        "source": str(source),
        "source_format": SOURCE_FORMAT,
        "output_format": TARGET_FORMAT,
        "rooms": [int(room.max()) + 1, len(rooms)],
        "doors": [int(door.max()) + 1 if door.numel() else 0, door.numel()],
        "connections": [
            int(connection.max()) + 1 if connection.numel() else 0,
            connection.numel(),
        ],
    }
    metadata["format"] = TARGET_FORMAT
    metadata["concrete_rooms_migration"] = json.dumps(report)
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
