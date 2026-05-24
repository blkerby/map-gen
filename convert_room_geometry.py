#!/usr/bin/env python3
"""Convert room geometry from parts-indexed doors to grouped doors.

Old schema:
    doors: [door, ...]
    parts: [[door_index, ...], ...]
    part_connections: [[part_index, part_index], ...]

New schema:
    doors: [[door, ...], ...]
    connections: [[part_index, part_index], ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


JsonObject = dict[str, Any]


def convert_room(room: JsonObject) -> JsonObject:
    """Return one converted room, validating its part door indexes."""
    if "parts" not in room:
        raise ValueError(f"room {room_label(room)} is missing 'parts'")
    if "part_connections" not in room:
        raise ValueError(f"room {room_label(room)} is missing 'part_connections'")

    doors = room.get("doors")
    parts = room["parts"]

    if not isinstance(doors, list):
        raise ValueError(f"room {room_label(room)} has non-list 'doors'")
    if not isinstance(parts, list):
        raise ValueError(f"room {room_label(room)} has non-list 'parts'")

    seen: set[int] = set()
    grouped_doors: list[list[Any]] = []
    for part_index, part in enumerate(parts):
        if not isinstance(part, list):
            raise ValueError(
                f"room {room_label(room)} part {part_index} is not a list"
            )

        group = []
        for door_index in part:
            if not isinstance(door_index, int):
                raise ValueError(
                    f"room {room_label(room)} part {part_index} contains "
                    f"non-integer door index {door_index!r}"
                )
            if door_index < 0 or door_index >= len(doors):
                raise ValueError(
                    f"room {room_label(room)} part {part_index} references "
                    f"door index {door_index}, but valid indexes are "
                    f"0 through {len(doors) - 1}"
                )
            if door_index in seen:
                raise ValueError(
                    f"room {room_label(room)} references door index "
                    f"{door_index} more than once"
                )

            seen.add(door_index)
            group.append(doors[door_index])

        grouped_doors.append(group)

    missing = sorted(set(range(len(doors))) - seen)
    if missing:
        raise ValueError(
            f"room {room_label(room)} has doors not included in any part: {missing}"
        )

    converted: JsonObject = {}
    for key, value in room.items():
        if key == "doors":
            converted["doors"] = grouped_doors
        elif key == "parts":
            continue
        elif key == "part_connections":
            converted["connections"] = value
        else:
            converted[key] = value

    return converted


def room_label(room: JsonObject) -> str:
    room_id = room.get("room_id", "?")
    name = room.get("name", "<unnamed>")
    return f"{room_id} ({name})"


def convert_file(input_path: Path, output_path: Path) -> int:
    with input_path.open("r", encoding="utf-8") as f:
        rooms = json.load(f)

    if not isinstance(rooms, list):
        raise ValueError("expected the input file to contain a top-level list")

    converted_rooms = [convert_room(room) for room in rooms]

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(converted_rooms, f, indent=4)
        f.write("\n")

    return len(converted_rooms)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert room_geometry.json to grouped doors and connections."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("room_geometry.json"),
        help="input room geometry JSON file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("room_geometry.converted.json"),
        help="output JSON file",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="overwrite the input file instead of writing to --output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.input if args.in_place else args.output
    count = convert_file(args.input, output_path)
    print(f"Converted {count} rooms: {args.input} -> {output_path}")


if __name__ == "__main__":
    main()
