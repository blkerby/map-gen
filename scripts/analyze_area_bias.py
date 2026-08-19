#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from train_config import (  # noqa: E402
    GENERATION_VARIABLE_FLOAT_FIELDS,
    HEAT_WATER_REWARD_FIELDS,
    VANILLA_AREA_CONDITION_FIELDS,
)


EXPERIENCE_FORMAT = "map-gen-experience-v2"
ROOM_DEFINITIONS_PATH = REPO_ROOT / "room_definitions" / "zebes.json"
REWARD_BIN_COUNT = 5
AREA_COUNT = 6
SPECIAL_ROOMS = (
    ("Ship", "ship"),
    ("Kraid", "kraid_boss"),
    ("Ridley", "ridley_boss"),
    ("Phantoon", "phantoon_boss"),
    ("Draygon", "draygon_boss"),
    ("Mother Brain", "mother_brain"),
)
AREA_NAMES = ("Crateria", "Brinstar", "Norfair", "Wrecked Ship", "Maridia", "Tourian")
HEAT_WATER = (
    ("water", "water", "maridia_water", 4),
    ("heat", "heat", "norfair_heat", 2),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze preferred-area placement in Zebes experience files.",
    )
    parser.add_argument(
        "experience",
        type=Path,
        nargs="+",
        help="One or more map-gen experience safetensors files.",
    )
    return parser.parse_args()


def load_experience(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    room_indices = []
    room_areas = []
    generation_variables = []
    for path in paths:
        with safe_open(path, framework="np") as experience:
            metadata = experience.metadata()
            if metadata is None or metadata.get("format") != EXPERIENCE_FORMAT:
                raise ValueError(f"unsupported experience format in {path}")
            required = {"room_idx", "room_area", "generation_variable_floats"}
            missing = required.difference(experience.keys())
            if missing:
                raise ValueError(f"{path} missing tensor(s): {', '.join(sorted(missing))}")
            room_indices.append(experience.get_tensor("room_idx"))
            room_areas.append(experience.get_tensor("room_area"))
            generation_variables.append(experience.get_tensor("generation_variable_floats"))
    return (
        np.concatenate(room_indices),
        np.concatenate(room_areas),
        np.concatenate(generation_variables),
    )


def reconstruct_assignments(
    room_indices: np.ndarray,
    room_areas: np.ndarray,
    room_count: int,
) -> np.ndarray:
    if room_indices.shape != room_areas.shape:
        raise ValueError("room_idx and room_area shapes differ")
    assignments = np.full((len(room_indices), room_count), -1, dtype=np.int8)
    episode_indices = np.broadcast_to(np.arange(len(room_indices))[:, None], room_indices.shape)
    placed = (room_indices >= 0) & (room_indices < room_count)
    assignments[episode_indices[placed], room_indices[placed]] = room_areas[placed]
    return assignments


def equal_count_bins(values: np.ndarray, count: int) -> list[np.ndarray]:
    bins = [indices for indices in np.array_split(np.argsort(values), count) if len(indices)]
    assert sum(map(len, bins)) == len(values)
    return bins


def percent(numerator: int, denominator: int) -> float:
    return 100 * numerator / denominator if denominator else float("nan")


def print_table(rows: list[tuple[str, ...]], left_columns: int) -> None:
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    for row in rows:
        print(
            "  ".join(
                value.ljust(widths[column])
                if column < left_columns
                else value.rjust(widths[column])
                for column, value in enumerate(row)
            )
        )


def print_heat_water(assignments: np.ndarray, variables: np.ndarray, rooms: list[dict]) -> None:
    print("heat_water_reward_response")
    rows = [
        (
            "family",
            "tier",
            "bin",
            "episodes",
            "rooms",
            "reward_min",
            "reward_mean",
            "reward_max",
            "preferred_pct",
        )
    ]
    for label, room_field, reward_family, preferred_area in HEAT_WATER:
        for tier in range(1, 4):
            room_idx = np.array(
                [i for i, room in enumerate(rooms) if room.get(room_field, 0) == tier]
            )
            reward_name = f"reward_{reward_family}_{tier}"
            assert reward_name in HEAT_WATER_REWARD_FIELDS
            rewards = variables[:, GENERATION_VARIABLE_FLOAT_FIELDS.index(reward_name)]
            for bin_idx, episodes in enumerate(equal_count_bins(rewards, REWARD_BIN_COUNT), 1):
                selected = assignments[episodes][:, room_idx]
                preferred = selected == preferred_area
                total = selected.size
                rows.append(
                    (
                        label,
                        str(tier),
                        str(bin_idx),
                        str(len(episodes)),
                        str(len(room_idx)),
                        f"{rewards[episodes].min():.6g}",
                        f"{rewards[episodes].mean():.6g}",
                        f"{rewards[episodes].max():.6g}",
                        f"{percent(preferred.sum(), total):.3f}",
                    )
                )
    print_table(rows, 1)


def print_unforced_special(assignments: np.ndarray, variables: np.ndarray, rooms: list[dict]) -> None:
    print("\nunforced_special_room_preferences")
    rows = [("room", "episodes", *(f"{name}_pct" for name in AREA_NAMES))]
    for constraint_idx, (label, special_type) in enumerate(SPECIAL_ROOMS):
        room_idx = next(
            i for i, room in enumerate(rooms) if room.get("special_type") == special_type
        )
        force_field = VANILLA_AREA_CONDITION_FIELDS[constraint_idx]
        unforced = ~variables[:, GENERATION_VARIABLE_FLOAT_FIELDS.index(force_field)].astype(bool)
        selected = assignments[unforced, room_idx]
        area_counts = [(selected == area).sum() for area in range(AREA_COUNT)]
        assert sum(area_counts) <= unforced.sum()
        area_rates = [percent(count, unforced.sum()) for count in area_counts]
        rows.append((label, str(unforced.sum()), *(f"{rate:.3f}" for rate in area_rates)))
    print_table(rows, 1)


def main() -> None:
    args = parse_args()
    with ROOM_DEFINITIONS_PATH.open() as room_file:
        rooms = json.load(room_file)
    room_indices, room_areas, variables = load_experience(args.experience)
    if variables.shape[1] != len(GENERATION_VARIABLE_FLOAT_FIELDS):
        raise ValueError(
            "generation_variable_floats has "
            f"{variables.shape[1]} fields; expected {len(GENERATION_VARIABLE_FLOAT_FIELDS)}"
        )
    assignments = reconstruct_assignments(room_indices, room_areas, len(rooms))
    print_table(
        [("files", str(len(args.experience))), ("episodes", str(len(assignments)))],
        1,
    )
    print()
    print_heat_water(assignments, variables, rooms)
    print_unforced_special(assignments, variables, rooms)


if __name__ == "__main__":
    main()
