"""Subdivide generated areas without changing room placements or area IDs."""

from dataclasses import dataclass

import torch

from env import DoorMatches


DIRECTIONS = ("left", "right", "up", "down")


@dataclass(frozen=True)
class DoorRoomLookup:
    left: torch.Tensor
    right: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor
    room_count: int


def build_door_room_lookup(rooms: list[dict], device: torch.device) -> DoorRoomLookup:
    door_rooms = {direction: [] for direction in DIRECTIONS}
    for room_idx, room in enumerate(rooms):
        for door_group in room["doors"]:
            for door in door_group:
                door_rooms[door["direction"]].append(room_idx)
    return DoorRoomLookup(
        left=torch.tensor(door_rooms["left"], device=device, dtype=torch.int64),
        right=torch.tensor(door_rooms["right"], device=device, dtype=torch.int64),
        up=torch.tensor(door_rooms["up"], device=device, dtype=torch.int64),
        down=torch.tensor(door_rooms["down"], device=device, dtype=torch.int64),
        room_count=len(rooms),
    )


def add_matched_door_edges(
    adjacency: torch.Tensor,
    matches: torch.Tensor,
    source_room_by_door: torch.Tensor,
    target_room_by_door: torch.Tensor,
    room_position_by_room: torch.Tensor,
) -> None:
    valid_match = matches >= 0
    environment_idx, source_door_idx = torch.where(valid_match)
    target_door_idx = matches[environment_idx, source_door_idx].to(torch.int64)
    source_room = source_room_by_door[source_door_idx]
    target_room = target_room_by_door[target_door_idx]
    source_position = room_position_by_room[environment_idx, source_room]
    target_position = room_position_by_room[environment_idx, target_room]
    placed_edge = (source_position >= 0) & (target_position >= 0)
    environment_idx = environment_idx[placed_edge]
    source_position = source_position[placed_edge]
    target_position = target_position[placed_edge]
    adjacency[environment_idx, source_position, target_position] = True
    adjacency[environment_idx, target_position, source_position] = True


def build_room_position_by_room(room_idx: torch.Tensor, total_room_count: int) -> torch.Tensor:
    environment_count, room_count = room_idx.shape
    room_position_by_room = torch.full(
        (environment_count, total_room_count),
        -1,
        device=room_idx.device,
        dtype=torch.int64,
    )
    environment_idx = torch.arange(environment_count, device=room_idx.device)[:, None].expand_as(
        room_idx
    )
    room_position = torch.arange(room_count, device=room_idx.device)[None, :].expand_as(room_idx)
    room_position_by_room[environment_idx, room_idx.to(torch.int64)] = room_position
    return room_position_by_room


def build_room_adjacency(
    room_idx: torch.Tensor,
    door_matches: DoorMatches,
    door_room_lookup: DoorRoomLookup,
) -> torch.Tensor:
    environment_count, room_count = room_idx.shape
    room_position_by_room = build_room_position_by_room(room_idx, door_room_lookup.room_count)
    adjacency = torch.zeros(
        (environment_count, room_count, room_count),
        device=room_idx.device,
        dtype=torch.bool,
    )
    diagonal = torch.arange(room_count, device=room_idx.device)
    adjacency[:, diagonal, diagonal] = True
    add_matched_door_edges(
        adjacency,
        door_matches.left,
        door_room_lookup.left,
        door_room_lookup.right,
        room_position_by_room,
    )
    add_matched_door_edges(
        adjacency,
        door_matches.right,
        door_room_lookup.right,
        door_room_lookup.left,
        room_position_by_room,
    )
    add_matched_door_edges(
        adjacency,
        door_matches.up,
        door_room_lookup.up,
        door_room_lookup.down,
        room_position_by_room,
    )
    add_matched_door_edges(
        adjacency,
        door_matches.down,
        door_room_lookup.down,
        door_room_lookup.up,
        room_position_by_room,
    )
    return adjacency


def compute_room_distances(adjacency: torch.Tensor) -> torch.Tensor:
    room_count = adjacency.shape[1]
    unreachable_distance = room_count + 1
    distances = torch.full(
        adjacency.shape,
        unreachable_distance,
        device=adjacency.device,
        dtype=torch.int16,
    )
    off_diagonal = ~torch.eye(room_count, device=adjacency.device, dtype=torch.bool)
    distances = torch.where(adjacency & off_diagonal, torch.ones_like(distances), distances)
    distances = torch.where(adjacency & ~off_diagonal, torch.zeros_like(distances), distances)
    for midpoint in range(room_count):
        distances = torch.minimum(
            distances,
            distances[:, :, midpoint, None] + distances[:, None, midpoint, :],
        )
    return distances


def induced_distances(adjacency: torch.Tensor, member_positions: torch.Tensor) -> torch.Tensor:
    parent_adjacency = adjacency[member_positions][:, member_positions]
    return compute_room_distances(parent_adjacency[None, :, :])[0]


def split_assignment_by_balanced_centers(
    adjacency: torch.Tensor,
    parent_assignments: tuple[torch.Tensor, ...],
    parent_counts: tuple[int, ...],
) -> torch.Tensor:
    parent_assignment = parent_assignments[0]
    environment_count, _room_count = parent_assignment.shape
    parent_key_count = 1
    for parent_count in parent_counts:
        parent_key_count *= parent_count
    child_assignment = torch.empty(
        parent_assignment.shape,
        device=parent_assignment.device,
        dtype=torch.int64,
    )
    for environment_idx in range(environment_count):
        environment_adjacency = adjacency[environment_idx]
        for parent_key in range(parent_key_count):
            remaining_key = parent_key
            member_mask = torch.ones(
                (parent_assignment.shape[1],),
                device=parent_assignment.device,
                dtype=torch.bool,
            )
            for assignment, parent_count in zip(parent_assignments, parent_counts):
                parent_id = remaining_key % parent_count
                remaining_key = remaining_key // parent_count
                member_mask = member_mask & (assignment[environment_idx] == parent_id)

            member_positions = torch.where(member_mask)[0]
            if member_positions.shape[0] < 2:
                child_assignment[environment_idx, member_positions] = 0
                continue

            parent_distances = induced_distances(environment_adjacency, member_positions)
            nearer_second = parent_distances[:, None, :] < parent_distances[:, :, None]
            second_count = torch.sum(nearer_second.to(torch.int64), dim=0)
            first_count = member_positions.shape[0] - second_count
            balance = torch.minimum(first_count, second_count)
            diagonal = torch.arange(member_positions.shape[0], device=adjacency.device)
            balance[diagonal, diagonal] = -1
            best_pair = torch.argmax(balance.reshape(-1))
            first_center = best_pair // member_positions.shape[0]
            second_center = best_pair % member_positions.shape[0]
            child_assignment[environment_idx, member_positions] = nearer_second[
                :,
                first_center,
                second_center,
            ].to(torch.int64)
    return child_assignment
