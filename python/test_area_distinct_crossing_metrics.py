import torch

from env import AREA_COUNT, Actions, DoorMatches, Engine
from test_room_area_plumbing import disabled_features, one_tile_room
from train import compute_area_distinct_crossing_counts, compute_valid_door_match_counts


def test_distinct_crossings_count_area_graph_edges_per_map() -> None:
    directions = ("left", "right", "up", "down")
    rooms = [{"doors": [[{"direction": direction} for direction in directions]]} for _ in range(7)]
    # Room 6 shares area 0 with room 0; actions are deliberately out of room order.
    room_idx = torch.tensor([[6, 5, 4, 3, 2, 1, 0]] * 4, dtype=torch.uint8)
    actions = Actions(
        room_idx=room_idx,
        room_x=torch.zeros_like(room_idx),
        room_y=torch.zeros_like(room_idx),
        room_area=room_idx % AREA_COUNT,
    )
    matches = torch.full((4, 28), -1, dtype=torch.int16)
    for episode in range(3):
        # A six-area path has five edges.
        for area in range(5):
            matches[episode, area] = area + 1
            matches[episode, 7 + area + 1] = area
    # Closing the path creates a cycle with six edges.
    matches[1, 14 + 5] = 0
    matches[1, 21] = 5
    # Repeated crossings, including reversed and vertical ones, still count once.
    matches[2, 14] = 1
    matches[2, 21 + 1] = 0
    matches[2, 14 + 1] = 6
    matches[2, 21 + 6] = 1
    # Connections within area 0 do not contribute, nor do invalid/unknown matches.
    matches[3, 0] = 6
    matches[3, 7 + 6] = 0
    matches[3, 1] = 7
    matches[3, 14 + 2] = 7

    door_matches = DoorMatches(
        left=matches[:, :7],
        right=matches[:, 7:14],
        up=matches[:, 14:21],
        down=matches[:, 21:],
    )
    counts = compute_area_distinct_crossing_counts(actions, door_matches, rooms)

    assert counts.tolist() == [5, 6, 5, 0]
    assert counts.float().mean().item() == 4.0


def test_distinct_crossings_use_engine_door_order_and_ignore_unplaced_rooms() -> None:
    rooms = [one_tile_room("Right", "right"), one_tile_room("Left", "left")]
    engine = Engine(rooms, disabled_features(), 1, 100)
    group = engine.create_environment_group(
        map_size=(4, 4),
        num_envs=2,
        candidate_spatial_cell_size=4,
        area_bounding_box_width=4,
        area_bounding_box_height=4,
        seed=0,
        num_threads=1,
    )
    actions = Actions(
        room_idx=torch.tensor([[1, 0], [1, 2]], dtype=torch.uint8),
        room_x=torch.tensor([[1, 0], [1, 0]], dtype=torch.int8),
        room_y=torch.zeros((2, 2), dtype=torch.int8),
        room_area=torch.tensor([[4, 2], [4, AREA_COUNT]], dtype=torch.uint8),
    )
    for step in range(2):
        group.step_known(
            Actions(
                room_idx=actions.room_idx[:, step],
                room_x=actions.room_x[:, step],
                room_y=actions.room_y[:, step],
                room_area=actions.room_area[:, step],
            )
        )
    group.finish()
    outcomes = group.get_outcomes(torch.device("cpu"), verify_consistency=True)

    door_matches, _ = engine.compute_balance_targets(actions, torch.device("cpu"))
    replay_matches = group.get_door_matches(torch.device("cpu"))
    for direction in ("left", "right", "up", "down"):
        torch.testing.assert_close(
            getattr(door_matches, direction), getattr(replay_matches, direction)
        )
    counts = compute_area_distinct_crossing_counts(actions, door_matches, rooms)

    assert counts.tolist() == [1, 0]
    assert outcomes.end_outcomes.area_crossings.tolist() == [1, 0]
    horizontal, vertical = compute_valid_door_match_counts(
        door_matches, torch.tensor([True, False])
    )
    assert horizontal.tolist() == [[1.0]]
    assert vertical.shape == (0, 0)
    horizontal, _ = compute_valid_door_match_counts(door_matches, torch.tensor([False, True]))
    assert horizontal.tolist() == [[0.0]]


def test_distinct_crossings_with_no_doors() -> None:
    actions = Actions(
        room_idx=torch.tensor([[0]], dtype=torch.uint8),
        room_x=torch.tensor([[0]], dtype=torch.int8),
        room_y=torch.tensor([[0]], dtype=torch.int8),
        room_area=torch.tensor([[2]], dtype=torch.uint8),
    )
    door_matches = DoorMatches(
        left=torch.empty((1, 0), dtype=torch.int64),
        right=torch.empty((1, 0), dtype=torch.int64),
        up=torch.empty((1, 0), dtype=torch.int64),
        down=torch.empty((1, 0), dtype=torch.int64),
    )
    counts = compute_area_distinct_crossing_counts(actions, door_matches, [{"doors": []}])
    assert counts.tolist() == [0]


if __name__ == "__main__":
    test_distinct_crossings_count_area_graph_edges_per_map()
    test_distinct_crossings_use_engine_door_order_and_ignore_unplaced_rooms()
    test_distinct_crossings_with_no_doors()
