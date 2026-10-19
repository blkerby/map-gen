from types import SimpleNamespace

import torch

from env import Actions, DoorMatches
from generate import GenerationProfiler
from serve import (
    GenerationRunResult,
    build_door_lookups,
    build_generate_response_data,
    warmup_generate_request,
)
from small_map import build_door_data, build_required_room_data, build_room_part_data
from subarea import build_door_room_lookup


def response_fixture() -> tuple[SimpleNamespace, GenerationRunResult]:
    rooms = []
    for index in range(8):
        doors = []
        if index > 0:
            doors.append({"id": 10, "direction": "left", "x": 0, "y": 0, "kind": 0})
        if index < 7:
            doors.append({"id": 20, "direction": "right", "x": 0, "y": 0, "kind": 0})
        rooms.append(
            {
                "room_id": 100 + index,
                "name": f"Room {index}",
                "map": [[1]],
                "doors": [doors],
                "connections": [],
                "missing_connections": [],
                "toilet_crossing_x": [],
                "map_station": index < 6,
            }
        )
    state = SimpleNamespace(
        rooms=rooms,
        device=torch.device("cpu"),
        profile=False,
        training_config=SimpleNamespace(
            generation=SimpleNamespace(min_area_size=1, max_area_size=3),
        ),
        door_room_lookup=build_door_room_lookup(rooms, torch.device("cpu")),
        door_lookups=build_door_lookups(rooms),
        door_data=build_door_data(rooms),
        room_part_data=build_room_part_data(rooms),
        required_room_data=build_required_room_data(rooms),
    )
    # Placement order and area IDs deliberately differ from room-index order.
    actions = Actions(
        room_idx=torch.tensor([[7, 2, 0, 6, 1, 4, 3, 5]] * 3, dtype=torch.uint8),
        room_x=torch.tensor([[8, 7, 6, 5, 4, 3, 2, 1]] * 3, dtype=torch.uint8),
        room_y=torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]] * 3, dtype=torch.uint8),
        room_area=torch.tensor([[4, 0, 2, 5, 1, 3, 4, 0]] * 3, dtype=torch.uint8),
    )
    step_outcomes = SimpleNamespace(
        door_invalid=torch.tensor([[1], [0], [0]]),
        connection_invalid=torch.zeros([3, 1]),
        return_distance_invalid=torch.zeros([3, 1]),
        toilet_invalid=torch.zeros([3]),
        phantoon_pair_invalid=torch.zeros([3]),
        phantoon_area_invalid=torch.zeros([3]),
        vanilla_area_invalid=torch.zeros([3, 6], dtype=torch.bool),
    )
    end_outcomes = SimpleNamespace(
        area_size=torch.tensor([[2, 1, 1, 1, 2, 1]] * 3),
        area_map_station_count=torch.ones([3, 6], dtype=torch.int64),
        toilet_crossed_room_idx=torch.tensor([-1, 2, -1]),
    )
    matches = torch.arange(7, dtype=torch.int16).expand(3, 7).clone()
    result = GenerationRunResult(
        episode_data=SimpleNamespace(actions=actions),
        outcomes=SimpleNamespace(step_outcomes=step_outcomes, end_outcomes=end_outcomes),
        door_matches=DoorMatches(
            left=matches,
            right=matches.clone(),
            up=torch.empty([3, 0], dtype=torch.int16),
            down=torch.empty([3, 0], dtype=torch.int16),
        ),
        profile_report=[],
    )
    return state, result


def test_response_preserves_generated_areas_and_placements() -> None:
    state, result = response_fixture()
    response = build_generate_response_data(
        state, warmup_generate_request(), GenerationProfiler(False), 0, result
    )
    assert response["stats"] == {"num_generated": 3, "num_pre_valid": 2, "num_valid": 2}
    assert response["rooms"]["id"] == [[107, 102, 100, 106, 101, 104, 103, 105]] * 2
    assert response["rooms"]["area"] == [[4, 0, 2, 5, 1, 3, 4, 0]] * 2
    assert response["rooms"]["x"] == [[8, 7, 6, 5, 4, 3, 2, 1]] * 2
    assert response["rooms"]["y"] == [[1, 2, 3, 4, 5, 6, 7, 8]] * 2
    assert response["toilet_crossing_room_placement_idx"] == [1, -1]
    edges = response["edges"]
    for map_index in range(2):
        ids = response["rooms"]["id"][map_index]
        pairs = {
            tuple(sorted((ids[source], ids[target])))
            for source, target in zip(
                edges["from_room_placement_idx"][map_index],
                edges["to_room_placement_idx"][map_index],
            )
        }
        assert pairs == {(100 + index, 101 + index) for index in range(7)}
    assert all(len(values) == 8 for values in response["rooms"]["subarea"])
    assert all(len(values) == 8 for values in response["rooms"]["subsubarea"])


def test_small_response_prunes_generated_areas_without_reassignment() -> None:
    state, result = response_fixture()
    request = warmup_generate_request().model_copy(
        update={"small_map": True, "min_rooms": 4, "max_rooms": 8, "target_rooms": 6}
    )
    response = build_generate_response_data(state, request, GenerationProfiler(False), 0, result)
    assert response["stats"]["num_valid"] == 2
    area_by_id = dict(zip([107, 102, 100, 106, 101, 104, 103, 105], [4, 0, 2, 5, 1, 3, 4, 0]))
    for map_index, ids in enumerate(response["rooms"]["id"]):
        assert len(ids) == 6
        assert response["rooms"]["area"][map_index] == [area_by_id[room_id] for room_id in ids]
        for field in ("x", "y", "subarea", "subsubarea"):
            assert len(response["rooms"][field][map_index]) == len(ids)
        for source, target in zip(
            response["edges"]["from_room_placement_idx"][map_index],
            response["edges"]["to_room_placement_idx"][map_index],
        ):
            assert abs(ids[target] - ids[source]) == 1


def test_response_handles_no_valid_maps() -> None:
    state, result = response_fixture()
    result.outcomes.step_outcomes.door_invalid.fill_(1)
    response = build_generate_response_data(
        state, warmup_generate_request(), GenerationProfiler(False), 0, result
    )
    assert response["stats"] == {"num_generated": 3, "num_pre_valid": 0, "num_valid": 0}
    assert all(values == [] for values in response["rooms"].values())
    assert all(values == [] for values in response["edges"].values())
    assert response["toilet_crossing_room_placement_idx"] == []


def main() -> None:
    test_response_preserves_generated_areas_and_placements()
    test_small_response_prunes_generated_areas_without_reassignment()
    test_response_handles_no_valid_maps()


if __name__ == "__main__":
    main()
