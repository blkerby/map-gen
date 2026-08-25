import tempfile
from pathlib import Path
from types import SimpleNamespace

import safetensors.torch
import torch

from env import Actions, AREA_COUNT, CandidateSlot, DUMMY_AREA, Engine, EpisodeData
from experience import ExperienceStorage
from generate import get_initial_candidate_batch
from train_config import FeatureConfig


def disabled_features() -> FeatureConfig:
    return FeatureConfig(
        inventory=False,
        temperature=False,
        recommended_candidates=False,
        generation_variable_floats=False,
        lookahead_outcomes=0,
        room_position=False,
        global_room_position=0,
        room_part_furthest_distance=0,
        room_part_save_distance=0,
        room_part_refill_distance=0,
        room_part_frontier_distance=0,
        frontier_mask=False,
        frontier_position=0,
        frontier_orientation=0,
        frontier_kind=0,
        frontier_door_variant=0,
        frontier_occupancy=False,
        frontier_neighbor=False,
        frontier_neighbor_position_embedding=0,
        frontier_neighbor_flags=False,
        connection_reachability=0,
        frontier_connection_reachability=False,
        area_state=False,
        frontier_area=0,
        missing_connect_query=False,
        save_utility_query=False,
        refill_utility_query=False,
        toilet_crossed_room=0,
        known_distance=0,
    )


def one_tile_room(name: str, direction: str) -> dict:
    return {
        "name": name,
        "map": [[1]],
        "doors": [[{"id": 0, "direction": direction, "x": 0, "y": 0, "kind": 0}]],
        "connections": [],
        "missing_connections": [],
        "toilet_crossing_x": [],
    }


def test_direct_balance_targets_match_replay() -> None:
    rooms = [
        one_tile_room("Right", "right"),
        one_tile_room("Left", "left"),
        {
            "name": "Toilet",
            "map": [[1], [1], [0], [0], [0], [0], [0], [0], [1], [1]],
            "toilet_crossing_x": [],
            "special_type": "toilet",
            "doors": [],
            "connections": [],
            "missing_connections": [],
        },
    ]
    engine = Engine(rooms, disabled_features(), 1, 100)
    actions = Actions(
        room_idx=torch.tensor([[2, 0, 1]], dtype=torch.uint8),
        room_x=torch.tensor([[0, 0, 1]], dtype=torch.int8),
        room_y=torch.tensor([[0, 2, 2]], dtype=torch.int8),
        room_area=torch.tensor([[0, 0, 0]], dtype=torch.uint8),
    )
    device = torch.device("cpu")

    direct_doors, direct_toilet = engine.compute_balance_targets(actions, device)

    env = engine.create_environment_group(
        map_size=(4, 12),
        num_envs=1,
        candidate_spatial_cell_size=4,
        area_bounding_box_width=4,
        area_bounding_box_height=12,
        seed=0,
        num_threads=1,
    )
    for step in range(actions.room_idx.shape[1]):
        env.step_known(
            Actions(
                room_idx=actions.room_idx[:, step],
                room_x=actions.room_x[:, step],
                room_y=actions.room_y[:, step],
                room_area=actions.room_area[:, step],
            )
        )
    replay_doors = env.get_door_matches(device)
    env.finish()
    replay_toilet = env.get_outcomes(device, verify_consistency=False).end_outcomes.toilet_crossed_room_idx

    for direction in ("left", "right", "up", "down"):
        assert torch.equal(getattr(direct_doors, direction), getattr(replay_doors, direction))
    assert torch.equal(direct_toilet, replay_toilet)
    assert direct_toilet.tolist() == [0]


def test_environment_group_round_trips_room_area() -> None:
    engine = Engine(
        [
            one_tile_room("Right", "right"),
            one_tile_room("Left", "left"),
        ],
        disabled_features(),
        1,
        100,
    )
    env = engine.create_environment_group(
        map_size=(4, 4),
        num_envs=1,
        candidate_spatial_cell_size=4,
        area_bounding_box_width=4,
        area_bounding_box_height=4,
        seed=0,
        num_threads=1,
    )
    device = torch.device("cpu")

    first = Actions(
        room_idx=torch.tensor([0], dtype=torch.uint8),
        room_x=torch.tensor([0], dtype=torch.int8),
        room_y=torch.tensor([0], dtype=torch.int8),
        room_area=torch.tensor([2], dtype=torch.uint8),
    )
    second = Actions(
        room_idx=torch.tensor([1], dtype=torch.uint8),
        room_x=torch.tensor([1], dtype=torch.int8),
        room_y=torch.tensor([0], dtype=torch.int8),
        room_area=torch.tensor([4], dtype=torch.uint8),
    )
    finish = Actions(
        room_idx=torch.tensor([2], dtype=torch.uint8),
        room_x=torch.tensor([0], dtype=torch.int8),
        room_y=torch.tensor([0], dtype=torch.int8),
        room_area=torch.tensor([DUMMY_AREA], dtype=torch.uint8),
    )

    env.step_known(first)
    env.step_known(second)
    env.step(finish)
    actions = env.get_actions(device)

    assert AREA_COUNT == 6
    assert actions.room_idx.tolist() == [[0, 1, 2]]
    assert actions.room_x.tolist() == [[0, 1, 0]]
    assert actions.room_y.tolist() == [[0, 0, 0]]
    assert actions.room_area.tolist() == [[2, 4, DUMMY_AREA]]


def test_initial_candidate_batch_scores_every_area() -> None:
    room = one_tile_room("Room", "right")
    room["water"] = 1
    engine = Engine(
        [room],
        disabled_features(),
        1,
        100,
    )
    env = engine.create_environment_group(
        map_size=(4, 4),
        num_envs=1,
        candidate_spatial_cell_size=4,
        area_bounding_box_width=4,
        area_bounding_box_height=4,
        seed=0,
        num_threads=1,
    )
    group = SimpleNamespace(
        env=env,
        config=SimpleNamespace(
            temperature=torch.ones([1]),
            recommended_candidates=1,
            vanilla_area_constraint_mask=torch.zeros([1, 6], dtype=torch.bool),
        ),
        candidate_slot=CandidateSlot(env, pin_memory=False),
    )

    batch = get_initial_candidate_batch(group)
    candidates = batch.candidates
    assert candidates.room_idx.shape == (1, AREA_COUNT)
    assert torch.all(candidates.room_idx == candidates.room_idx[:, :1])
    assert torch.all(candidates.room_x == candidates.room_x[:, :1])
    assert torch.all(candidates.room_y == candidates.room_y[:, :1])
    assert candidates.room_area.tolist() == [list(range(AREA_COUNT))]
    assert batch.reward_outcomes.maridia_water.tolist() == [[-1]]
    assert batch.post_candidate_outcomes.maridia_water.tolist() == [
        [[0], [0], [0], [0], [1], [0]]
    ]


def test_initial_candidate_batch_enforces_enabled_ship_area() -> None:
    ship = one_tile_room("Ship", "right")
    ship["special_type"] = "ship"
    engine = Engine([ship], disabled_features(), 1, 100)
    env = engine.create_environment_group(
        map_size=(4, 4),
        num_envs=1,
        candidate_spatial_cell_size=4,
        area_bounding_box_width=4,
        area_bounding_box_height=4,
        seed=0,
        num_threads=1,
    )
    group = SimpleNamespace(
        env=env,
        config=SimpleNamespace(
            temperature=torch.ones([1]),
            recommended_candidates=1,
            vanilla_area_constraint_mask=torch.tensor(
                [[True, False, False, False, False, False]]
            ),
        ),
        candidate_slot=CandidateSlot(env, pin_memory=False),
    )

    candidates = get_initial_candidate_batch(group).candidates
    assert candidates.room_area.tolist() == [[0] + [DUMMY_AREA] * 5]


def test_environment_group_reports_area_outcome_state() -> None:
    two_tile_room = one_tile_room("Right", "right")
    two_tile_room["map"] = [[1, 0, 1]]
    two_tile_room["heat"] = 2
    water_room = one_tile_room("Left", "left")
    water_room["water"] = 3
    engine = Engine(
        [
            two_tile_room,
            water_room,
        ],
        disabled_features(),
        1,
        100,
    )
    env = engine.create_environment_group(
        map_size=(4, 4),
        num_envs=1,
        candidate_spatial_cell_size=4,
        area_bounding_box_width=4,
        area_bounding_box_height=4,
        seed=0,
        num_threads=1,
    )
    device = torch.device("cpu")

    env.step_known(
        Actions(
            room_idx=torch.tensor([0], dtype=torch.uint8),
            room_x=torch.tensor([0], dtype=torch.int8),
            room_y=torch.tensor([0], dtype=torch.int8),
            room_area=torch.tensor([2], dtype=torch.uint8),
        )
    )
    env.step_known(
        Actions(
            room_idx=torch.tensor([1], dtype=torch.uint8),
            room_x=torch.tensor([1], dtype=torch.int8),
            room_y=torch.tensor([0], dtype=torch.int8),
            room_area=torch.tensor([4], dtype=torch.uint8),
        )
    )

    state = env.get_area_outcome_state(device)
    assert state.area_crossings.tolist() == [1]
    assert state.area_size.tolist() == [[0, 0, 2, 0, 1, 0]]
    assert state.area_map_station_count.tolist() == [[0, 0, 0, 0, 0, 0]]

    env.finish()
    outcomes = env.get_outcomes(device, verify_consistency=True)
    assert outcomes.end_outcomes.area_crossings.tolist() == state.area_crossings.tolist()
    assert outcomes.end_outcomes.maridia_water.tolist() == [[1]]
    assert outcomes.end_outcomes.norfair_heat.tolist() == [[1]]
    assert outcomes.step_outcomes.maridia_water.tolist() == [[1]]
    assert outcomes.step_outcomes.norfair_heat.tolist() == [[1]]
    assert outcomes.end_outcomes.area_size.tolist() == state.area_size.tolist()
    assert outcomes.end_outcomes.area_x.tolist() == [[0.0, 0.0, 1.0, 0.0, 1.0, 0.0]]
    assert outcomes.end_outcomes.area_y.tolist() == [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    assert outcomes.end_outcomes.area_map_station_count.tolist() == (
        state.area_map_station_count.tolist()
    )
    assert outcomes.step_outcomes.area_size_bucket.tolist() == [[0, 0, 1, 0, 1, 0]]
    assert outcomes.step_outcomes.area_map_station_count_bucket.tolist() == [
        [0, 0, 0, 0, 0, 0]
    ]
    assert outcomes.step_outcomes.door_match.shape == (1, 0)


def test_experience_storage_round_trips_room_area() -> None:
    episode_data = EpisodeData(
        actions=Actions(
            room_idx=torch.tensor([[0, 1], [1, 2]], dtype=torch.uint8),
            room_x=torch.tensor([[0, 1], [1, 2]], dtype=torch.int8),
            room_y=torch.tensor([[2, 3], [3, 4]], dtype=torch.int8),
            room_area=torch.tensor([[0, 5], [3, DUMMY_AREA]], dtype=torch.uint8),
        ),
        temperature=torch.tensor([1.0, 2.0]),
        recommended_candidates=torch.tensor([8.0, 8.0]),
        generation_variable_floats=torch.empty((2, 0)),
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        storage = ExperienceStorage(
            num_rooms=2,
            data_path=Path(temp_dir),
            episodes_per_file=2,
        )
        storage.store(episode_data)
        loaded = storage.read_files([0], episodes_per_file=2)

    loaded_order = torch.argsort(loaded.actions.room_idx[:, 0])
    episode_order = torch.argsort(episode_data.actions.room_idx[:, 0])
    assert torch.equal(
        loaded.actions.room_idx[loaded_order],
        episode_data.actions.room_idx[episode_order],
    )
    assert torch.equal(
        loaded.actions.room_x[loaded_order],
        episode_data.actions.room_x[episode_order],
    )
    assert torch.equal(
        loaded.actions.room_y[loaded_order],
        episode_data.actions.room_y[episode_order],
    )
    assert torch.equal(
        loaded.actions.room_area[loaded_order],
        episode_data.actions.room_area[episode_order],
    )


def test_experience_storage_rejects_unversioned_files() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        safetensors.torch.save_file(
            {"room_idx": torch.zeros([1, 1], dtype=torch.uint8)},
            str(Path(temp_dir) / "0.safetensors"),
        )
        storage = ExperienceStorage(
            num_rooms=1,
            data_path=Path(temp_dir),
            episodes_per_file=1,
        )
        try:
            storage.read_files([0], episodes_per_file=1)
        except ValueError:
            return
    raise AssertionError("Unversioned experience file was accepted")


def main() -> None:
    test_environment_group_round_trips_room_area()
    test_initial_candidate_batch_scores_every_area()
    test_initial_candidate_batch_enforces_enabled_ship_area()
    test_environment_group_reports_area_outcome_state()
    test_experience_storage_round_trips_room_area()
    test_experience_storage_rejects_unversioned_files()


if __name__ == "__main__":
    main()
