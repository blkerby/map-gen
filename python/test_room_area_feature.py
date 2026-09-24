from types import SimpleNamespace

import unittest
import torch

from env import (
    AREA_COUNT,
    DUMMY_AREA,
    Actions,
    CandidateSlot,
    Engine,
    FeatureSlot,
    extract_candidate_features,
)
from features import RoomAreaFeature
from generate import get_initial_candidate_batch
from test_room_area_plumbing import disabled_features, one_tile_room
from train_config import GENERATION_VARIABLE_FLOAT_FIELDS


def create_env(rooms: list[dict], enabled: bool):
    config = disabled_features().model_copy(update={"room_area": enabled})
    engine = Engine(rooms, config, 1, 100)
    return engine.create_environment_group(
        map_size=(8, 8),
        num_envs=2,
        candidate_spatial_cell_size=4,
        area_bounding_box_width=8,
        area_bounding_box_height=8,
        seed=0,
        num_threads=2,
    )


def current_features(env, slot):
    return env.extract_features(
        slot,
        torch.zeros(2),
        False,
        torch.zeros(2),
        False,
        torch.zeros((2, len(GENERATION_VARIABLE_FLOAT_FIELDS))),
        False,
        env.get_current_feature_outcomes(torch.device("cpu"), 0, 2),
        False,
        0,
        2,
    )


def test_room_area_feature_distinguishes_room_identity_and_unplaced() -> None:
    features = SimpleNamespace(
        global_features=SimpleNamespace(
            room_area=torch.tensor([[2, DUMMY_AREA, 4], [4, DUMMY_AREA, 2]], dtype=torch.uint8)
        )
    )
    feature = RoomAreaFeature()
    encoded = feature(features, torch.float32).view(2, 3, AREA_COUNT)
    assert encoded.sum(2).tolist() == [[1, 0, 1], [1, 0, 1]]
    assert encoded[0, 0, 2] == encoded[0, 2, 4] == 1
    assert not torch.equal(encoded[0], encoded[1])
    assert feature(features, torch.bfloat16).dtype == torch.bfloat16


def check_candidate_room_areas_match_replay_and_padding(enabled: bool) -> None:
    # The ship constraint produces one real candidate and five padding slots.
    ship = one_tile_room("Ship", "right")
    ship["special_type"] = "ship"
    env = create_env([ship, one_tile_room("Other", "left")], enabled)
    slot = FeatureSlot(env=env, pin_memory=False)
    group = SimpleNamespace(
        env=env,
        candidate_slot=CandidateSlot(env=env, pin_memory=False),
        config=SimpleNamespace(
            temperature=torch.ones(2),
            recommended_candidates=1,
            vanilla_area_constraint_mask=torch.tensor(
                [[True, False, False, False, False, False]] * 2
            ),
        ),
    )
    before = current_features(env, slot)
    assert before.global_features.room_area.shape == (2, 2 * enabled)
    assert torch.all(before.global_features.room_area == DUMMY_AREA)
    batch = get_initial_candidate_batch(group)
    n, k = batch.candidates.room_idx.shape
    features = extract_candidate_features(
        env,
        batch.candidates,
        torch.zeros(n, k),
        False,
        torch.zeros(n, k),
        False,
        torch.zeros(n, k, len(GENERATION_VARIABLE_FLOAT_FIELDS)),
        False,
        batch.post_candidate_outcomes,
        False,
        batch.feature_requirements,
        slot,
    )
    areas = features.global_features.room_area.clone().view(n, k, 2 * enabled)
    assert areas.shape == (n, k, 2 * enabled)
    assert torch.equal(
        features.to(torch.device("cpu")).global_features.room_area, areas.flatten(0, 1)
    )
    if enabled:
        for e in range(n):
            for c in range(k):
                r = int(batch.candidates.room_idx[e, c])
                expected = torch.full((2,), DUMMY_AREA, dtype=torch.uint8)
                if r < 2:
                    expected[r] = batch.candidates.room_area[e, c]
                assert torch.equal(areas[e, c], expected)
    assert torch.all(current_features(env, slot).global_features.room_area == DUMMY_AREA)
    # Candidate 0 must equal the state obtained by actually taking that action.
    env.step(batch.candidates.select(torch.zeros(n, dtype=torch.int64)))
    torch.testing.assert_close(current_features(env, slot).global_features.room_area, areas[:, 0])
    replay = create_env([ship, one_tile_room("Other", "left")], enabled)
    replay.step_known(batch.candidates.select(torch.zeros(n, dtype=torch.int64)))
    torch.testing.assert_close(
        current_features(
            replay, FeatureSlot(env=replay, pin_memory=False)
        ).global_features.room_area,
        areas[:, 0],
    )
    env.clear()
    assert torch.all(current_features(env, slot).global_features.room_area == DUMMY_AREA)


def test_closed_one_door_room_retains_its_area_feature() -> None:
    env = create_env([one_tile_room("Right", "right"), one_tile_room("Left", "left")], True)
    slot = FeatureSlot(env=env, pin_memory=False)
    for room, x, areas in [(0, 1, [2, 4]), (1, 2, [2, 4])]:
        env.step(
            Actions(
                room_idx=torch.full((2,), room, dtype=torch.uint8),
                room_x=torch.full((2,), x, dtype=torch.int8),
                room_y=torch.ones(2, dtype=torch.int8),
                room_area=torch.tensor(areas, dtype=torch.uint8),
            )
        )
    assert current_features(env, slot).global_features.room_area.tolist() == [[2, 2], [4, 4]]


class RoomAreaFeatureTest(unittest.TestCase):
    def test_encoding(self):
        test_room_area_feature_distinguishes_room_identity_and_unplaced()

    def test_enabled_candidate_replay_and_padding(self):
        check_candidate_room_areas_match_replay_and_padding(True)

    def test_disabled_candidate_replay_and_padding(self):
        check_candidate_room_areas_match_replay_and_padding(False)

    def test_closed_room(self):
        test_closed_one_door_room_retains_its_area_feature()


if __name__ == "__main__":
    unittest.main()
