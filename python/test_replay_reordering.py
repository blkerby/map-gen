from dataclasses import fields
from types import SimpleNamespace

import unittest
from unittest.mock import Mock
import torch

from env import Actions, Engine, EpisodeData
from learn import TrainBatchTask, prepare_feature_batches, prepare_train_batch_task
from test_room_area_plumbing import disabled_features, one_tile_room
from train_config import GENERATION_VARIABLE_FLOAT_FIELDS


def replay_fixture(batch_size: int, force_ship: bool):
    rooms = []
    for index in range(6):
        room = one_tile_room(str(index), "left")
        directions = ["left", "right", "up", "down"] if index == 0 else ["left", "right"]
        room["doors"] = [[
            {"id": i, "direction": direction, "x": 0, "y": 0, "kind": 0}
            for i, direction in enumerate(directions)
        ]]
        rooms.append(room)
    rooms[1]["special_type"] = "ship"
    feature_config = disabled_features().model_copy(update={"room_area": True})
    engine = Engine(rooms, feature_config, 1, 100)
    env = engine.create_environment_group(
        map_size=(10, 10), num_envs=batch_size, candidate_spatial_cell_size=4,
        area_bounding_box_width=10, area_bounding_box_height=10,
        seed=0, num_threads=2,
    )
    variables = torch.zeros((batch_size, len(GENERATION_VARIABLE_FLOAT_FIELDS)))
    variables[:, GENERATION_VARIABLE_FLOAT_FIELDS.index("force_ship_in_crateria")] = force_ship
    episode = EpisodeData(
        actions=Actions(
            room_idx=torch.tensor([[0, 1, 3, 2, 6, 6]], dtype=torch.uint8).repeat(batch_size, 1),
            room_x=torch.tensor([[3, 2, 4, 5, 0, 0]], dtype=torch.int8).repeat(batch_size, 1),
            room_y=torch.tensor([[3, 3, 3, 3, 0, 0]], dtype=torch.int8).repeat(batch_size, 1),
            room_area=torch.tensor([[2, 2, 2, 2, 6, 6]], dtype=torch.uint8).repeat(batch_size, 1),
        ),
        temperature=torch.ones(batch_size),
        recommended_candidates=torch.ones(batch_size, dtype=torch.int64),
        generation_variable_floats=variables,
    )
    return env, episode, feature_config


def run_replay(env, episode, probabilities):
    reordering = env.create_replay_reordering(episode, probabilities)
    env.clear()
    for _ in range(episode.actions.room_idx.shape[1]):
        env.step_reordered(reordering)
    return env.get_actions(torch.device("cpu"))


def test_reordering_skips_rejections_and_falls_back_without_losing_actions():
    env, episode, _ = replay_fixture(64, force_ship=True)
    probabilities = torch.cat((torch.zeros(32), torch.ones(32)))
    actual = run_replay(env, episode, probabilities)
    # The left ship placement is legal but rejected by the area constraint. The
    # far-right room cannot attach yet; the near-right corridor is the only clean
    # alternative. Once both right rooms are placed, fall back to the rejected ship.
    assert actual.room_idx[:32].tolist() == [[0, 1, 3, 2, 6, 6]] * 32
    assert actual.room_idx[32:].tolist() == [[0, 3, 2, 1, 6, 6]] * 32
    for field in fields(Actions):
        original = getattr(episode.actions, field.name)
        reordered = getattr(actual, field.name)
        torch.testing.assert_close(reordered[:32], original[:32])
        torch.testing.assert_close(reordered[32:], original[32:, [0, 2, 3, 1, 4, 5]])
    assert episode.actions.room_idx.tolist() == [[0, 1, 3, 2, 6, 6]] * 64


def test_random_selection_explores_both_legal_orders_and_preserves_dependencies():
    env, episode, _ = replay_fixture(128, force_ship=False)
    actual = run_replay(env, episode, torch.ones(128))
    second = actual.room_idx[:, 1]
    assert set(second.tolist()) == {1, 3}
    assert 35 < (second == 1).sum().item() < 93
    for row in actual.room_idx.tolist():
        assert row.index(3) < row.index(2)
        assert row[0] == 0 and row[-2:] == [6, 6]


def test_reordered_features_match_direct_replay_and_final_outcomes():
    env, episode, features = replay_fixture(2, force_ship=True)
    reordering = env.create_replay_reordering(episode, torch.ones(2))
    config = SimpleNamespace(train=SimpleNamespace(sample_period=1), features=features)
    batches, _, _, _ = prepare_feature_batches(
        config, episode, None, env, 6, False, None, reordering,
    )
    reordered = env.get_actions(torch.device("cpu"))
    env.finish()
    augmented_outcomes = env.get_outcomes(torch.device("cpu"), verify_consistency=True)
    direct_episode = EpisodeData(
        actions=reordered, temperature=episode.temperature,
        recommended_candidates=episode.recommended_candidates,
        generation_variable_floats=episode.generation_variable_floats,
    )
    direct_batches, _, _, _ = prepare_feature_batches(
        config, direct_episode, None, env, 6, False, None, None,
    )
    for augmented, direct in zip(batches, direct_batches, strict=True):
        torch.testing.assert_close(
            augmented.features.global_features.room_placed,
            direct.features.global_features.room_placed,
        )
        torch.testing.assert_close(
            augmented.features.global_features.room_area,
            direct.features.global_features.room_area,
        )
    env.finish()
    direct_outcomes = env.get_outcomes(torch.device("cpu"), verify_consistency=True)
    for group in ("step_outcomes", "end_outcomes"):
        augmented = getattr(augmented_outcomes, group)
        direct = getattr(direct_outcomes, group)
        for field in fields(augmented):
            torch.testing.assert_close(getattr(augmented, field.name), getattr(direct, field.name))


def test_reordering_rejects_invalid_probabilities():
    env, episode, _ = replay_fixture(2, force_ship=False)
    for probability in [-0.1, 1.1, float("nan")]:
        with unittest.TestCase().assertRaisesRegex(ValueError, "probabilities"):
            env.create_replay_reordering(episode, torch.full((2,), probability))


class ReplayReorderingTest(unittest.TestCase):
    def test_training_batch_uses_augmented_actions_and_leaves_stored_episode_unchanged(self):
        env, episode, features = replay_fixture(2, force_ship=True)
        config = SimpleNamespace(
            train=SimpleNamespace(sample_period=1, episodes_per_file=2, hist_c=1.0),
            features=features,
        )
        context = SimpleNamespace(
            config=config,
            step_config=SimpleNamespace(train=SimpleNamespace(batch_size=2, reorder_probability=1.0)),
            device=torch.device("cpu"), train_batch_envs=[env], episode_length=6,
            experience=SimpleNamespace(sample=Mock(return_value=episode)),
        )
        batch = prepare_train_batch_task(
            context=context,
            task=TrainBatchTask(kind="replay", start=None, env_index=0),
            fresh_episode_data=episode,
            fresh_outcomes=Mock(), fresh_proposal_data=Mock(), generated_feature_data=Mock(),
        )
        self.assertEqual(batch.episode_data.actions.room_idx.tolist(), [[0, 3, 2, 1, 6, 6]] * 2)
        self.assertEqual(episode.actions.room_idx.tolist(), [[0, 1, 3, 2, 6, 6]] * 2)
        self.assertEqual(len(batch.feature_batches), 6)
        self.assertEqual(batch.kind, "replay")

    def test_rejection_and_fallback(self):
        test_reordering_skips_rejections_and_falls_back_without_losing_actions()

    def test_random_orders(self):
        test_random_selection_explores_both_legal_orders_and_preserves_dependencies()

    def test_features_and_outcomes(self):
        test_reordered_features_match_direct_replay_and_final_outcomes()

    def test_probability_validation(self):
        test_reordering_rejects_invalid_probabilities()


if __name__ == "__main__":
    unittest.main()
