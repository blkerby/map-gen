import unittest
from dataclasses import replace
from pathlib import Path

import torch
from pydantic import ValidationError

from env import AREA_PAIR_COUNT, Actions, Engine, FeatureSlot
from generate import compute_expected_reward
from model import FrontierModel
from model_loading import frontier_model_kwargs
from test_area_rewards import area_predictions, unknown_outcomes, zero_generate_config
from test_room_area_plumbing import disabled_features
from train_config import Config, GENERATION_VARIABLE_FLOAT_FIELDS, validate_config


def grid_rooms(edges: list[tuple[int, int]]) -> list[dict]:
    rooms = [
        {
            "name": str(i),
            "map": [[1]],
            "doors": [],
            "connections": [],
            "missing_connections": [],
            "toilet_crossing_x": [],
        }
        for i in range(6)
    ]
    for source, target in edges:
        directions = ("right", "left") if target - source == 1 else ("down", "up")
        for room, direction in zip((source, target), directions, strict=True):
            doors = rooms[room]["doors"]
            doors.append([{"id": len(doors), "direction": direction, "x": 0, "y": 0, "kind": 0}])
    for room in rooms:
        room["connections"] = [
            [a, b] for a in range(len(room["doors"])) for b in range(len(room["doors"])) if a != b
        ]
    return rooms


def replay_grid(rooms: list[dict], areas: list[int], lookahead_width: int):
    feature_config = disabled_features().model_copy(update={"lookahead_outcomes": lookahead_width})
    engine = Engine(rooms, feature_config, 1, 100)
    group = engine.create_environment_group(
        map_size=(8, 8),
        num_envs=1,
        candidate_spatial_cell_size=4,
        area_bounding_box_width=8,
        area_bounding_box_height=8,
        seed=0,
        num_threads=1,
    )
    for room in reversed(range(len(areas))):
        group.step_known(
            Actions(
                room_idx=torch.tensor([room], dtype=torch.uint8),
                room_x=torch.tensor([room % 3], dtype=torch.int8),
                room_y=torch.tensor([room // 3], dtype=torch.int8),
                room_area=torch.tensor([areas[room]], dtype=torch.uint8),
            )
        )
    return engine, group


class AreaConnectionsTest(unittest.TestCase):
    def test_terminal_graph_counts_and_pair_order(self) -> None:
        tree = [(0, 1), (1, 2), (0, 3), (3, 4), (4, 5)]
        cases = (
            (tree, list(range(6)), 5),
            (tree + [(2, 5)], list(range(6)), 6),
            (tree + [(2, 5), (1, 4)], [0, 1, 2, 0, 1, 2], 2),
            (tree, [0] * 6, 0),
            (tree, [0, 1], 1),
            (tree, [], 0),
        )
        counts = []
        for edges, areas, expected_count in cases:
            with self.subTest(edges=edges, areas=areas):
                _, group = replay_grid(grid_rooms(edges), areas, 0)
                current = group.get_current_feature_outcomes(torch.device("cpu"), 0, 1)
                if not areas:
                    self.assertFalse(current.area_connections.any())
                    counts.append(0)
                    continue
                group.finish()
                final = group.get_outcomes(torch.device("cpu"), verify_consistency=True)
                connections = final.step_outcomes.area_connections
                self.assertEqual(connections.dtype, torch.bool)
                self.assertEqual(connections.shape, (1, AREA_PAIR_COUNT))
                expected = {
                    tuple(sorted((areas[a], areas[b])))
                    for a, b in edges
                    if max(a, b) < len(areas) and areas[a] != areas[b]
                }
                self.assertEqual(
                    connections.tolist(),
                    [[(a, b) in expected for a in range(6) for b in range(a + 1, 6)]],
                )
                torch.testing.assert_close(current.area_connections, connections)
                counts.append(connections.sum().item())
                self.assertEqual(counts[-1], expected_count)
        self.assertEqual(sum(counts) / len(counts), 14 / 6)

    def test_connection_state_and_head_with_lookahead_enabled_or_disabled(self) -> None:
        rooms = grid_rooms([(0, 1)])
        for width in (0, 4):
            with self.subTest(lookahead=width):
                engine, group = replay_grid(rooms, [0, 1], width)
                outcomes = group.get_current_feature_outcomes(torch.device("cpu"), 0, 1)
                features = group.extract_features(
                    FeatureSlot(group, pin_memory=False),
                    torch.zeros(1),
                    False,
                    torch.zeros(1),
                    False,
                    torch.zeros((1, len(GENERATION_VARIABLE_FLOAT_FIELDS))),
                    False,
                    outcomes,
                    width > 0,
                    0,
                    1,
                )
                self.assertTrue(features.global_features.area_connections[0, 0])
                self.assertEqual(features.global_features.area_connections.sum().item(), 1)
                self.assertEqual(
                    features.global_features.lookahead_door_match.shape[-1] > 0, width > 0
                )
                config = Config.model_validate_json(Path("configs/debug.json").read_text())
                config.features = disabled_features().model_copy(
                    update={"lookahead_outcomes": width}
                )
                model = FrontierModel(**frontier_model_kwargs(config, rooms, engine))
                preds = model(features, return_proposal_state=False)
                self.assertEqual(preds.area_connection_logits.shape, (1, 1, 15))
                self.assertEqual(preds.area_connection_logits.dtype, torch.float32)
                self.assertEqual(torch.count_nonzero(preds.area_connection_logits).item(), 0)

    def test_reward_uses_post_candidate_connections_and_signed_weights(self) -> None:
        logits = torch.full((2, 2, 15), -100.0)
        logits[:, :, 1] = 0.0
        logits.requires_grad_()
        predictions = replace(area_predictions(), area_connection_logits=logits)
        connections = torch.zeros_like(logits, dtype=torch.bool)
        connections[:, 0, 0] = True
        outcomes = unknown_outcomes()
        for coefficient in (0.0, 2.0, -2.0, torch.tensor([2.0, -3.0])):
            reward = compute_expected_reward(
                predictions,
                outcomes,
                zero_generate_config(reward_area_distinct_crossing=coefficient),
                connections,
            )
            weights = torch.as_tensor(coefficient).reshape(-1, 1)
            torch.testing.assert_close(reward, weights * torch.tensor([[1.5, 0.5], [1.5, 0.5]]))
        gradient = torch.autograd.grad(reward.sum(), logits)[0]
        self.assertEqual(torch.count_nonzero(gradient[connections]).item(), 0)

    def test_required_config_fields_and_signed_reward(self) -> None:
        config = Config.model_validate_json(Path("configs/debug.json").read_text())
        self.assertEqual(config.generation.reward_area_distinct_crossing, 0.0)
        self.assertEqual(config.train.area_distinct_crossing_weight, 1.0)
        for value in (-3.0, 0.0, 2.0, {"linear": [-2.0, 3.0]}):
            data = config.model_dump(mode="json")
            data["generation"]["reward_area_distinct_crossing"] = value
            validate_config(Config.model_validate(data))
        for section, field in (
            ("generation", "reward_area_distinct_crossing"),
            ("train", "area_distinct_crossing_weight"),
        ):
            data = config.model_dump(mode="json")
            del data[section][field]
            with self.assertRaises(ValidationError):
                Config.model_validate(data)
        config.train.area_distinct_crossing_weight = -1.0
        with self.assertRaisesRegex(ValueError, "area_distinct_crossing_weight"):
            validate_config(config)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
