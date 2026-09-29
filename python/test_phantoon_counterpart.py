from types import SimpleNamespace

import torch

from env import AREA_COUNT, Actions, CandidateSlot, Engine
from generate import get_shortlist_candidate_batch
from test_room_area_plumbing import disabled_features, one_tile_room


def test_counterpart_option_reaches_each_environment_worker() -> None:
    rooms = []
    for special_type, directions in [
        (None, ["left", "right", "up", "down"]),
        (None, ["left", "right", "up", "down"]),
        (None, ["left", "right", "up", "down"]),
        ("phantoon_boss", ["right"]),
        ("phantoon_map", ["left"]),
        ("phantoon_save", ["down"]),
        (None, ["up", "down"]),
    ]:
        room = one_tile_room(str(len(rooms)), directions[0])
        room["special_type"] = special_type
        room["map_station"] = special_type == "phantoon_map"
        room["doors"] = [[
            {"id": idx, "direction": direction, "x": 0, "y": 0, "kind": 0}
            for idx, direction in enumerate(directions)
        ]]
        rooms.append(room)
    engine = Engine(rooms, disabled_features(), 1, 100)
    env = engine.create_environment_group(
        map_size=(8, 8),
        num_envs=2,
        candidate_spatial_cell_size=4,
        area_bounding_box_width=8,
        area_bounding_box_height=8,
        seed=0,
        num_threads=2,
    )
    for room_idx, room_x in [([0, 0], [3, 3]), ([3, 4], [2, 4])]:
        env.step(Actions(
            room_idx=torch.tensor(room_idx, dtype=torch.uint8),
            room_x=torch.tensor(room_x, dtype=torch.int8),
            room_y=torch.tensor([3, 3], dtype=torch.int8),
            room_area=torch.tensor([0, 0], dtype=torch.uint8),
        ))
    metadata = engine.get_output_metadata()
    ordinary_variants = [variant for room, variant in metadata.door if room == 6]
    counterpart_variants = [
        next(variant for room, variant in metadata.door if room == counterpart)
        for counterpart in [4, 3]
    ]
    # Ordinary corridor proposals precede every counterpart proposal. Each state
    # has three frontiers; the normal resolver filters incompatible orientations.
    shortlist_actions = torch.tensor([
        [variant * AREA_COUNT for variant in [*ordinary_variants, counterpart] for _ in range(3)]
        for counterpart in counterpart_variants
    ], dtype=torch.int16)
    shortlist_frontiers = torch.tensor([[0, 1, 2] * 3] * 2, dtype=torch.int16)
    group = SimpleNamespace(
        env=env,
        candidate_slot=CandidateSlot(env, pin_memory=False),
        config=SimpleNamespace(
            recommended_candidates=2,
            num_scored_invalid_candidates=9,
            max_candidate_areas_per_placement=1,
            recommended_candidates_same_frontier=False,
            guarantee_phantoon_counterpart=False,
            vanilla_area_constraint_mask=torch.zeros((2, AREA_COUNT), dtype=torch.bool),
        ),
    )
    for enabled in (False, True):
        group.config.guarantee_phantoon_counterpart = enabled
        batch = get_shortlist_candidate_batch(
            group, shortlist_frontiers, shortlist_actions, torch.tensor([9, 9]),
        )
        assert batch.candidates.room_idx.shape == (2, 2)
        assert batch.stats.clean_counts.tolist() == [2, 2]
        assert batch.proposal_rejected.tolist() == [[False, False], [False, False]]
        if enabled:
            assert batch.candidates.room_idx.tolist() == [[4, 6], [3, 6]]
            assert batch.proposal_action_idx[:, 0].tolist() == [
                variant * AREA_COUNT for variant in counterpart_variants
            ]
        else:
            assert batch.candidates.room_idx.tolist() == [[6, 6], [6, 6]]


if __name__ == "__main__":
    test_counterpart_option_reaches_each_environment_worker()
