import json
from pathlib import Path

import torch

from env import AREA_COUNT, Actions, GenerateConfig, StepOutcomes
from generate import (
    apply_candidate_area_balance_scores,
    apply_candidate_toilet_balance_score,
    balance_reward,
    compute_expected_reward,
)
from model import Predictions
from train import (
    VANILLA_AREA_SPECIAL_ROOM_TYPES,
    compute_unforced_special_room_area_ss,
    create_generate_config,
)
from train_config import (
    Config,
    GENERATION_VARIABLE_FLOAT_FIELDS,
    HEAT_WATER_FAMILIES,
    HEAT_WATER_TARGET_AREAS,
    VANILLA_AREA_CONDITION_FIELDS,
    instantiate_scheduleable_config,
)


def zero_generate_config(**rewards) -> GenerateConfig:
    values = {
        "reward_phantoon_pair": 0.0,
        "reward_phantoon_area": 0.0,
        "reward_area_crossing": 0.0,
        "reward_area_size_valid": 0.0,
        "reward_area_map_station": 0.0,
        "reward_area_x": 0.0,
        "reward_area_y": 0.0,
    }
    values.update(rewards)
    generation_variables = torch.zeros([1, len(GENERATION_VARIABLE_FLOAT_FIELDS)])
    area_probability = torch.full([1, 1, AREA_COUNT], 1.0 / AREA_COUNT)
    return GenerateConfig(
        episode_length=1,
        recommended_candidates=2,
        shortlist_candidates=2,
        num_scored_invalid_candidates=1,
        max_candidate_areas_per_placement=2,
        recommended_candidates_same_frontier=False,
        gpu_prefetch_batches=0,
        temperature=torch.ones([1]),
        proposal_temperature=torch.ones([1]),
        balance_price_limit=20.0,
        reward_door=0.0,
        reward_connection=0.0,
        reward_toilet=0.0,
        reward_phantoon_pair=values["reward_phantoon_pair"],
        reward_phantoon_area=values["reward_phantoon_area"],
        reward_vanilla_area=torch.zeros([1, AREA_COUNT]),
        reward_frontier=0.0,
        reward_graph_diameter=0.0,
        preferred_area_probability=torch.full([1, 2, 3], 1.0 / AREA_COUNT),
        reward_save_distance=0.0,
        reward_refill_distance=0.0,
        reward_missing_connect_utility=0.0,
        reward_area_crossing=values["reward_area_crossing"],
        reward_area_size_valid=values["reward_area_size_valid"],
        reward_area_map_station=values["reward_area_map_station"],
        reward_area_x=values["reward_area_x"],
        reward_area_y=values["reward_area_y"],
        target_area_rooms=torch.ones([1, AREA_COUNT]) / AREA_COUNT,
        target_area_x=torch.zeros([1, AREA_COUNT]),
        target_area_y=torch.zeros([1, AREA_COUNT]),
        vanilla_area_constraint_mask=torch.zeros([1, AREA_COUNT], dtype=torch.bool),
        area_balance_probability=area_probability,
        area_balance_dual_mask=torch.ones([1, 1], dtype=torch.bool),
        effective_target_area_rooms=area_probability.sum(dim=1),
        generation_variable_floats=generation_variables,
        log_temperature_model=torch.zeros([1]),
        log_recommended_candidates_model=torch.zeros([1]),
        generation_variable_floats_model=generation_variables,
        candidate_log_temperature_model=torch.zeros([1, 2]),
        candidate_log_recommended_candidates_model=torch.zeros([1, 2]),
        candidate_generation_variable_floats_model=torch.zeros(
            [1, 2, len(GENERATION_VARIABLE_FLOAT_FIELDS)]
        ),
        distance_proximity_scale=1.0,
        autocast=False,
    )


def area_predictions() -> Predictions:
    batch, candidate, door, connection, room_part = 1, 2, 3, 4, 5
    return Predictions(
        door_invalid=torch.zeros([batch, candidate, door]),
        connection_invalid=torch.zeros([batch, candidate, connection]),
        toilet_invalid=torch.zeros([batch, candidate]),
        phantoon_pair_invalid=torch.zeros([batch, candidate]),
        phantoon_area_invalid=torch.zeros([batch, candidate]),
        vanilla_area_invalid=torch.zeros([batch, candidate, AREA_COUNT]),
        balance_score=torch.zeros([batch, candidate, door]),
        area_balance_score=torch.zeros([batch, candidate, room_part]),
        toilet_balance_score=torch.zeros([batch, candidate]),
        avg_frontiers=torch.zeros([batch, candidate]),
        graph_diameter=torch.zeros([batch, candidate]),
        save_to_room_utility=torch.zeros([batch, candidate, room_part]),
        save_from_room_utility=torch.zeros([batch, candidate, room_part]),
        refill_to_room_utility=torch.zeros([batch, candidate, room_part]),
        refill_from_room_utility=torch.zeros([batch, candidate, room_part]),
        missing_connect_utility=torch.zeros([batch, candidate, connection]),
        area_crossings=torch.tensor([[2.0, 0.0]]),
        area_size=torch.zeros([batch, candidate, AREA_COUNT, 3]),
        area_map_station_count=torch.zeros([batch, candidate, AREA_COUNT, 3]),
        area_x=torch.zeros([batch, candidate, AREA_COUNT]),
        area_y=torch.zeros([batch, candidate, AREA_COUNT]),
        proposal_state=torch.empty([0]),
        proposal_row_snapshot_idx=torch.empty([0], dtype=torch.int64),
        proposal_row_frontier_idx=torch.empty([0], dtype=torch.int64),
    )


def unknown_outcomes() -> StepOutcomes:
    return StepOutcomes(
        door_invalid=torch.full([1, 2, 3], -1.0),
        connection_invalid=torch.full([1, 2, 4], -1.0),
        toilet_invalid=torch.full([1, 2], -1.0),
        phantoon_pair_invalid=torch.full([1, 2], -1.0),
        phantoon_area_invalid=torch.full([1, 2], -1.0),
        vanilla_area_invalid=torch.full([1, 2, AREA_COUNT], -1.0),
        area_size_bucket=torch.full([1, 2, AREA_COUNT], -1.0),
        area_map_station_count_bucket=torch.full([1, 2, AREA_COUNT], -1.0),
        maridia_water=torch.full([1, 2, 3], -1.0),
        norfair_heat=torch.full([1, 2, 3], -1.0),
        door_match=torch.full([1, 2, 3], -1.0),
    )


def test_ordinary_area_rewards_are_unchanged() -> None:
    predictions = area_predictions()
    outcomes = unknown_outcomes()
    assert torch.equal(
        compute_expected_reward(predictions, outcomes, zero_generate_config()),
        torch.zeros([1, 2]),
    )

    predictions.phantoon_pair_invalid = torch.tensor([[0.0, 1.0]])
    reward = compute_expected_reward(
        predictions,
        outcomes,
        zero_generate_config(reward_phantoon_pair=2.0),
    )
    torch.testing.assert_close(
        reward,
        2.0 * torch.nn.functional.logsigmoid(-predictions.phantoon_pair_invalid),
    )

    predictions = area_predictions()
    reward = compute_expected_reward(
        predictions,
        outcomes,
        zero_generate_config(
            reward_area_crossing=5.0,
            reward_area_size_valid=7.0,
            reward_area_map_station=11.0,
        ),
    )
    middle_log_probability = torch.log_softmax(torch.zeros([3]), dim=0)[1]
    expected = (
        -5.0 * torch.tensor([[2.0, 0.0]])
        + 7.0 * AREA_COUNT * middle_log_probability
        + 11.0 * AREA_COUNT * middle_log_probability
    )
    torch.testing.assert_close(reward, expected)


def test_candidate_area_balance_uses_exact_placed_room_price() -> None:
    score_table = torch.arange(12, dtype=torch.float32).reshape(1, 2, AREA_COUNT)
    scores = apply_candidate_area_balance_scores(
        predicted_score=torch.tensor([[[100.0, 2.0], [3.0, 100.0]]]),
        room_placed=torch.tensor([[[True, False], [False, True]]]),
        exempt_room=torch.zeros([1, 2], dtype=torch.bool),
        candidates=Actions(
            room_idx=torch.tensor([[0, 1]]),
            room_x=torch.zeros([1, 2], dtype=torch.int8),
            room_y=torch.zeros([1, 2], dtype=torch.int8),
            room_area=torch.tensor([[4, 2]]),
        ),
        score_table=score_table,
    )
    assert torch.equal(scores, torch.tensor([[[4.0, 2.0], [3.0, 8.0]]]))


def test_candidate_toilet_balance_uses_exact_known_crossing_price() -> None:
    scores = apply_candidate_toilet_balance_score(
        predicted_score=torch.tensor([[10.0, 20.0, 30.0]]),
        crossed_room_idx=torch.tensor([[-1, 2, 0]]),
        score_table=torch.tensor([[1.0, 2.0, 3.0]]),
    )

    assert torch.equal(scores, torch.tensor([[10.0, 3.0, 1.0]]))


def test_known_valid_door_match_uses_exact_price() -> None:
    reward = balance_reward(
        balance_score=torch.tensor([[[3.0, 5.0]]]),
        door_invalid=torch.zeros([1, 1, 2]),
        known_invalid=torch.tensor([[[0.0, 1.0]]]),
    )

    assert torch.equal(reward, torch.tensor([[[-3.0, 0.0]]]))


def test_training_samples_tiered_preferred_probabilities() -> None:
    torch.manual_seed(0)
    rooms = json.loads(Path("room_definitions/zebes.json").read_text())
    config = instantiate_scheduleable_config(
        Config.model_validate_json(Path("configs/zebes.json").read_text()), 0
    )
    generate_config = create_generate_config(
        config=config,
        rooms=rooms,
        episode_length=4,
        num_envs=4096,
        device=torch.device("cpu"),
        ignore_scores=False,
    )

    baseline = generate_config.target_area_rooms / len(rooms)
    for family_idx, (family, preferred_area) in enumerate(
        zip(HEAT_WATER_FAMILIES, HEAT_WATER_TARGET_AREAS, strict=True)
    ):
        probabilities = generate_config.preferred_area_probability[:, family_idx]
        family_baseline = baseline[:, preferred_area]
        assert torch.all(probabilities[:, 1:] >= probabilities[:, :-1])
        assert torch.all(probabilities <= 0.75)
        assert torch.all(probabilities >= family_baseline.unsqueeze(1))
        active = probabilities[:, 2] > family_baseline
        assert 0.47 < active.to(torch.float32).mean() < 0.53
        indices = [
            GENERATION_VARIABLE_FLOAT_FIELDS.index(f"{family}_preferred_probability_{tier}")
            for tier in range(1, 4)
        ]
        torch.testing.assert_close(
            generate_config.generation_variable_floats[:, indices], probabilities
        )
    torch.testing.assert_close(
        generate_config.area_balance_probability.sum(dim=-1),
        torch.ones((4096, len(rooms))),
    )
    torch.testing.assert_close(
        generate_config.effective_target_area_rooms.sum(dim=-1),
        torch.full((4096,), float(len(rooms))),
    )


def test_unforced_special_room_area_ss_excludes_forced_episodes() -> None:
    episode_count = room_count = AREA_COUNT
    actions = Actions(
        room_idx=torch.arange(room_count).repeat(episode_count, 1),
        room_x=torch.zeros([episode_count, room_count], dtype=torch.int8),
        room_y=torch.zeros([episode_count, room_count], dtype=torch.int8),
        room_area=torch.arange(episode_count, dtype=torch.int8).unsqueeze(1).repeat(1, room_count),
    )
    variables = torch.zeros([episode_count, len(GENERATION_VARIABLE_FLOAT_FIELDS)])
    force_idx = GENERATION_VARIABLE_FLOAT_FIELDS.index(VANILLA_AREA_CONDITION_FIELDS[0])
    variables[1:, force_idx] = 1.0
    rooms = [{"special_type": value} for value in VANILLA_AREA_SPECIAL_ROOM_TYPES]
    torch.testing.assert_close(
        compute_unforced_special_room_area_ss(actions, variables, rooms),
        torch.tensor(11.0 / 36.0),
    )


def main() -> None:
    test_ordinary_area_rewards_are_unchanged()
    test_candidate_area_balance_uses_exact_placed_room_price()
    test_candidate_toilet_balance_uses_exact_known_crossing_price()
    test_known_valid_door_match_uses_exact_price()
    test_training_samples_tiered_preferred_probabilities()
    test_unforced_special_room_area_ss_excludes_forced_episodes()


if __name__ == "__main__":
    main()
