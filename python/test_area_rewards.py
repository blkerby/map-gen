import json
from pathlib import Path

import torch

from env import Actions, GenerateConfig, StepOutcomes, area_balance_exempt_room_mask
from generate import apply_candidate_area_balance_scores, compute_expected_reward
from train import (
    VANILLA_AREA_SPECIAL_ROOM_TYPES,
    compute_heat_water_tier_tile_counts,
    compute_unforced_special_room_area_ss,
    create_generate_config,
    variable_float_metric_value,
)
from model import Predictions
from train_config import (
    Config,
    GENERATION_VARIABLE_FLOAT_FIELDS,
    VANILLA_AREA_CONDITION_FIELDS,
    VANILLA_AREA_REWARD_FIELDS,
    instantiate_scheduleable_config,
)


def zero_generate_config(**rewards) -> GenerateConfig:
    values = {
        "reward_phantoon_pair": 0.0,
        "reward_phantoon_area": 0.0,
        "reward_area_balance": 0.0,
        "reward_area_crossing": 0.0,
        "reward_area_size_valid": 0.0,
        "reward_area_map_station": 0.0,
        "reward_area_tiles": 0.0,
        "reward_area_x": 0.0,
        "reward_area_y": 0.0,
        "reward_maridia_water": torch.zeros([1, 3]),
        "reward_norfair_heat": torch.zeros([1, 3]),
    }
    values.update(rewards)
    generation_variable_floats = torch.zeros([1, len(GENERATION_VARIABLE_FLOAT_FIELDS)])
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
        reward_door=0.0,
        reward_connection=0.0,
        reward_toilet=0.0,
        reward_phantoon_pair=values["reward_phantoon_pair"],
        reward_phantoon_area=values["reward_phantoon_area"],
        reward_vanilla_area=torch.zeros([1, 6]),
        vanilla_area_constraint_mask=torch.zeros([1, 6], dtype=torch.bool),
        reward_balance=0.0,
        reward_area_balance=values["reward_area_balance"],
        reward_toilet_balance=0.0,
        reward_frontier=0.0,
        reward_graph_diameter=0.0,
        reward_maridia_water=values["reward_maridia_water"],
        reward_norfair_heat=values["reward_norfair_heat"],
        reward_save_distance=0.0,
        reward_refill_distance=0.0,
        reward_missing_connect_utility=0.0,
        reward_area_crossing=values["reward_area_crossing"],
        reward_area_size_valid=values["reward_area_size_valid"],
        reward_area_map_station=values["reward_area_map_station"],
        reward_area_tiles=values["reward_area_tiles"],
        reward_area_x=values["reward_area_x"],
        reward_area_y=values["reward_area_y"],
        target_area_tiles=torch.zeros([1, 6]),
        target_area_x=torch.zeros([1, 6]),
        target_area_y=torch.zeros([1, 6]),
        generation_variable_floats=generation_variable_floats,
        log_temperature_model=torch.zeros([1]),
        log_recommended_candidates_model=torch.zeros([1]),
        generation_variable_floats_model=generation_variable_floats,
        candidate_log_temperature_model=torch.zeros([1, 2]),
        candidate_log_recommended_candidates_model=torch.zeros([1, 2]),
        candidate_generation_variable_floats_model=torch.zeros(
            [1, 2, len(GENERATION_VARIABLE_FLOAT_FIELDS)]
        ),
        distance_proximity_scale=1.0,
        autocast=False,
    )


def area_predictions() -> Predictions:
    batch = 1
    candidate = 2
    door = 3
    connection = 4
    room_part = 5
    area = 6
    return Predictions(
        door_invalid=torch.zeros([batch, candidate, door]),
        connection_invalid=torch.zeros([batch, candidate, connection]),
        toilet_invalid=torch.zeros([batch, candidate]),
        phantoon_pair_invalid=torch.zeros([batch, candidate]),
        phantoon_area_invalid=torch.zeros([batch, candidate]),
        vanilla_area_invalid=torch.zeros([batch, candidate, 6]),
        balance_score=torch.zeros([batch, candidate, door]),
        area_balance_score=torch.zeros([batch, candidate, room_part]),
        toilet_balance_score=torch.zeros([batch, candidate]),
        avg_frontiers=torch.zeros([batch, candidate]),
        graph_diameter=torch.zeros([batch, candidate]),
        maridia_water=torch.zeros([batch, candidate, 3]),
        norfair_heat=torch.zeros([batch, candidate, 3]),
        maridia_water_count=torch.zeros([batch, candidate, 3]),
        norfair_heat_count=torch.zeros([batch, candidate, 3]),
        save_to_room_utility=torch.zeros([batch, candidate, room_part]),
        save_from_room_utility=torch.zeros([batch, candidate, room_part]),
        refill_to_room_utility=torch.zeros([batch, candidate, room_part]),
        refill_from_room_utility=torch.zeros([batch, candidate, room_part]),
        missing_connect_utility=torch.zeros([batch, candidate, connection]),
        area_crossings=torch.tensor([[2.0, 0.0]]),
        area_size=torch.zeros([batch, candidate, area, 3]),
        area_map_station_count=torch.zeros([batch, candidate, area, 3]),
        area_tiles=torch.zeros([batch, candidate, area]),
        area_x=torch.zeros([batch, candidate, area]),
        area_y=torch.zeros([batch, candidate, area]),
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
        vanilla_area_invalid=torch.full([1, 2, 6], -1.0),
        area_size_bucket=torch.full([1, 2, 6], -1.0),
        area_map_station_count_bucket=torch.full([1, 2, 6], -1.0),
        maridia_water=torch.full([1, 2, 3], -1.0),
        norfair_heat=torch.full([1, 2, 3], -1.0),
        door_match=torch.full([1, 2, 3], -1.0),
    )


def test_zero_area_rewards_leave_reward_unchanged() -> None:
    reward = compute_expected_reward(
        area_predictions(),
        unknown_outcomes(),
        zero_generate_config(),
    )
    assert torch.equal(reward, torch.zeros([1, 2]))


def test_phantoon_rewards_use_independent_coefficients() -> None:
    predictions = area_predictions()
    predictions.phantoon_pair_invalid = torch.tensor([[0.0, 1.0]])
    predictions.phantoon_area_invalid = torch.tensor([[2.0, 3.0]])
    outcomes = unknown_outcomes()

    pair_reward = compute_expected_reward(
        predictions,
        outcomes,
        zero_generate_config(reward_phantoon_pair=2.0),
    )
    area_reward = compute_expected_reward(
        predictions,
        outcomes,
        zero_generate_config(reward_phantoon_area=3.0),
    )

    assert torch.allclose(
        pair_reward,
        2.0 * torch.nn.functional.logsigmoid(-predictions.phantoon_pair_invalid),
    )
    assert torch.allclose(
        area_reward,
        3.0 * torch.nn.functional.logsigmoid(-predictions.phantoon_area_invalid),
    )


def test_vanilla_area_reward_is_masked_by_enabled_constraint() -> None:
    predictions = area_predictions()
    predictions.vanilla_area_invalid[:, :, 0] = torch.tensor([[0.0, 1.0]])
    config = zero_generate_config()
    config.reward_vanilla_area[:, 0] = 2.0

    assert torch.equal(
        compute_expected_reward(predictions, unknown_outcomes(), config),
        torch.zeros([1, 2]),
    )

    config.vanilla_area_constraint_mask[:, 0] = True
    assert torch.allclose(
        compute_expected_reward(predictions, unknown_outcomes(), config),
        2.0 * torch.nn.functional.logsigmoid(-predictions.vanilla_area_invalid[:, :, 0]),
    )


def test_area_rewards_use_valid_bucket_logprobs() -> None:
    reward = compute_expected_reward(
        area_predictions(),
        unknown_outcomes(),
        zero_generate_config(
            reward_area_crossing=5.0,
            reward_area_size_valid=7.0,
            reward_area_map_station=11.0,
        ),
    )
    middle_bucket_log_probability = torch.log_softmax(torch.zeros([3]), dim=0)[1]
    expected = (
        -5.0 * torch.tensor([[2.0, 0.0]])
        + 7.0 * 6 * middle_bucket_log_probability
        + 11.0 * 6 * middle_bucket_log_probability
    )
    assert torch.allclose(reward, expected, atol=1e-6)


def test_numeric_area_rewards_use_negative_mean_squared_error() -> None:
    predictions = area_predictions()
    predictions.area_tiles = torch.tensor([[[1.0] * 6, [2.0] * 6]])
    reward = compute_expected_reward(
        predictions,
        unknown_outcomes(),
        zero_generate_config(reward_area_tiles=2.0),
    )
    assert torch.equal(reward, torch.tensor([[-12.0, -48.0]]))


def test_heat_water_rewards_use_final_tier_coefficients() -> None:
    predictions = area_predictions()
    predictions.maridia_water_count = torch.tensor(
        [[[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]]]
    )
    predictions.norfair_heat_count = torch.tensor(
        [[[4.0, 5.0, 6.0], [2.0, 0.0, 1.0]]]
    )
    reward = compute_expected_reward(
        predictions,
        unknown_outcomes(),
        zero_generate_config(
            reward_maridia_water=torch.tensor([[0.1, 0.2, 0.3]]),
            reward_norfair_heat=torch.tensor([[0.4, 0.5, 0.6]]),
        ),
    )
    assert torch.allclose(reward, torch.tensor([[9.1, 1.6]]))


def test_area_balance_reward_penalizes_common_area_scores() -> None:
    predictions = area_predictions()
    predictions.area_balance_score = torch.tensor(
        [[[1.0, -0.5], [0.25, 0.25]]]
    )
    reward = compute_expected_reward(
        predictions,
        unknown_outcomes(),
        zero_generate_config(reward_area_balance=2.0),
    )
    assert torch.equal(reward, torch.tensor([[-1.0, -1.0]]))


def test_candidate_area_balance_uses_exact_placed_room_score() -> None:
    score_table = torch.arange(12, dtype=torch.float32).reshape(1, 2, 6)
    scores = apply_candidate_area_balance_scores(
        torch.tensor([[[100.0, 2.0], [3.0, 100.0]]]),
        torch.tensor([[[True, False], [False, True]]]),
        torch.zeros([1, 2], dtype=torch.bool),
        Actions(
            room_idx=torch.tensor([[0, 1]]),
            room_x=torch.zeros([1, 2], dtype=torch.int8),
            room_y=torch.zeros([1, 2], dtype=torch.int8),
            room_area=torch.tensor([[4, 2]]),
        ),
        score_table,
    )
    assert torch.equal(scores, torch.tensor([[[4.0, 2.0], [3.0, 8.0]]]))


def test_forced_and_preferred_rooms_are_exempt_from_area_balance() -> None:
    rooms = [
        {"special_type": "ship"},
        {"special_type": "phantoon_boss"},
        {"special_type": "phantoon_map"},
        {"special_type": "phantoon_save"},
        {},
        {"water": 1},
        {"water": 2},
        {"heat": 3},
    ]
    heat_water_reward = torch.zeros([2, 2, 3])
    heat_water_reward[1, 0, 1] = 0.5
    heat_water_reward[1, 1, 2] = 0.5
    exempt = area_balance_exempt_room_mask(
        rooms,
        torch.tensor(
            [
                [True, False, False, True, False, False],
                [False, False, False, False, False, False],
            ]
        ),
        heat_water_reward,
    )
    assert exempt.tolist() == [
        [True, True, True, True, False, False, False, False],
        [False, False, False, False, False, False, True, True],
    ]

    scores = apply_candidate_area_balance_scores(
        torch.ones([1, 1, len(rooms)]),
        torch.zeros([1, 1, len(rooms)], dtype=torch.bool),
        exempt[:1],
        Actions(
            room_idx=torch.tensor([[2]]),
            room_x=torch.zeros([1, 1], dtype=torch.int8),
            room_y=torch.zeros([1, 1], dtype=torch.int8),
            room_area=torch.tensor([[3]]),
        ),
        torch.ones([1, len(rooms), 6]),
    )
    assert scores.tolist() == [[[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]]]


def test_training_samples_use_top_down_heat_water_coefficients() -> None:
    torch.manual_seed(0)
    config = instantiate_scheduleable_config(
        Config.model_validate_json(Path("configs/zebes.json").read_text()), 0
    )
    assert variable_float_metric_value(
        config.generation.reward_maridia_water_3,
        "generation.reward_maridia_water_3",
    ) == 0.1875
    generate_config = create_generate_config(
        config,
        episode_length=4,
        num_envs=4096,
        device=torch.device("cpu"),
        ignore_scores=False,
        area_tile_scale=1.0,
        heat_water_tier_tile_counts=compute_heat_water_tier_tile_counts(
            json.loads(Path("room_definitions/zebes.json").read_text())
        ),
    )
    zero_masks = []
    for family, coefficients in (
        ("maridia_water", generate_config.reward_maridia_water),
        ("norfair_heat", generate_config.reward_norfair_heat),
    ):
        assert torch.all(coefficients[:, 1:] >= coefficients[:, :-1])
        assert torch.all(coefficients <= torch.tensor([0.25, 0.5, 0.75]))
        zero_mask = coefficients[:, 2] == 0.0
        zero_fraction = zero_mask.to(torch.float32).mean()
        assert 0.47 < zero_fraction < 0.53
        assert torch.all(coefficients[zero_mask] == 0.0)
        zero_masks.append(zero_mask)
        indices = [
            GENERATION_VARIABLE_FLOAT_FIELDS.index(f"reward_{family}_{tier}")
            for tier in range(1, 4)
        ]
        assert torch.equal(generate_config.generation_variable_floats[:, indices], coefficients)
    joint_zero_fraction = (zero_masks[0] & zero_masks[1]).to(torch.float32).mean()
    assert 0.23 < joint_zero_fraction < 0.27
    vanilla_reward_indices = [
        GENERATION_VARIABLE_FLOAT_FIELDS.index(name) for name in VANILLA_AREA_REWARD_FIELDS
    ]
    assert torch.equal(
        generate_config.generation_variable_floats[:, vanilla_reward_indices],
        generate_config.reward_vanilla_area
        * generate_config.vanilla_area_constraint_mask.to(torch.float32),
    )


def test_heat_water_rewards_floor_area_tile_targets_before_normalization() -> None:
    config = instantiate_scheduleable_config(
        Config.model_validate_json(Path("configs/zebes.json").read_text()), 0
    )
    config.generation.target_area_tiles = [100.0] * 6
    tier_tile_counts = {
        "maridia_water": (116, 184, 57),
        "norfair_heat": (34, 138, 74),
    }
    generate_config = create_generate_config(
        config,
        episode_length=4,
        num_envs=1024,
        device=torch.device("cpu"),
        ignore_scores=False,
        area_tile_scale=1.0,
        heat_water_tier_tile_counts=tier_tile_counts,
    )
    raw_targets = torch.full([1024, 6], 100.0)
    for family, area, rewards in (
        ("maridia_water", 4, generate_config.reward_maridia_water),
        ("norfair_heat", 2, generate_config.reward_norfair_heat),
    ):
        floor_scale = getattr(config.generation, f"{family}_floor_scale")
        floor = floor_scale * (rewards @ rewards.new_tensor(tier_tile_counts[family]))
        raw_targets[:, area] = torch.maximum(raw_targets[:, area], floor)
    expected = raw_targets * 6 / raw_targets.sum(dim=1, keepdim=True)
    torch.testing.assert_close(generate_config.target_area_tiles, expected)


def test_unforced_special_room_area_ss_excludes_forced_episodes() -> None:
    episode_count = 6
    room_count = 6
    actions = Actions(
        room_idx=torch.arange(room_count).repeat(episode_count, 1),
        room_x=torch.zeros([episode_count, room_count], dtype=torch.int8),
        room_y=torch.zeros([episode_count, room_count], dtype=torch.int8),
        room_area=torch.arange(episode_count, dtype=torch.int8)
        .unsqueeze(1)
        .repeat(1, room_count),
    )
    generation_variables = torch.zeros(
        [episode_count, len(GENERATION_VARIABLE_FLOAT_FIELDS)],
    )
    ship_force_idx = GENERATION_VARIABLE_FLOAT_FIELDS.index(VANILLA_AREA_CONDITION_FIELDS[0])
    generation_variables[1:, ship_force_idx] = 1.0
    rooms = [{"special_type": special_type} for special_type in VANILLA_AREA_SPECIAL_ROOM_TYPES]
    actual = compute_unforced_special_room_area_ss(actions, generation_variables, rooms)
    torch.testing.assert_close(actual, torch.tensor(11.0 / 36.0))


def main() -> None:
    test_zero_area_rewards_leave_reward_unchanged()
    test_phantoon_rewards_use_independent_coefficients()
    test_vanilla_area_reward_is_masked_by_enabled_constraint()
    test_area_rewards_use_valid_bucket_logprobs()
    test_numeric_area_rewards_use_negative_mean_squared_error()
    test_heat_water_rewards_use_final_tier_coefficients()
    test_area_balance_reward_penalizes_common_area_scores()
    test_candidate_area_balance_uses_exact_placed_room_score()
    test_forced_and_preferred_rooms_are_exempt_from_area_balance()
    test_training_samples_use_top_down_heat_water_coefficients()
    test_heat_water_rewards_floor_area_tile_targets_before_normalization()
    test_unforced_special_room_area_ss_excludes_forced_episodes()


if __name__ == "__main__":
    main()
