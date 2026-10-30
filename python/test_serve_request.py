from types import SimpleNamespace

from pydantic import ValidationError
import torch

import serve
from serve import (
    GenerateRequest,
    ServingConfig,
    create_generate_configs,
    run_warmup_requests,
    validate_generate_request,
    validate_serving_config,
    warmup_generate_request,
)
from train_config import GENERATION_VARIABLE_FLOAT_FIELDS, VANILLA_AREA_REWARD_FIELDS


def base_payload() -> dict:
    return {
        "max_num_maps": 1,
        "episode_length": 3,
        "recommended_candidates": 1,
        "shortlist_candidates": 1,
        "num_scored_invalid_candidates": 1,
        "max_candidate_areas_per_placement": 2,
        "recommended_candidates_same_frontier": False,
        "guarantee_phantoon_counterpart": False,
        "temperature": 1.0,
        "proposal_temperature": 1.0,
        "reward_door": 1.0,
        "reward_connection": 1.0,
        "reward_toilet": 1.0,
        "reward_phantoon_pair": 1.0,
        "reward_phantoon_area": 1.0,
        "reward_success": 0.0,
        "reward_initial_item": 0.0,
        "initial_item_start": None,
        "initial_item_difficulty": "Basic",
        "initial_item_wall_jump": True,
        "reward_ship_in_crateria": 1.0,
        "reward_kraid_in_brinstar": 1.0,
        "reward_ridley_in_norfair": 1.0,
        "reward_phantoon_in_wrecked_ship": 1.0,
        "reward_draygon_in_maridia": 1.0,
        "reward_mother_brain_in_tourian": 1.0,
        "force_ship_in_crateria": False,
        "force_kraid_in_brinstar": False,
        "force_ridley_in_norfair": False,
        "force_phantoon_in_wrecked_ship": False,
        "force_draygon_in_maridia": False,
        "force_mother_brain_in_tourian": False,
        "reward_frontier": 1.0,
        "reward_graph_diameter": 1.0,
        "maridia_water_preferred_probability_1": 0.25,
        "maridia_water_preferred_probability_2": 0.5,
        "maridia_water_preferred_probability_3": 0.75,
        "norfair_heat_preferred_probability_1": 0.25,
        "norfair_heat_preferred_probability_2": 0.5,
        "norfair_heat_preferred_probability_3": 0.75,
        "reward_save_distance": 1.0,
        "reward_refill_distance": 1.0,
        "reward_missing_connect_utility": 1.0,
        "door_balance_strength": 1.0,
        "max_return_distance": None,
        "reward_area_distinct_crossing": 0.0,
        "reward_area_crossing": 1.0,
        "reward_area_size_valid": 1.0,
        "reward_area_map_station": 1.0,
        "reward_area_x": 1.0,
        "reward_area_y": 1.0,
        "target_area_rooms": [1.0] * 6,
        "target_area_x": [1.0] * 6,
        "target_area_y": [1.0] * 6,
    }


def test_valid_map_mask_includes_area_outcomes() -> None:
    batch_size = 6
    phantoon_pair_invalid = torch.zeros([batch_size])
    phantoon_pair_invalid[4] = 1
    phantoon_area_invalid = torch.zeros([batch_size])
    phantoon_area_invalid[5] = 1
    step_outcomes = SimpleNamespace(
        door_invalid=torch.zeros([batch_size, 1]),
        connection_invalid=torch.zeros([batch_size, 1]),
        return_distance_invalid=torch.zeros([batch_size, 1]),
        toilet_invalid=torch.zeros([batch_size]),
        phantoon_pair_invalid=phantoon_pair_invalid,
        phantoon_area_invalid=phantoon_area_invalid,
        vanilla_area_invalid=torch.zeros([batch_size, 6], dtype=torch.bool),
    )
    area_size = torch.full([batch_size, 6], 2)
    area_size[1, 0] = 0
    area_size[2, 0] = 4
    area_map_station_count = torch.ones([batch_size, 6], dtype=torch.int64)
    area_map_station_count[3, 0] = 2
    outcomes = SimpleNamespace(
        step_outcomes=step_outcomes,
        end_outcomes=SimpleNamespace(
            area_size=area_size,
            area_map_station_count=area_map_station_count,
        ),
    )

    assert serve.valid_map_mask(outcomes, torch.zeros([6], dtype=torch.bool), 1, 3).tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
    ]

    step_outcomes.vanilla_area_invalid[0, 0] = True
    assert not serve.valid_map_mask(
        outcomes,
        torch.tensor([True, False, False, False, False, False]),
        1,
        3,
    )[0]


def test_create_generate_configs_normalizes_area_targets() -> None:
    generate_request = GenerateRequest.model_validate(
        base_payload()
        | {
            "small_map": False,
            "target_area_rooms": [1.0, 2.0, 3.0, 4.0, 5.0, 9.0],
            "temperature": 0.03,
            "proposal_temperature": 100.0,
            "target_area_x": [5.0] * 6,
            "target_area_y": [10.0] * 6,
            "guarantee_phantoon_counterpart": True,
        }
    )
    state = SimpleNamespace(
        rooms=[{"map": [[1] * 6]} for _ in range(3)],
        serving_config=SimpleNamespace(gpu_prefetch_batches=0, autocast=False),
        training_config=SimpleNamespace(
            map_size=(10, 20),
            distance_proximity_scale=1.0,
            model=SimpleNamespace(generation_autocast=False),
        ),
    )
    config = create_generate_configs(
        generate_request,
        state,
        [SimpleNamespace(num_envs=2)],
        torch.device("cpu"),
    )[0]
    expected_area_rooms = (
        torch.tensor([1, 2, 3, 4, 5, 9], dtype=torch.float32).div(8).expand(2, 6)
    )
    assert torch.equal(config.target_area_rooms, expected_area_rooms)
    assert config.guarantee_phantoon_counterpart
    assert torch.equal(config.target_area_x, torch.full([2, 6], 0.5))
    assert torch.equal(config.target_area_y, torch.full([2, 6], 0.5))
    target_rooms_index = GENERATION_VARIABLE_FLOAT_FIELDS.index("target_area_probability_0")
    assert torch.equal(
        config.generation_variable_floats[:, target_rooms_index : target_rooms_index + 6],
        expected_area_rooms / 3,
    )
    for field, value in (("log_temperature", 0.03), ("log_proposal_temperature", 100.0)):
        torch.testing.assert_close(
            config.generation_variable_floats[:, GENERATION_VARIABLE_FLOAT_FIELDS.index(field)],
            torch.full((2,), value).log(),
        )
    target_x_index = GENERATION_VARIABLE_FLOAT_FIELDS.index("target_area_x_0")
    assert torch.equal(
        config.generation_variable_floats[:, target_x_index],
        torch.full([2], 0.5),
    )
    vanilla_reward_indices = [
        GENERATION_VARIABLE_FLOAT_FIELDS.index(name) for name in VANILLA_AREA_REWARD_FIELDS
    ]
    assert torch.equal(
        config.generation_variable_floats[:, vanilla_reward_indices],
        torch.zeros([2, len(VANILLA_AREA_REWARD_FIELDS)]),
    )
    assert torch.equal(config.reward_vanilla_area, torch.ones([2, 6]))


def base_serving_config_payload() -> dict:
    return {
        "host": "127.0.0.1",
        "port": 5000,
        "device": "cpu",
        "compile_model": False,
        "gpu_memory_fraction": 1.0,
        "autocast": False,
        "verify_outcome_consistency": False,
        "gpu_prefetch_batches": 1,
        "num_warmup_requests": 0,
        "prefetch_capacity": 4,
        "prefetch_initial_target": 2,
        "prefetch_low_capacity_fraction": 0.5,
        "prefetch_refill_batches": 2,
        "foreground_max_batches": 1,
        "prefetch_max_queues": 2,
        "room_set": "room_definitions/zebes.json",
        "num_environments": 1,
        "pipeline_groups": 1,
        "num_threads": 1,
    }


def assert_invalid_value(payload: dict, expected_message: str) -> None:
    request = GenerateRequest.model_validate(payload)
    try:
        validate_generate_request(request, rooms=[{}, {}, {}])
    except ValueError as error:
        assert str(error) == expected_message
    else:
        raise AssertionError("expected ValueError")


def assert_invalid_serving_config(payload: dict, expected_message: str) -> None:
    serving_config = ServingConfig.model_validate(payload)
    try:
        validate_serving_config(serving_config)
    except ValueError as error:
        assert str(error) == expected_message
    else:
        raise AssertionError("expected ValueError")


def assert_warmup_requests_run() -> None:
    original_generate_response_data = serve.generate_response_data
    original_generate_response_data_uncached_validated = (
        serve.generate_response_data_uncached_validated
    )
    calls = []
    state = SimpleNamespace(serving_config=SimpleNamespace(num_warmup_requests=3))

    def generate_response_data(_state, generate_request):
        raise AssertionError("warmup should not use prefetch-aware generation")

    def generate_response_data_uncached_validated(_state, generate_request):
        calls.append(generate_request)
        return {"stats": {"num_valid": 0}}

    try:
        serve.generate_response_data = generate_response_data
        serve.generate_response_data_uncached_validated = generate_response_data_uncached_validated
        run_warmup_requests(state)
    finally:
        serve.generate_response_data = original_generate_response_data
        serve.generate_response_data_uncached_validated = (
            original_generate_response_data_uncached_validated
        )

    assert len(calls) == 3
    assert all(isinstance(call, GenerateRequest) for call in calls)


def assert_warmup_request_failure_propagates() -> None:
    original_generate_response_data_uncached_validated = (
        serve.generate_response_data_uncached_validated
    )
    state = SimpleNamespace(serving_config=SimpleNamespace(num_warmup_requests=1))

    def generate_response_data_uncached_validated(_state, _generate_request):
        raise RuntimeError("warmup failed")

    try:
        serve.generate_response_data_uncached_validated = generate_response_data_uncached_validated
        try:
            run_warmup_requests(state)
        except RuntimeError as error:
            assert str(error) == "warmup failed"
        else:
            raise AssertionError("expected RuntimeError")
    finally:
        serve.generate_response_data_uncached_validated = (
            original_generate_response_data_uncached_validated
        )


def main() -> None:
    test_valid_map_mask_includes_area_outcomes()
    test_create_generate_configs_normalizes_area_targets()
    full_map_payload = base_payload() | {"small_map": False}
    full_map_request = GenerateRequest.model_validate(full_map_payload)
    validate_generate_request(full_map_request, rooms=[{}, {}, {}])
    for coefficient in (-1.0, 0.0, 1.0):
        request = GenerateRequest.model_validate(
            full_map_payload | {"reward_area_crossing": coefficient}
        )
        validate_generate_request(request, rooms=[{}, {}, {}])
    for coefficient in (float("nan"), float("inf"), -float("inf")):
        assert_invalid_value(
            full_map_payload | {"reward_area_crossing": coefficient},
            "reward_area_crossing must be finite",
        )

    missing_same_frontier_payload = dict(full_map_payload)
    del missing_same_frontier_payload["recommended_candidates_same_frontier"]
    try:
        GenerateRequest.model_validate(missing_same_frontier_payload)
    except ValidationError:
        pass
    else:
        raise AssertionError("recommended_candidates_same_frontier should be required")

    missing_counterpart_payload = dict(full_map_payload)
    del missing_counterpart_payload["guarantee_phantoon_counterpart"]
    try:
        GenerateRequest.model_validate(missing_counterpart_payload)
    except ValidationError:
        pass
    else:
        raise AssertionError("guarantee_phantoon_counterpart should be required")

    assert_invalid_value(
        base_payload() | {"small_map": False, "max_candidate_areas_per_placement": 0},
        "max_candidate_areas_per_placement must be greater than zero",
    )
    assert_invalid_value(
        base_payload() | {"small_map": False, "max_candidate_areas_per_placement": 7},
        "max_candidate_areas_per_placement must be at most AREA_COUNT",
    )

    try:
        GenerateRequest.model_validate(
            base_payload() | {"small_map": False, "target_area_x": [0.0] * 5}
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("target_area_x should require six values")
    assert_invalid_value(
        base_payload() | {"small_map": False, "target_area_x": [float("inf")] * 6},
        "target_area_x[0] must be finite",
    )
    assert_invalid_value(
        base_payload() | {"small_map": False, "target_area_rooms": [-1.0] + [1.0] * 5},
        "target_area_rooms values must be greater than zero",
    )
    assert_invalid_value(
        base_payload() | {"small_map": False, "target_area_rooms": [0.0] * 6},
        "target_area_rooms values must be greater than zero",
    )
    assert_invalid_value(
        base_payload()
        | {
            "small_map": False,
            "maridia_water_preferred_probability_1": 0.6,
            "maridia_water_preferred_probability_2": 0.5,
        },
        "maridia_water_preferred_probability values must be positive, below one, and nondecreasing",
    )

    try:
        GenerateRequest.model_validate(base_payload())
    except ValidationError:
        pass
    else:
        raise AssertionError("small_map should be required")

    assert_invalid_value(
        base_payload() | {"small_map": True},
        "small_map requires min_rooms, max_rooms, target_rooms",
    )
    assert_invalid_value(
        base_payload()
        | {
            "small_map": True,
            "min_rooms": 0,
            "max_rooms": 2,
            "target_rooms": 2,
        },
        "min_rooms must be greater than zero",
    )
    assert_invalid_value(
        base_payload()
        | {
            "small_map": True,
            "min_rooms": 3,
            "max_rooms": 2,
            "target_rooms": 2,
        },
        "max_rooms must be at least min_rooms",
    )
    assert_invalid_value(
        base_payload()
        | {
            "small_map": True,
            "min_rooms": 1,
            "max_rooms": 2,
            "target_rooms": 0,
        },
        "target_rooms must be greater than zero",
    )

    small_map_request = GenerateRequest.model_validate(
        base_payload()
        | {
            "small_map": True,
            "min_rooms": 1,
            "max_rooms": 2,
            "target_rooms": 2,
        }
    )
    validate_generate_request(small_map_request, rooms=[{}, {}, {}])

    try:
        ServingConfig.model_validate(
            {
                key: value
                for key, value in base_serving_config_payload().items()
                if key != "num_warmup_requests"
            }
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("num_warmup_requests should be required")
    for field in (
        "prefetch_capacity",
        "prefetch_initial_target",
        "prefetch_low_capacity_fraction",
        "prefetch_refill_batches",
        "foreground_max_batches",
        "prefetch_max_queues",
    ):
        try:
            ServingConfig.model_validate(
                {
                    key: value
                    for key, value in base_serving_config_payload().items()
                    if key != field
                }
            )
        except ValidationError:
            pass
        else:
            raise AssertionError(f"{field} should be required")

    zero_warmup_config = ServingConfig.model_validate(base_serving_config_payload())
    validate_serving_config(zero_warmup_config)
    assert_invalid_serving_config(
        base_serving_config_payload() | {"num_warmup_requests": -1},
        "num_warmup_requests must be greater than or equal to zero",
    )
    assert_invalid_serving_config(
        base_serving_config_payload() | {"prefetch_capacity": -1},
        "prefetch_capacity must be greater than or equal to zero",
    )
    assert_invalid_serving_config(
        base_serving_config_payload() | {"prefetch_max_queues": -1},
        "prefetch_max_queues must be greater than or equal to zero",
    )
    assert_invalid_serving_config(
        base_serving_config_payload() | {"prefetch_low_capacity_fraction": -1.0},
        "prefetch_low_capacity_fraction must be between zero and one",
    )

    warmup_request = warmup_generate_request()
    assert warmup_request.episode_length == 253
    assert warmup_request.recommended_candidates == 8
    assert warmup_request.shortlist_candidates == 1024
    assert warmup_request.num_scored_invalid_candidates == 0
    assert warmup_request.max_candidate_areas_per_placement == 2
    assert not warmup_request.recommended_candidates_same_frontier
    assert warmup_request.guarantee_phantoon_counterpart
    assert warmup_request.temperature == 0.04
    assert warmup_request.proposal_temperature == 0.03
    assert warmup_request.reward_missing_connect_utility == 1.0
    assert warmup_request.target_area_rooms == [150, 300, 300, 100, 200, 100]
    assert not warmup_request.small_map
    validate_generate_request(warmup_request, rooms=[{} for _ in range(253)])
    assert_warmup_requests_run()
    assert_warmup_request_failure_propagates()


if __name__ == "__main__":
    main()
