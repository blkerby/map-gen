import json
from pathlib import Path

from pydantic import ValidationError

from train_config import (
    Config,
    VariableMixture,
    VariableRange,
    VariableSchedule,
    instantiate_scheduleable_config,
    validate_config,
)


def load_debug_config() -> dict:
    return json.loads(Path("configs/debug.json").read_text())


def test_generation_area_bounding_box_fields_are_required() -> None:
    config_data = load_debug_config()
    del config_data["generation"]["area_bounding_box_width"]

    try:
        Config.model_validate(config_data)
    except ValidationError:
        pass
    else:
        raise AssertionError("generation.area_bounding_box_width should be required")


def test_vanilla_area_probability_is_required_and_bounded() -> None:
    config_data = load_debug_config()
    del config_data["generation"]["force_ship_in_crateria_probability"]
    try:
        Config.model_validate(config_data)
    except ValidationError:
        pass
    else:
        raise AssertionError("force_ship_in_crateria_probability should be required")

    config_data = load_debug_config()
    config_data["generation"]["force_ship_in_crateria_probability"] = 1.1
    try:
        validate_config(Config.model_validate(config_data))
    except ValueError as err:
        assert "force_ship_in_crateria_probability" in str(err)
    else:
        raise AssertionError("force_ship_in_crateria_probability should reject values above one")

    config_data = load_debug_config()
    config_data["generation"]["force_ship_in_crateria_probability"] = {"linear": [0.0, 0.5]}
    config = Config.model_validate(config_data)
    validate_config(config)
    instantiated = instantiate_scheduleable_config(config, 320)
    assert instantiated.generation.force_ship_in_crateria_probability == 0.25

    config_data["generation"]["force_ship_in_crateria_probability"] = {"linear": [0.0, 0.5, 1.0]}
    try:
        validate_config(Config.model_validate(config_data))
    except ValueError as err:
        assert "knot_episodes" in str(err)
    else:
        raise AssertionError("force probability schedule should match knot_episodes")

    config_data["generation"]["force_ship_in_crateria_probability"] = {"linear": [0.0, 1.1]}
    try:
        validate_config(Config.model_validate(config_data))
    except ValueError as err:
        assert "force_ship_in_crateria_probability[1]" in str(err)
    else:
        raise AssertionError("force probability schedule should reject values above one")

    config_data["generation"]["force_ship_in_crateria_probability"] = {"log": [0.0, 0.5]}
    try:
        validate_config(Config.model_validate(config_data))
    except ValueError as err:
        assert "greater than zero for a log schedule" in str(err)
    else:
        raise AssertionError("log force probability schedule should reject zero")


def test_recommended_candidates_same_frontier_is_required() -> None:
    config_data = load_debug_config()
    del config_data["generation"]["recommended_candidates_same_frontier"]

    try:
        Config.model_validate(config_data)
    except ValidationError:
        pass
    else:
        raise AssertionError("generation.recommended_candidates_same_frontier should be required")


def test_proposal_target_temperature_is_required() -> None:
    config_data = load_debug_config()
    del config_data["train"]["proposal_target_temperature"]

    try:
        Config.model_validate(config_data)
    except ValidationError:
        pass
    else:
        raise AssertionError("train.proposal_target_temperature should be required")


def test_balance_train_is_required_and_batch_size_divides_round() -> None:
    config_data = load_debug_config()
    del config_data["balance_train"]
    try:
        Config.model_validate(config_data)
    except ValidationError:
        pass
    else:
        raise AssertionError("balance_train should be required")

    config_data = load_debug_config()
    config_data["balance_train"]["batch_size"] = 3
    try:
        validate_config(Config.model_validate(config_data))
    except ValueError as err:
        assert "balance_train.batch_size" in str(err)
    else:
        raise AssertionError("balance_train.batch_size should evenly divide a round")

    config_data = load_debug_config()
    del config_data["balance_train"]["door_eta"]
    try:
        Config.model_validate(config_data)
    except ValidationError:
        pass
    else:
        raise AssertionError("balance_train.door_eta should be required")

    config_data = load_debug_config()
    config_data["balance_train"]["door_eta"] = 0.0
    validate_config(Config.model_validate(config_data))

    config_data["balance_train"]["door_eta"] = -0.01
    try:
        validate_config(Config.model_validate(config_data))
    except ValueError as err:
        assert "balance_train.door_eta" in str(err)
    else:
        raise AssertionError("balance_train.door_eta should reject negative values")

    config_data = load_debug_config()
    del config_data["balance_train"]["door_beta"]
    try:
        Config.model_validate(config_data)
    except ValidationError:
        pass
    else:
        raise AssertionError("balance_train.door_beta should be required")

    config_data = load_debug_config()
    config_data["balance_train"]["door_beta"] = 0.0
    try:
        validate_config(Config.model_validate(config_data))
    except ValueError as err:
        assert "balance_train.door_beta" in str(err)
    else:
        raise AssertionError("balance_train.door_beta should reject zero")


def test_proposal_target_temperature_must_be_positive() -> None:
    config_data = load_debug_config()
    config_data["train"]["proposal_target_temperature"] = 0.0
    config = Config.model_validate(config_data)

    try:
        validate_config(config)
    except ValueError as err:
        assert "train.proposal_target_temperature" in str(err)
    else:
        raise AssertionError("train.proposal_target_temperature should reject zero")


def test_scheduled_ints_instantiate() -> None:
    config_data = load_debug_config()
    config_data["balance_train"]["batch_size"] = {"step": [1, 3]}
    config_data["train"]["batch_size"] = {"step": [1, 3]}
    config_data["train"]["gradient_accumulation_steps"] = {"linear": [1, 3]}
    config = Config.model_validate(config_data)
    validate_config(config)

    midpoint = instantiate_scheduleable_config(config, 320)
    second_knot = instantiate_scheduleable_config(config, 640)

    assert midpoint.balance_train.batch_size == 1
    assert midpoint.train.batch_size == 1
    assert midpoint.train.gradient_accumulation_steps == 2
    assert second_knot.balance_train.batch_size == 3
    assert second_knot.train.batch_size == 3


def test_generation_area_bounding_box_fields_must_be_positive() -> None:
    config_data = load_debug_config()
    config_data["generation"]["area_bounding_box_height"] = 0
    config = Config.model_validate(config_data)

    try:
        validate_config(config)
    except ValueError as err:
        assert "generation.area_bounding_box_height" in str(err)
    else:
        raise AssertionError("generation.area_bounding_box_height should reject zero")


def test_max_candidate_areas_per_placement_must_be_in_range() -> None:
    config_data = load_debug_config()
    config_data["generation"]["max_candidate_areas_per_placement"] = 0
    config = Config.model_validate(config_data)

    try:
        validate_config(config)
    except ValueError as err:
        assert "generation.max_candidate_areas_per_placement" in str(err)
    else:
        raise AssertionError("max_candidate_areas_per_placement should reject zero")

    config_data = load_debug_config()
    config_data["generation"]["max_candidate_areas_per_placement"] = 7
    config = Config.model_validate(config_data)

    try:
        validate_config(config)
    except ValueError as err:
        assert "generation.max_candidate_areas_per_placement" in str(err)
    else:
        raise AssertionError("max_candidate_areas_per_placement should reject seven")


def test_num_scored_invalid_candidates_must_fit_shortlist() -> None:
    config_data = load_debug_config()
    config_data["generation"]["num_scored_invalid_candidates"] = -1
    config = Config.model_validate(config_data)

    try:
        validate_config(config)
    except ValueError as err:
        assert "generation.num_scored_invalid_candidates" in str(err)
    else:
        raise AssertionError("num_scored_invalid_candidates should reject negatives")

    config_data = load_debug_config()
    config_data["generation"]["num_scored_invalid_candidates"] = 17
    config = Config.model_validate(config_data)

    try:
        validate_config(config)
    except ValueError as err:
        assert "generation.num_scored_invalid_candidates" in str(err)
    else:
        raise AssertionError("num_scored_invalid_candidates should fit the shortlist")


def test_area_targets_require_six_finite_values_and_instantiate_schedules() -> None:
    config_data = load_debug_config()
    config_data["generation"]["target_area_x"] = [0.0] * 5
    try:
        Config.model_validate(config_data)
    except ValidationError:
        pass
    else:
        raise AssertionError("target_area_x should require six values")

    config_data = load_debug_config()
    config_data["generation"]["target_area_x"][0] = float("inf")
    config = Config.model_validate(config_data)
    try:
        validate_config(config)
    except ValueError as err:
        assert "generation.target_area_x[0]" in str(err)
    else:
        raise AssertionError("target_area_x should reject non-finite values")

    config_data = load_debug_config()
    config_data["generation"]["target_area_x"][0] = {
        "step": {"min": [0.0, 10.0], "max": [20.0, 30.0]}
    }
    config_data["generation"]["target_area_x"][1] = {"step": [1.0, 3.0]}
    config = Config.model_validate(config_data)
    midpoint = instantiate_scheduleable_config(config, 320)
    second_knot = instantiate_scheduleable_config(config, 640)
    midpoint_range = midpoint.generation.target_area_x[0]
    second_knot_range = second_knot.generation.target_area_x[0]
    assert isinstance(midpoint_range, VariableSchedule)
    assert isinstance(midpoint_range.step, VariableRange)
    assert midpoint_range.step.min == 0.0
    assert midpoint_range.step.max == 20.0
    assert midpoint.generation.target_area_x[1] == 1.0
    assert isinstance(second_knot_range, VariableSchedule)
    assert isinstance(second_knot_range.step, VariableRange)
    assert second_knot_range.step.min == 10.0
    assert second_knot_range.step.max == 30.0
    assert second_knot.generation.target_area_x[1] == 3.0

    config_data = load_debug_config()
    config_data["generation"]["target_area_rooms"][0] = 0.0
    config = Config.model_validate(config_data)
    try:
        validate_config(config)
    except ValueError as err:
        assert "generation.target_area_rooms[0]" in str(err)
    else:
        raise AssertionError("target_area_rooms should reject nonpositive values")


def test_variable_float_mixture_instantiates_weights_and_rejects_invalid_values() -> None:
    config_data = load_debug_config()
    config_data["generation"]["reward_area_x"] = {
        "mixture": [
            {"weight": {"linear": [0.25, 0.75]}, "value": 0.0},
            {
                "weight": 1.0,
                "value": {"linear": {"min": [1.0, 2.0], "max": [3.0, 4.0]}},
            },
        ]
    }
    config = Config.model_validate(config_data)
    validate_config(config)
    instantiated = instantiate_scheduleable_config(config, 320)
    mixture = instantiated.generation.reward_area_x
    assert isinstance(mixture, VariableMixture)
    assert mixture.mixture[0].weight == 0.5
    value = mixture.mixture[1].value
    assert isinstance(value, VariableSchedule)
    assert isinstance(value.linear, VariableRange)
    assert value.linear.min == 1.5
    assert value.linear.max == 3.5

    config_data["generation"]["reward_area_x"]["mixture"][0]["weight"] = -1.0
    try:
        validate_config(Config.model_validate(config_data))
    except ValueError as err:
        assert "reward_area_x.mixture[0].weight" in str(err)
    else:
        raise AssertionError("mixture weights should reject negative values")


def main() -> None:
    test_generation_area_bounding_box_fields_are_required()
    test_vanilla_area_probability_is_required_and_bounded()
    test_recommended_candidates_same_frontier_is_required()
    test_proposal_target_temperature_is_required()
    test_balance_train_is_required_and_batch_size_divides_round()
    test_proposal_target_temperature_must_be_positive()
    test_scheduled_ints_instantiate()
    test_generation_area_bounding_box_fields_must_be_positive()
    test_max_candidate_areas_per_placement_must_be_in_range()
    test_num_scored_invalid_candidates_must_fit_shortlist()
    test_area_targets_require_six_finite_values_and_instantiate_schedules()
    test_variable_float_mixture_instantiates_weights_and_rejects_invalid_values()


if __name__ == "__main__":
    main()
