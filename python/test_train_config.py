import json
from pathlib import Path

from pydantic import ValidationError

from train_config import Config, validate_config


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


def test_proposal_target_temperature_is_required() -> None:
    config_data = load_debug_config()
    del config_data["train"]["proposal_target_temperature"]

    try:
        Config.model_validate(config_data)
    except ValidationError:
        pass
    else:
        raise AssertionError("train.proposal_target_temperature should be required")


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


def main() -> None:
    test_generation_area_bounding_box_fields_are_required()
    test_proposal_target_temperature_is_required()
    test_proposal_target_temperature_must_be_positive()
    test_generation_area_bounding_box_fields_must_be_positive()
    test_max_candidate_areas_per_placement_must_be_in_range()
    test_num_scored_invalid_candidates_must_fit_shortlist()


if __name__ == "__main__":
    main()
