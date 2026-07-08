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


def test_area_connected_component_bucket_bounds_must_start_with_zero_one() -> None:
    config_data = load_debug_config()
    config_data["train"]["area_connected_component_bucket_upper_bounds"] = [0, 2, 3]
    config = Config.model_validate(config_data)

    try:
        validate_config(config)
    except ValueError as err:
        assert "area_connected_component_bucket_upper_bounds" in str(err)
    else:
        raise AssertionError("area connected component bucket bounds should reject [0, 2, 3]")


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


def main() -> None:
    test_generation_area_bounding_box_fields_are_required()
    test_generation_area_bounding_box_fields_must_be_positive()
    test_area_connected_component_bucket_bounds_must_start_with_zero_one()
    test_max_candidate_areas_per_placement_must_be_in_range()


if __name__ == "__main__":
    main()
