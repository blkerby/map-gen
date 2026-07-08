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


def main() -> None:
    test_generation_area_bounding_box_fields_are_required()
    test_generation_area_bounding_box_fields_must_be_positive()


if __name__ == "__main__":
    main()
