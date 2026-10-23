from typing import Any

import torch

from env import Engine
from model import BalanceModel, FrontierModel
from train_config import Config


MODEL_EXPORT_FORMAT = "map-gen-model-export-v21"
INFERENCE_BACKBONE_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def inference_backbone_dtype(name: str) -> torch.dtype:
    if name not in INFERENCE_BACKBONE_DTYPES:
        raise ValueError('backbone_dtype must be "float32" or "bfloat16"')
    return INFERENCE_BACKBONE_DTYPES[name]


def without_prefix(
    tensors: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    full_prefix = f"{prefix}."
    compiled_prefix = f"{full_prefix}_orig_mod."
    return {
        (
            name[len(compiled_prefix) :]
            if name.startswith(compiled_prefix)
            else name[len(full_prefix) :]
        ): value
        for name, value in tensors.items()
        if name.startswith(full_prefix)
    }


def frontier_model_kwargs(
    config: Config,
    rooms: list[dict],
    engine: Engine,
) -> dict[str, Any]:
    return {
        "num_rooms": len(rooms),
        "output_metadata": engine.get_output_metadata(),
        "map_x": config.map_size[0],
        "map_y": config.map_size[1],
        "embedding_width": config.model.embedding_width,
        "global_embedding_width": config.model.global_embedding_width,
        "hidden_width": config.model.hidden_width,
        "proposal_hidden_widths": config.model.proposal_hidden_widths,
        "missing_connect_query_hidden_width": config.model.missing_connect_query_hidden_width,
        "missing_connect_query_frontier_width": config.model.missing_connect_query_frontier_width,
        "missing_connect_query_distance_width": config.model.missing_connect_query_distance_width,
        "utility_query_hidden_width": config.model.utility_query_hidden_width,
        "utility_query_frontier_width": config.model.utility_query_frontier_width,
        "known_save_refill_utility_override": config.model.known_save_refill_utility_override,
        "distance_proximity_scale": config.distance_proximity_scale,
        "num_layers": config.model.num_layers,
        "door_counts": (
            count_room_doors_by_direction(rooms, "left"),
            count_room_doors_by_direction(rooms, "right"),
            count_room_doors_by_direction(rooms, "up"),
            count_room_doors_by_direction(rooms, "down"),
        ),
        "frontier_window_size": config.generation.frontier_window_size,
        "area_bounding_box_width": config.generation.area_bounding_box_width,
        "area_bounding_box_height": config.generation.area_bounding_box_height,
        "max_area_size": config.generation.max_area_size,
        "features": config.features,
    }


def count_room_doors_by_direction(rooms: list[dict], direction: str) -> int:
    return sum(
        1
        for room in rooms
        for door_group in room["doors"]
        for door in door_group
        if door["direction"] == direction
    )


def balance_model_kwargs(
    config: Config,
    rooms: list[dict],
    engine: Engine,
) -> dict[str, Any]:
    output_metadata = engine.get_output_metadata()
    has_toilet = any(room.get("special_type") == "toilet" for room in rooms)
    toilet_compatibility = torch.tensor(
        [
            has_toilet
            and bool(room["toilet_crossing_x"])
            and room.get("special_type") != "toilet"
            for room in rooms
        ],
        dtype=torch.bool,
    )
    if has_toilet and not torch.any(toilet_compatibility):
        raise ValueError("Toilet balancing requires at least one compatible crossing room")
    return dict(
        left_count=count_room_doors_by_direction(rooms, "left"),
        right_count=count_room_doors_by_direction(rooms, "right"),
        up_count=count_room_doors_by_direction(rooms, "up"),
        down_count=count_room_doors_by_direction(rooms, "down"),
        door_room_idx=torch.tensor(
            output_metadata.door,
            dtype=torch.int64,
        ),
        door_variant_compatibility=output_metadata.door_variant_compatibility,
        num_rooms=len(rooms),
        toilet_compatibility=toilet_compatibility,
        hidden_width=config.balance_model.hidden_width,
        num_layers=config.balance_model.num_layers,
    )


def create_balance_model(
    config: Config,
    rooms: list[dict],
    engine: Engine,
    device: torch.device,
) -> BalanceModel:
    return BalanceModel(**balance_model_kwargs(config, rooms, engine)).to(device)


def load_inference_model_state(
    model: FrontierModel | BalanceModel,
    tensors: dict[str, torch.Tensor],
    backbone_dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Validate the saved precision policy and preserve dtypes when loading weights."""
    model.configure_inference_dtype(backbone_dtype)
    for name, expected in model.state_dict().items():
        if name in tensors and tensors[name].dtype != expected.dtype:
            raise ValueError(
                f"inference tensor {name!r} has dtype {tensors[name].dtype}, "
                f"expected {expected.dtype}"
            )
    # Copying into the model's initial parameters would silently change saved dtypes.
    model.load_state_dict(tensors, assign=True)
    model.to(device)
    model.requires_grad_(False)
    model.eval()


def create_inference_models(
    config: Config,
    rooms: list[dict],
    engine: Engine,
    tensors: dict[str, torch.Tensor],
    backbone_dtype: torch.dtype,
    device: torch.device,
) -> tuple[FrontierModel, BalanceModel]:
    model = FrontierModel(**frontier_model_kwargs(config, rooms, engine))
    balance_model = create_balance_model(config, rooms, engine, torch.device("cpu"))
    for module, prefix in ((model, "ema_model"), (balance_model, "balance_model")):
        load_inference_model_state(module, without_prefix(tensors, prefix), backbone_dtype, device)
    return model, balance_model
