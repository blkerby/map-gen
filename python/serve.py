from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, NamedTuple

import torch
from flask import Flask, jsonify, request
from hypercorn.asyncio import serve
from hypercorn.config import Config as HypercornConfig
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from safetensors import safe_open
from werkzeug.exceptions import BadRequest

from env import (
    AREA_COUNT,
    DoorMatches,
    Engine,
    EpisodeData,
    EpisodeOutcomes,
    GenerateConfig,
    compute_area_balance_targets,
)
from generate import (
    GenerationCancelled,
    GenerationProfiler,
    profile_start,
    run_generation_groups,
    sync_profile_device,
)
from features import GenerationVariableFloatsFeature
from initial_item import NODES, GRAPHS
from small_map import (
    DoorData,
    RequiredRoomData,
    RoomPartData,
    SmallMapConfig,
    build_door_data,
    build_required_room_data,
    build_room_part_data,
    prune_small_maps,
)
from device_util import gpu_backend, is_gpu
from model_loading import MODEL_EXPORT_FORMAT, create_inference_models, inference_backbone_dtype
from subarea import (
    DoorRoomLookup,
    build_door_room_lookup,
    build_room_adjacency,
    split_assignment_by_balanced_centers,
)
from train_config import (
    Config,
    GENERATION_VARIABLE_FLOAT_FIELDS,
    VANILLA_AREA_CONDITION_FIELDS,
    VANILLA_AREA_OUTCOME_NAMES,
    VANILLA_AREA_REWARD_FIELDS,
    validate_config,
)


TRAINING_CHECKPOINT_FORMAT = "map-gen-training-session-checkpoint-v28"
MODEL_INPUT_FORMATS = (MODEL_EXPORT_FORMAT, TRAINING_CHECKPOINT_FORMAT)
MODEL_PREFIXES = ("ema_model", "balance_model")

DIRECTIONS = ("left", "right", "up", "down")
OPPOSITE_DIRECTIONS = {
    "left": "right",
    "right": "left",
    "up": "down",
    "down": "up",
}

app = Flask(__name__)
app.json.sort_keys = False  # Disables alphabetical key sorting


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ServingConfig(StrictBaseModel):
    host: str
    port: int
    device: str
    compile_model: bool
    gpu_memory_fraction: float
    autocast: bool
    verify_outcome_consistency: bool
    gpu_prefetch_batches: int
    num_warmup_requests: int
    prefetch_capacity: int
    prefetch_initial_target: int
    prefetch_low_capacity_fraction: float
    prefetch_refill_batches: int
    prefetch_max_queues: int
    foreground_max_batches: int
    room_set: Path
    num_environments: int
    pipeline_groups: int
    num_threads: int


class GenerateRequest(StrictBaseModel):
    max_num_maps: Annotated[int, Field(strict=True, gt=0)]
    episode_length: int
    recommended_candidates: int
    shortlist_candidates: int
    num_scored_invalid_candidates: int
    max_candidate_areas_per_placement: int
    recommended_candidates_same_frontier: bool
    guarantee_phantoon_counterpart: bool
    temperature: float
    proposal_temperature: float
    reward_door: float
    reward_connection: float
    reward_toilet: float
    reward_phantoon_pair: float
    reward_phantoon_area: float
    reward_success: float
    reward_initial_item: float
    initial_item_start: tuple[int, int] | None
    initial_item_difficulty: str
    initial_item_wall_jump: bool
    reward_ship_in_crateria: float
    reward_kraid_in_brinstar: float
    reward_ridley_in_norfair: float
    reward_phantoon_in_wrecked_ship: float
    reward_draygon_in_maridia: float
    reward_mother_brain_in_tourian: float
    force_ship_in_crateria: bool
    force_kraid_in_brinstar: bool
    force_ridley_in_norfair: bool
    force_phantoon_in_wrecked_ship: bool
    force_draygon_in_maridia: bool
    force_mother_brain_in_tourian: bool
    reward_frontier: float
    reward_graph_diameter: float
    maridia_water_preferred_probability_1: float
    maridia_water_preferred_probability_2: float
    maridia_water_preferred_probability_3: float
    norfair_heat_preferred_probability_1: float
    norfair_heat_preferred_probability_2: float
    norfair_heat_preferred_probability_3: float
    reward_save_distance: float
    reward_refill_distance: float
    reward_missing_connect_utility: float
    max_return_distance: float | None
    reward_area_distinct_crossing: float
    reward_area_crossing: float
    reward_area_size_valid: float
    reward_area_map_station: float
    reward_area_x: float
    reward_area_y: float
    target_area_rooms: Annotated[list[float], Field(min_length=6, max_length=6)]
    target_area_x: Annotated[list[float], Field(min_length=6, max_length=6)]
    target_area_y: Annotated[list[float], Field(min_length=6, max_length=6)]
    small_map: bool
    min_rooms: int | None = None
    max_rooms: int | None = None
    target_rooms: int | None = None


@dataclass
class ModelExport:
    training_config: Config
    tensors: dict[str, torch.Tensor]
    backbone_dtype: torch.dtype


class DoorLookup(NamedTuple):
    room_idx: list[int]
    door_id: list[int]


@dataclass
class DoorLookups:
    left: DoorLookup
    right: DoorLookup
    up: DoorLookup
    down: DoorLookup


@dataclass
class GenerationRunResult:
    episode_data: EpisodeData
    outcomes: EpisodeOutcomes
    door_matches: DoorMatches
    profile_report: list[tuple[str, int, int]]


@dataclass
class PrefetchQueueState:
    request: GenerateRequest
    maps: deque[dict]
    target: int
    refill_batches: int
    growth_armed: bool
    last_used: float
    last_low_capacity: float


@dataclass
class PrefetchState:
    queues: OrderedDict[str, PrefetchQueueState]
    condition: threading.Condition
    total_maps: int
    foreground_waiting: int
    active_key: str | None
    active_cancellation: threading.Event | None
    priority_key: str | None


@dataclass
class ServingState:
    serving_config: ServingConfig
    training_config: Config
    rooms: list[dict]
    device: torch.device
    envs: list
    model: torch.nn.Module
    balance_model: torch.nn.Module
    door_room_lookup: DoorRoomLookup
    door_lookups: DoorLookups
    door_data: DoorData
    room_part_data: RoomPartData
    required_room_data: RequiredRoomData
    profile: bool
    lock: threading.Lock
    prefetch: PrefetchState | None


SERVING_STATE: ServingState | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve map generation requests from an exported model safetensors file.",
    )
    parser.add_argument(
        "serving_config",
        type=Path,
        help="Serving config JSON file.",
    )
    parser.add_argument(
        "model_export",
        type=Path,
        help="Model export or training checkpoint safetensors file.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Collect and return Python/Rust generation profile timings.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Use deterministic environment group seeds starting from this value.",
    )
    return parser.parse_args()


def load_serving_config(path: Path) -> ServingConfig:
    return ServingConfig.model_validate_json(path.read_text())


def validate_model_input_metadata(path: Path, metadata: dict[str, str] | None) -> dict[str, str]:
    if metadata is None:
        raise ValueError(f"model input metadata missing in {path}")
    for field in ("format", "config"):
        if field not in metadata:
            raise ValueError(f"model input metadata field {field!r} missing in {path}")
    if metadata["format"] not in MODEL_INPUT_FORMATS:
        raise ValueError(f"unsupported model input format in {path}")
    if metadata["format"] == MODEL_EXPORT_FORMAT and "backbone_dtype" not in metadata:
        raise ValueError(f"model input metadata field 'backbone_dtype' missing in {path}")
    return metadata


def load_model_input(path: Path) -> ModelExport:
    with safe_open(path, framework="pt", device="cpu") as model_input:
        metadata = validate_model_input_metadata(path, model_input.metadata())
        if metadata["format"] == TRAINING_CHECKPOINT_FORMAT:
            backbone_dtype = torch.float32
            tensors = {
                name: model_input.get_tensor(name)
                for name in model_input.keys()
                if name.startswith(("ema_model.", "balance_model."))
            }
        else:
            backbone_dtype = inference_backbone_dtype(metadata["backbone_dtype"])
            tensors = {name: model_input.get_tensor(name) for name in model_input.keys()}
    missing_prefixes = [
        prefix
        for prefix in MODEL_PREFIXES
        if not any(name.startswith(f"{prefix}.") for name in tensors)
    ]
    if missing_prefixes:
        raise ValueError(f"model input missing tensor group(s): {', '.join(missing_prefixes)}")
    training_config = Config.model_validate_json(metadata["config"])
    validate_config(training_config)
    return ModelExport(
        training_config=training_config, tensors=tensors, backbone_dtype=backbone_dtype
    )


def create_environment_groups(
    serving_config: ServingConfig,
    training_config: Config,
    engine: Engine,
    seed: int | None,
) -> list:
    validate_serving_config(serving_config)
    generation_config = training_config.generation
    group_environments = serving_config.num_environments // serving_config.pipeline_groups
    group_threads = serving_config.num_threads // serving_config.pipeline_groups
    return [
        engine.create_environment_group(
            training_config.map_size,
            group_environments,
            generation_config.candidate_spatial_cell_size,
            generation_config.area_bounding_box_width,
            generation_config.area_bounding_box_height,
            seed=None if seed is None else seed + group_index,
            frontier_neighbor_algorithm=generation_config.frontier_neighbor_algorithm,
            frontier_neighbor_count=generation_config.frontier_neighbor_count,
            frontier_window_size=generation_config.frontier_window_size,
            num_threads=group_threads,
        )
        for group_index in range(serving_config.pipeline_groups)
    ]


def validate_serving_config(serving_config: ServingConfig) -> None:
    if serving_config.num_environments <= 0:
        raise ValueError("num_environments must be greater than zero")
    if serving_config.pipeline_groups <= 0:
        raise ValueError("pipeline_groups must be greater than zero")
    if serving_config.num_environments % serving_config.pipeline_groups != 0:
        raise ValueError("num_environments must be divisible by pipeline_groups")
    if serving_config.num_threads <= 0:
        raise ValueError("num_threads must be greater than zero")
    if serving_config.num_threads // serving_config.pipeline_groups <= 0:
        raise ValueError("num_threads must be at least pipeline_groups")
    if serving_config.gpu_prefetch_batches < 0:
        raise ValueError("gpu_prefetch_batches must be greater than or equal to zero")
    if serving_config.num_warmup_requests < 0:
        raise ValueError("num_warmup_requests must be greater than or equal to zero")
    if serving_config.prefetch_capacity < 0:
        raise ValueError("prefetch_capacity must be greater than or equal to zero")
    if serving_config.prefetch_initial_target <= 0:
        raise ValueError("prefetch_initial_target must be greater than zero")
    if (
        serving_config.prefetch_capacity > 0
        and serving_config.prefetch_initial_target > serving_config.prefetch_capacity
    ):
        raise ValueError("prefetch_initial_target must not exceed prefetch_capacity")
    if not 0 < serving_config.prefetch_low_capacity_fraction < 1:
        raise ValueError("prefetch_low_capacity_fraction must be between zero and one")
    if serving_config.prefetch_refill_batches <= 0:
        raise ValueError("prefetch_refill_batches must be greater than zero")
    if serving_config.foreground_max_batches <= 0:
        raise ValueError("foreground_max_batches must be greater than zero")
    if serving_config.prefetch_max_queues < 0:
        raise ValueError("prefetch_max_queues must be greater than or equal to zero")


def create_prefetch_state() -> PrefetchState:
    lock = threading.Lock()
    return PrefetchState(
        queues=OrderedDict(),
        condition=threading.Condition(lock),
        total_maps=0,
        foreground_waiting=0,
        active_key=None,
        active_cancellation=None,
        priority_key=None,
    )


def prefetch_enabled(serving_config: ServingConfig) -> bool:
    return serving_config.prefetch_capacity > 0 and serving_config.prefetch_max_queues > 0


def create_serving_state(
    serving_config: ServingConfig,
    model_export: ModelExport,
    seed: int | None,
    profile: bool,
) -> ServingState:
    rooms = json.loads(serving_config.room_set.read_text())
    device = torch.device(serving_config.device)
    if is_gpu(device):
        backend = gpu_backend(device)
        backend.set_device(device)
        backend.set_per_process_memory_fraction(
            serving_config.gpu_memory_fraction,
            device,
        )
        torch.set_float32_matmul_precision("high")
    engine = Engine(
        rooms,
        model_export.training_config.features,
        model_export.training_config.generation.min_area_size,
        model_export.training_config.generation.max_area_size,
    )
    model, balance_model = create_inference_models(
        model_export.training_config,
        rooms,
        engine,
        model_export.tensors,
        model_export.backbone_dtype,
        device,
    )
    if serving_config.compile_model:
        model = torch.compile(model)
        balance_model = torch.compile(balance_model)
    envs = create_environment_groups(serving_config, model_export.training_config, engine, seed)
    door_room_lookup = build_door_room_lookup(rooms, device)
    door_lookups = build_door_lookups(rooms)
    door_data = build_door_data(rooms)
    room_part_data = build_room_part_data(rooms)
    required_room_data = build_required_room_data(rooms)
    prefetch = create_prefetch_state() if prefetch_enabled(serving_config) else None
    state = ServingState(
        serving_config=serving_config,
        training_config=model_export.training_config,
        rooms=rooms,
        device=device,
        envs=envs,
        model=model,
        balance_model=balance_model,
        door_room_lookup=door_room_lookup,
        door_lookups=door_lookups,
        door_data=door_data,
        room_part_data=room_part_data,
        required_room_data=required_room_data,
        profile=profile,
        lock=threading.Lock(),
        prefetch=prefetch,
    )
    if prefetch is not None:
        threading.Thread(
            target=run_prefetch_worker,
            args=(state,),
            daemon=True,
        ).start()
    return state


def validate_generate_request(generate_request: GenerateRequest, rooms: list[dict]) -> None:
    start = generate_request.initial_item_start
    if start is not None:
        matching = [
            index + 1
            for index, node in enumerate(NODES)
            if (node["room_id"], node["node_id"]) == start and node["start"]
        ]
        if not matching or start[0] not in {room["room_id"] for room in rooms if "room_id" in room}:
            raise ValueError("initial_item_start must identify an extracted start in the room set")
    matching_graphs = [
        index
        for index, graph in enumerate(GRAPHS)
        if graph["difficulty"] == generate_request.initial_item_difficulty
        and graph["wall_jump"] == generate_request.initial_item_wall_jump
    ]
    if not matching_graphs:
        raise ValueError("unsupported initial_item_difficulty")
    if not math.isfinite(generate_request.reward_initial_item):
        raise ValueError("reward_initial_item must be finite")
    if generate_request.episode_length <= 0:
        raise ValueError("episode_length must be greater than zero")
    if generate_request.episode_length > len(rooms):
        raise ValueError("episode_length must not exceed the room count")
    if generate_request.recommended_candidates <= 0:
        raise ValueError("recommended_candidates must be greater than zero")
    if generate_request.shortlist_candidates < generate_request.recommended_candidates:
        raise ValueError("shortlist_candidates must be at least recommended_candidates")
    if generate_request.num_scored_invalid_candidates < 0:
        raise ValueError("num_scored_invalid_candidates must be greater than or equal to zero")
    if generate_request.num_scored_invalid_candidates > generate_request.shortlist_candidates:
        raise ValueError("num_scored_invalid_candidates must not exceed shortlist_candidates")
    if generate_request.max_candidate_areas_per_placement <= 0:
        raise ValueError("max_candidate_areas_per_placement must be greater than zero")
    if generate_request.max_candidate_areas_per_placement > 6:
        raise ValueError("max_candidate_areas_per_placement must be at most AREA_COUNT")
    if generate_request.temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    if generate_request.proposal_temperature <= 0:
        raise ValueError("proposal_temperature must be greater than zero")
    if generate_request.max_return_distance is not None and (
        not math.isfinite(generate_request.max_return_distance)
        or not 0 < generate_request.max_return_distance <= 254
    ):
        raise ValueError("max_return_distance must be null or a finite value in (0, 254]")
    for name in (
        "reward_area_distinct_crossing",
        "reward_area_crossing",
        "reward_area_size_valid",
        "reward_area_map_station",
        "reward_area_x",
        "reward_area_y",
    ):
        if not math.isfinite(getattr(generate_request, name)):
            raise ValueError(f"{name} must be finite")
    for family in ("maridia_water", "norfair_heat"):
        probabilities = [
            getattr(generate_request, f"{family}_preferred_probability_{tier}")
            for tier in range(1, 4)
        ]
        if any(not math.isfinite(value) for value in probabilities):
            raise ValueError(f"{family}_preferred_probability values must be finite")
        if not 0.0 < probabilities[0] <= probabilities[1] <= probabilities[2] < 1.0:
            raise ValueError(
                f"{family}_preferred_probability values must be positive, below one, and nondecreasing"
            )
    for name in ("target_area_rooms", "target_area_x", "target_area_y"):
        for area, value in enumerate(getattr(generate_request, name)):
            if not math.isfinite(value):
                raise ValueError(f"{name}[{area}] must be finite")
    if any(value <= 0.0 for value in generate_request.target_area_rooms):
        raise ValueError("target_area_rooms values must be greater than zero")
    if generate_request.small_map:
        missing_fields = [
            field
            for field in ("min_rooms", "max_rooms", "target_rooms")
            if getattr(generate_request, field) is None
        ]
        if missing_fields:
            raise ValueError(f"small_map requires {', '.join(missing_fields)}")
        assert generate_request.min_rooms is not None
        assert generate_request.max_rooms is not None
        assert generate_request.target_rooms is not None
        if generate_request.min_rooms <= 0:
            raise ValueError("min_rooms must be greater than zero")
        if generate_request.max_rooms < generate_request.min_rooms:
            raise ValueError("max_rooms must be at least min_rooms")
        if generate_request.target_rooms <= 0:
            raise ValueError("target_rooms must be greater than zero")


def create_generate_configs(
    generate_request: GenerateRequest,
    state: ServingState,
    envs: list,
    device: torch.device,
) -> list[GenerateConfig]:
    start_index = (
        0 if generate_request.initial_item_start is None else next(
            index + 1
            for index, node in enumerate(NODES)
            if (node["room_id"], node["node_id"]) == generate_request.initial_item_start
            and node["start"]
        )
    )
    graph_index = next(
        index
        for index, graph in enumerate(GRAPHS)
        if graph["difficulty"] == generate_request.initial_item_difficulty
        and graph["wall_jump"] == generate_request.initial_item_wall_jump
    )
    target_scales = {
        "target_area_x": state.training_config.map_size[0],
        "target_area_y": state.training_config.map_size[1],
    }
    normalized_targets = {
        name: [value / target_scales[name] for value in getattr(generate_request, name)]
        for name in target_scales
    }
    area_room_sum = sum(generate_request.target_area_rooms)
    normalized_targets["target_area_rooms"] = [
        value * len(state.rooms) / area_room_sum for value in generate_request.target_area_rooms
    ]
    generation_variable_float_values = {
        "temperature": generate_request.temperature,
        "proposal_temperature": generate_request.proposal_temperature,
        "reward_door": generate_request.reward_door,
        "reward_connection": generate_request.reward_connection,
        "reward_toilet": generate_request.reward_toilet,
        "reward_phantoon_pair": generate_request.reward_phantoon_pair,
        "reward_phantoon_area": generate_request.reward_phantoon_area,
        "reward_success": generate_request.reward_success,
        "reward_initial_item": generate_request.reward_initial_item * (start_index > 0),
        "initial_item_start": float(start_index),
        "initial_item_graph": float(graph_index),
        "reward_frontier": generate_request.reward_frontier,
        "reward_graph_diameter": generate_request.reward_graph_diameter,
        **{
            f"{family}_preferred_probability_{tier}": getattr(
                generate_request, f"{family}_preferred_probability_{tier}"
            )
            for family in ("maridia_water", "norfair_heat")
            for tier in range(1, 4)
        },
        "reward_save_distance": generate_request.reward_save_distance,
        "reward_refill_distance": generate_request.reward_refill_distance,
        "reward_missing_connect_utility": generate_request.reward_missing_connect_utility,
        "max_return_distance": (
            0.0 if generate_request.max_return_distance is None
            else generate_request.max_return_distance
        ),
        "reward_area_distinct_crossing": generate_request.reward_area_distinct_crossing,
        "reward_area_crossing": generate_request.reward_area_crossing,
        "reward_area_size_valid": generate_request.reward_area_size_valid,
        "reward_area_map_station": generate_request.reward_area_map_station,
        "reward_area_x": generate_request.reward_area_x,
        "reward_area_y": generate_request.reward_area_y,
    }
    generation_variable_float_values.update(
        {
            reward_field: getattr(generate_request, reward_field)
            * float(getattr(generate_request, condition_field))
            for reward_field, condition_field in zip(
                VANILLA_AREA_REWARD_FIELDS,
                VANILLA_AREA_CONDITION_FIELDS,
            )
        }
    )
    generation_variable_float_values.update(
        {
            field_name: float(getattr(generate_request, field_name))
            for field_name in VANILLA_AREA_CONDITION_FIELDS
        }
    )
    vanilla_area_constraint_values = [
        getattr(generate_request, f"force_{name}") for name in VANILLA_AREA_OUTCOME_NAMES
    ]
    generation_variable_float_values.update(
        {
            f"{name}_{area}": values[area]
            for name, values in normalized_targets.items()
            for area in range(6)
        }
    )
    generation_variable_tensor = GenerationVariableFloatsFeature.construct_tensor(
        {
            name: torch.tensor([value], dtype=torch.float32, device=device)
            for name, value in generation_variable_float_values.items()
        },
        num_rooms=len(state.rooms),
    )
    configs = []
    for env in envs:
        temperature = torch.full(
            [env.num_envs],
            generate_request.temperature,
            dtype=torch.float32,
            device=device,
        )
        proposal_temperature = torch.full(
            [env.num_envs],
            generate_request.proposal_temperature,
            dtype=torch.float32,
            device=device,
        )
        generation_variable_floats_model = (
            generation_variable_tensor.expand(env.num_envs, len(GENERATION_VARIABLE_FLOAT_FIELDS))
            .contiguous()
        )
        log_temperature_model = generation_variable_floats_model[
            :, GENERATION_VARIABLE_FLOAT_FIELDS.index("log_temperature")
        ].detach().contiguous()
        log_recommended_candidates_model = torch.full(
            [env.num_envs],
            math.log(generate_request.recommended_candidates + 1),
            dtype=torch.float32,
            device=device,
        )
        candidate_shape = torch.Size([env.num_envs, generate_request.recommended_candidates])
        candidate_log_temperature_model = (
            log_temperature_model.unsqueeze(1).expand(candidate_shape).contiguous()
        )
        candidate_log_recommended_candidates_model = torch.full(
            candidate_shape,
            math.log(generate_request.recommended_candidates + 1),
            dtype=torch.float32,
            device=device,
        )
        candidate_generation_variable_floats_model = (
            generation_variable_floats_model.unsqueeze(1)
            .expand(
                env.num_envs,
                generate_request.recommended_candidates,
                len(GENERATION_VARIABLE_FLOAT_FIELDS),
            )
            .contiguous()
        )
        target_area_rooms = torch.tensor(
            normalized_targets["target_area_rooms"], dtype=torch.float32, device=device
        ).expand(env.num_envs, AREA_COUNT)
        preferred_area_probability = torch.tensor(
            [
                [
                    getattr(generate_request, f"{family}_preferred_probability_{tier}")
                    for tier in range(1, 4)
                ]
                for family in ("maridia_water", "norfair_heat")
            ],
            dtype=torch.float32,
            device=device,
        ).expand(env.num_envs, 2, 3)
        vanilla_area_constraint_mask = torch.tensor(
            vanilla_area_constraint_values, dtype=torch.bool, device=device
        ).expand(env.num_envs, AREA_COUNT)
        area_balance_targets = compute_area_balance_targets(
            state.rooms,
            target_area_rooms,
            vanilla_area_constraint_mask,
            preferred_area_probability,
        )
        configs.append(
            GenerateConfig(
                episode_length=generate_request.episode_length,
                recommended_candidates=generate_request.recommended_candidates,
                shortlist_candidates=generate_request.shortlist_candidates,
                num_scored_invalid_candidates=(generate_request.num_scored_invalid_candidates),
                max_candidate_areas_per_placement=(
                    generate_request.max_candidate_areas_per_placement
                ),
                recommended_candidates_same_frontier=(
                    generate_request.recommended_candidates_same_frontier
                ),
                guarantee_phantoon_counterpart=generate_request.guarantee_phantoon_counterpart,
                gpu_prefetch_batches=state.serving_config.gpu_prefetch_batches,
                temperature=temperature,
                proposal_temperature=proposal_temperature,
                reward_door=generate_request.reward_door,
                reward_connection=generate_request.reward_connection,
                reward_toilet=generate_request.reward_toilet,
                reward_phantoon_pair=generate_request.reward_phantoon_pair,
                reward_phantoon_area=generate_request.reward_phantoon_area,
                reward_success=generate_request.reward_success,
                reward_initial_item=generate_request.reward_initial_item * (start_index > 0),
                reward_vanilla_area=torch.tensor(
                    [getattr(generate_request, name) for name in VANILLA_AREA_REWARD_FIELDS],
                    dtype=torch.float32,
                    device=device,
                ).expand(env.num_envs, 6),
                reward_frontier=generate_request.reward_frontier,
                reward_graph_diameter=generate_request.reward_graph_diameter,
                preferred_area_probability=preferred_area_probability,
                reward_save_distance=generate_request.reward_save_distance,
                reward_refill_distance=generate_request.reward_refill_distance,
                reward_missing_connect_utility=generate_request.reward_missing_connect_utility,
                reward_area_distinct_crossing=generate_request.reward_area_distinct_crossing,
                reward_area_crossing=generate_request.reward_area_crossing,
                reward_area_size_valid=generate_request.reward_area_size_valid,
                reward_area_map_station=generate_request.reward_area_map_station,
                reward_area_x=generate_request.reward_area_x,
                reward_area_y=generate_request.reward_area_y,
                target_area_rooms=target_area_rooms,
                target_area_x=torch.tensor(
                    normalized_targets["target_area_x"], dtype=torch.float32, device=device
                ).expand(env.num_envs, 6),
                target_area_y=torch.tensor(
                    normalized_targets["target_area_y"], dtype=torch.float32, device=device
                ).expand(env.num_envs, 6),
                vanilla_area_constraint_mask=vanilla_area_constraint_mask,
                area_balance_probability=area_balance_targets.probability,
                area_balance_dual_mask=area_balance_targets.dual_mask,
                effective_target_area_rooms=area_balance_targets.effective_area_rooms,
                generation_variable_floats=generation_variable_floats_model,
                log_temperature_model=log_temperature_model,
                log_recommended_candidates_model=log_recommended_candidates_model,
                generation_variable_floats_model=generation_variable_floats_model,
                candidate_log_temperature_model=candidate_log_temperature_model,
                candidate_log_recommended_candidates_model=(
                    candidate_log_recommended_candidates_model
                ),
                candidate_generation_variable_floats_model=(
                    candidate_generation_variable_floats_model
                ),
                distance_proximity_scale=state.training_config.distance_proximity_scale,
                autocast=state.serving_config.autocast,
            )
        )
    return configs


def tensor_to_list(tensor: torch.Tensor) -> list:
    return tensor.detach().cpu().tolist()


def flat_doors(room: dict) -> list[dict]:
    return [door for door_group in room["doors"] for door in door_group]


def build_door_lookups(rooms: list[dict]) -> DoorLookups:
    door_rooms = {direction: [] for direction in DIRECTIONS}
    door_ids = {direction: [] for direction in DIRECTIONS}
    for room_idx, room in enumerate(rooms):
        for door in flat_doors(room):
            direction = door["direction"]
            door_rooms[direction].append(room_idx)
            door_ids[direction].append(int(door["id"]))
    return DoorLookups(
        left=DoorLookup(room_idx=door_rooms["left"], door_id=door_ids["left"]),
        right=DoorLookup(room_idx=door_rooms["right"], door_id=door_ids["right"]),
        up=DoorLookup(room_idx=door_rooms["up"], door_id=door_ids["up"]),
        down=DoorLookup(room_idx=door_rooms["down"], door_id=door_ids["down"]),
    )


def tensor_has_invalid_outcome(tensor: torch.Tensor) -> torch.Tensor:
    invalid = tensor > 0
    if invalid.ndim == 1:
        return invalid
    return torch.any(invalid, dim=tuple(range(1, invalid.ndim)))


def valid_map_mask(
    outcomes,
    vanilla_area_constraint_mask: torch.Tensor,
    min_area_size: int,
    max_area_size: int,
) -> torch.Tensor:
    step_outcomes = outcomes.step_outcomes
    end_outcomes = outcomes.end_outcomes
    area_size_invalid = (end_outcomes.area_size < min_area_size) | (
        end_outcomes.area_size > max_area_size
    )
    area_map_station_invalid = end_outcomes.area_map_station_count != 1
    invalid = (
        tensor_has_invalid_outcome(step_outcomes.door_invalid)
        | tensor_has_invalid_outcome(step_outcomes.connection_invalid)
        | tensor_has_invalid_outcome(step_outcomes.return_distance_invalid)
        | tensor_has_invalid_outcome(step_outcomes.toilet_invalid)
        | tensor_has_invalid_outcome(step_outcomes.phantoon_pair_invalid)
        | tensor_has_invalid_outcome(step_outcomes.phantoon_area_invalid)
        | tensor_has_invalid_outcome(
            step_outcomes.vanilla_area_invalid & vanilla_area_constraint_mask
        )
        | tensor_has_invalid_outcome(area_size_invalid)
        | tensor_has_invalid_outcome(area_map_station_invalid)
    )
    return ~invalid


def collect_door_matches(envs: list, device: torch.device) -> DoorMatches:
    group_door_matches = [env.get_door_matches(device) for env in envs]
    return DoorMatches(
        left=torch.cat([door_matches.left for door_matches in group_door_matches]),
        right=torch.cat([door_matches.right for door_matches in group_door_matches]),
        up=torch.cat([door_matches.up for door_matches in group_door_matches]),
        down=torch.cat([door_matches.down for door_matches in group_door_matches]),
    )


def filter_door_matches(door_matches: DoorMatches, mask: torch.Tensor) -> DoorMatches:
    return DoorMatches(
        left=door_matches.left[mask],
        right=door_matches.right[mask],
        up=door_matches.up[mask],
        down=door_matches.down[mask],
    )


def select_door_matches(door_matches: DoorMatches, index: torch.Tensor) -> DoorMatches:
    return DoorMatches(
        left=door_matches.left[index],
        right=door_matches.right[index],
        up=door_matches.up[index],
        down=door_matches.down[index],
    )


def placement_positions_by_room(room_idx: list[int]) -> dict[int, int]:
    return {room: position for position, room in enumerate(room_idx)}


def append_edge(
    edges: dict[str, list[int]],
    seen_edges: set[tuple[int, int, int, int]],
    from_endpoint: tuple[int, int],
    to_endpoint: tuple[int, int],
) -> None:
    if to_endpoint < from_endpoint:
        from_endpoint, to_endpoint = to_endpoint, from_endpoint
    edge = (*from_endpoint, *to_endpoint)
    if edge in seen_edges:
        return
    seen_edges.add(edge)
    edges["from_room_placement_idx"].append(from_endpoint[0])
    edges["from_door_id"].append(from_endpoint[1])
    edges["to_room_placement_idx"].append(to_endpoint[0])
    edges["to_door_id"].append(to_endpoint[1])


def collect_response_direction_edges(
    edges: dict[str, list[int]],
    seen_edges: set[tuple[int, int, int, int]],
    source_matches: list[int],
    source_lookup: DoorLookup,
    target_lookup: DoorLookup,
    room_position: dict[int, int],
) -> None:
    for source_direction_door_idx, target_direction_door_idx in enumerate(source_matches):
        if target_direction_door_idx < 0:
            continue
        source_room_idx = source_lookup.room_idx[source_direction_door_idx]
        target_room_idx = target_lookup.room_idx[target_direction_door_idx]
        source_room_position = room_position.get(source_room_idx)
        target_room_position = room_position.get(target_room_idx)
        if source_room_position is None or target_room_position is None:
            continue
        append_edge(
            edges,
            seen_edges,
            (source_room_position, source_lookup.door_id[source_direction_door_idx]),
            (target_room_position, target_lookup.door_id[target_direction_door_idx]),
        )


def response_edges(
    room_idx: list[list[int]],
    door_matches: DoorMatches,
    door_lookups: DoorLookups,
) -> dict[str, list[list[int]]]:
    edge_lists = {
        "from_room_placement_idx": [],
        "from_door_id": [],
        "to_room_placement_idx": [],
        "to_door_id": [],
    }
    direction_matches = {
        "left": tensor_to_list(door_matches.left),
        "right": tensor_to_list(door_matches.right),
        "up": tensor_to_list(door_matches.up),
        "down": tensor_to_list(door_matches.down),
    }
    direction_lookups = {
        "left": door_lookups.left,
        "right": door_lookups.right,
        "up": door_lookups.up,
        "down": door_lookups.down,
    }
    for map_idx, map_room_idx in enumerate(room_idx):
        map_edges = {key: [] for key in edge_lists}
        seen_edges = set()
        room_position = placement_positions_by_room(map_room_idx)
        for direction in DIRECTIONS:
            collect_response_direction_edges(
                map_edges,
                seen_edges,
                direction_matches[direction][map_idx],
                direction_lookups[direction],
                direction_lookups[OPPOSITE_DIRECTIONS[direction]],
                room_position,
            )
        for key, values in map_edges.items():
            edge_lists[key].append(values)
    return edge_lists


def response_toilet_crossing_room_placement_idx(
    room_idx: list[list[int]],
    toilet_crossed_room_idx: list[int],
) -> list[int]:
    return [
        placement_positions_by_room(map_room_idx).get(crossed_room_idx, -1)
        for map_room_idx, crossed_room_idx in zip(room_idx, toilet_crossed_room_idx)
    ]


def response_room_ids(room_idx: list[list[int]], rooms: list[dict]) -> list[list[int]]:
    return [[int(rooms[room]["room_id"]) for room in map_room_idx] for map_room_idx in room_idx]


def initialize_serving_state(state: ServingState) -> None:
    global SERVING_STATE
    SERVING_STATE = state


def warmup_generate_request() -> GenerateRequest:
    return GenerateRequest(
        max_num_maps=1,
        episode_length=253,
        recommended_candidates=8,
        shortlist_candidates=1024,
        num_scored_invalid_candidates=0,
        max_candidate_areas_per_placement=2,
        recommended_candidates_same_frontier=False,
        guarantee_phantoon_counterpart=True,
        temperature=0.04,
        proposal_temperature=0.03,
        reward_door=1.0,
        reward_connection=1.0,
        reward_toilet=1.0,
        reward_phantoon_pair=1.0,
        reward_phantoon_area=1.0,
        reward_success=0.0,
        reward_initial_item=0.0,
        initial_item_start=None,
        initial_item_difficulty="Basic",
        initial_item_wall_jump=True,
        reward_ship_in_crateria=1.0,
        reward_kraid_in_brinstar=1.0,
        reward_ridley_in_norfair=1.0,
        reward_phantoon_in_wrecked_ship=1.0,
        reward_draygon_in_maridia=1.0,
        reward_mother_brain_in_tourian=1.0,
        force_ship_in_crateria=True,
        force_kraid_in_brinstar=True,
        force_ridley_in_norfair=True,
        force_phantoon_in_wrecked_ship=True,
        force_draygon_in_maridia=True,
        force_mother_brain_in_tourian=True,
        reward_frontier=0.0,
        reward_graph_diameter=0.1,
        maridia_water_preferred_probability_1=0.3,
        maridia_water_preferred_probability_2=0.45,
        maridia_water_preferred_probability_3=0.6,
        norfair_heat_preferred_probability_1=0.3,
        norfair_heat_preferred_probability_2=0.45,
        norfair_heat_preferred_probability_3=0.6,
        reward_save_distance=0.1,
        reward_refill_distance=0.1,
        reward_missing_connect_utility=1.0,
        max_return_distance=None,
        reward_area_distinct_crossing=1.0,
        reward_area_crossing=0.0,
        reward_area_size_valid=1.0,
        reward_area_map_station=1.0,
        reward_area_x=5.0,
        reward_area_y=5.0,
        target_area_rooms=[150, 300, 300, 100, 200, 100],
        target_area_x=[16, 10, 40, 48, 40, 16],
        target_area_y=[12, 32, 64, 12, 32, 24],
        small_map=False,
    )


def run_warmup_requests(state: ServingState) -> None:
    for request_idx in range(state.serving_config.num_warmup_requests):
        logging.info(
            "Running warmup request %s/%s",
            request_idx + 1,
            state.serving_config.num_warmup_requests,
        )
        _ = generate_response_data_uncached_validated(state, warmup_generate_request())


def serving_state() -> ServingState:
    if SERVING_STATE is None:
        raise RuntimeError("serving state has not been initialized")
    return SERVING_STATE


def add_serving_profile(
    profiler: GenerationProfiler,
    device: torch.device,
    name: str,
    start: int,
) -> None:
    sync_profile_device(device, profiler.enabled)
    profiler.add(name, start)


def prefetch_request_key(generate_request: GenerateRequest) -> str:
    return generate_request.model_dump_json(exclude={"max_num_maps"})


def queue_id(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def log_queue_state(state: ServingState, key: str, event: str) -> None:
    # Call with the queue condition held when prefetch is enabled.
    prefetch = state.prefetch
    queue = None if prefetch is None else prefetch.queues.get(key)
    logging.info(
        "%s queue=%s size=%d target=%d refill_batches=%d total_maps=%d/%d",
        event,
        queue_id(key),
        0 if queue is None else len(queue.maps),
        0 if queue is None else queue.target,
        0 if queue is None else queue.refill_batches,
        0 if prefetch is None else prefetch.total_maps,
        state.serving_config.prefetch_capacity,
    )


def prefetch_eviction_order(prefetch: PrefetchState, key: str) -> list[str]:
    return sorted(
        (other for other in prefetch.queues if other != key),
        key=lambda other: (
            prefetch.queues[other].last_low_capacity,
            prefetch.queues[other].last_used,
        ),
    )


def drop_queue_maps(state: ServingState, key: str, count: int) -> None:
    prefetch = state.prefetch
    queue = prefetch.queues[key]
    for _ in range(count):
        queue.maps.pop()
    prefetch.total_maps -= count
    queue.target = min(queue.target, len(queue.maps))
    queue.growth_armed = False
    queue.refill_batches = 0
    log_queue_state(state, key, f"Evicted {count} maps")


def reserve_queue_target(state: ServingState, key: str, target: int) -> None:
    prefetch = state.prefetch
    queue = prefetch.queues[key]
    queue.target = min(target, state.serving_config.prefetch_capacity)
    excess = (
        sum(other.target for other in prefetch.queues.values())
        - state.serving_config.prefetch_capacity
    )
    for other_key in prefetch_eviction_order(prefetch, key):
        if excess <= 0:
            break
        other = prefetch.queues[other_key]
        reduction = min(excess, other.target)
        other.target -= reduction
        excess -= reduction
        if len(other.maps) > other.target:
            drop_queue_maps(state, other_key, len(other.maps) - other.target)
        elif reduction:
            other.growth_armed = False
            if other.target == 0:
                other.refill_batches = 0
            log_queue_state(state, other_key, "Reduced target")


def mark_queue_full(queue: PrefetchQueueState) -> None:
    if queue.target > 0 and len(queue.maps) >= queue.target:
        queue.growth_armed = True
        queue.refill_batches = 0


def touch_prefetch_queue(
    state: ServingState,
    key: str,
    generate_request: GenerateRequest,
) -> PrefetchQueueState:
    if state.prefetch is None:
        raise RuntimeError("prefetch is not enabled")
    prefetch = state.prefetch
    queue_state = prefetch.queues.get(key)
    if queue_state is None:
        queue_state = PrefetchQueueState(
            request=generate_request,
            maps=deque(),
            target=0,
            refill_batches=0,
            growth_armed=False,
            last_used=time.monotonic(),
            last_low_capacity=-math.inf,
        )
        prefetch.queues[key] = queue_state
        while len(prefetch.queues) > state.serving_config.prefetch_max_queues:
            victim = prefetch_eviction_order(prefetch, key)[0]
            drop_queue_maps(state, victim, len(prefetch.queues[victim].maps))
            del prefetch.queues[victim]
    else:
        prefetch.queues.move_to_end(key)
        queue_state.last_used = time.monotonic()
    if queue_state.target == 0:
        reserve_queue_target(state, key, state.serving_config.prefetch_initial_target)
    if queue_state.target < generate_request.max_num_maps:
        reserve_queue_target(state, key, generate_request.max_num_maps)
    mark_queue_full(queue_state)
    return queue_state


def split_response_maps(response: dict) -> list[dict]:
    return [
        {
            "rooms": {name: values[index] for name, values in response["rooms"].items()},
            "edges": {name: values[index] for name, values in response["edges"].items()},
            "toilet_crossing_room_placement_idx": response["toilet_crossing_room_placement_idx"][
                index
            ],
        }
        for index in range(len(response["rooms"]["id"]))
    ]


def append_queue_maps(state: ServingState, key: str, maps: list[dict]) -> None:
    prefetch = state.prefetch
    queue = prefetch.queues[key]
    needed = min(len(maps), max(0, queue.target - len(queue.maps)))
    shortage = needed - (state.serving_config.prefetch_capacity - prefetch.total_maps)
    for victim in prefetch_eviction_order(prefetch, key):
        if shortage <= 0:
            break
        count = min(shortage, len(prefetch.queues[victim].maps))
        if count:
            drop_queue_maps(state, victim, count)
            shortage -= count
    count = min(len(maps), state.serving_config.prefetch_capacity - prefetch.total_maps)
    queue.maps.extend(maps[:count])
    prefetch.total_maps += count
    mark_queue_full(queue)
    log_queue_state(state, key, f"Stored {count}/{len(maps)} generated maps")


def take_queue_maps(state: ServingState, key: str, count: int) -> list[dict]:
    prefetch = state.prefetch
    queue = prefetch.queues[key]
    maps = [queue.maps.popleft() for _ in range(count)]
    prefetch.total_maps -= count
    if len(queue.maps) < queue.target * state.serving_config.prefetch_low_capacity_fraction:
        queue.last_low_capacity = time.monotonic()
        if queue.growth_armed:
            queue.growth_armed = False
            reserve_queue_target(state, key, queue.target * 2)
            log_queue_state(state, key, "Low capacity: increased target")
    prefetch.condition.notify_all()
    return maps


def acquire_foreground_generation_lock(state: ServingState, key: str) -> None:
    if state.prefetch is not None:
        with state.prefetch.condition:
            state.prefetch.foreground_waiting += 1
            state.prefetch.priority_key = key
            if state.prefetch.active_key is not None:
                if state.prefetch.active_key != key:
                    log_queue_state(
                        state,
                        state.prefetch.active_key,
                        f"Cancellation triggered by foreground queue={queue_id(key)}",
                    )
                    state.prefetch.active_cancellation.set()
                else:
                    log_queue_state(state, key, "Foreground waiting for matching background batch")
            state.prefetch.condition.notify_all()
    state.lock.acquire()
    if state.prefetch is not None:
        with state.prefetch.condition:
            state.prefetch.foreground_waiting -= 1
            state.prefetch.condition.notify_all()


def release_generation_lock(state: ServingState) -> None:
    state.lock.release()
    if state.prefetch is not None:
        with state.prefetch.condition:
            state.prefetch.condition.notify_all()


def run_generation_with_generation_lock_held(
    state: ServingState,
    configs: list[GenerateConfig],
    serving_profiler: GenerationProfiler,
    cancellation_event: threading.Event,
) -> GenerationRunResult:
    if cancellation_event.is_set():
        raise GenerationCancelled("generation pipeline cancelled")
    profile_time = profile_start(state.profile)
    (
        episode_data,
        outcomes,
        _door_match_counts,
        _proposal_data,
        _generated_feature_data,
        _generation_stats,
        profile_report,
    ) = run_generation_groups(
        state.envs,
        state.model,
        state.balance_model,
        configs,
        state.device,
        cancellation_event=cancellation_event,
        verify_outcome_consistency=state.serving_config.verify_outcome_consistency,
        profile=state.profile,
    )
    add_serving_profile(
        serving_profiler,
        state.device,
        "python.serve.run_generation_groups",
        profile_time,
    )

    profile_time = profile_start(state.profile)
    door_matches = collect_door_matches(state.envs, state.device)
    add_serving_profile(
        serving_profiler,
        state.device,
        "python.serve.collect_door_matches",
        profile_time,
    )
    return GenerationRunResult(
        episode_data=episode_data,
        outcomes=outcomes,
        door_matches=door_matches,
        profile_report=profile_report,
    )


def run_foreground_generation(
    state: ServingState,
    configs: list[GenerateConfig],
    serving_profiler: GenerationProfiler,
    key: str,
) -> GenerationRunResult:
    acquire_foreground_generation_lock(state, key)
    try:
        with torch.inference_mode():
            return run_generation_with_generation_lock_held(
                state, configs, serving_profiler, threading.Event()
            )
    finally:
        release_generation_lock(state)


def build_generate_response_data(
    state: ServingState,
    generate_request: GenerateRequest,
    serving_profiler: GenerationProfiler,
    request_start: int,
    generation_result: GenerationRunResult,
) -> dict:
    episode_data = generation_result.episode_data
    outcomes = generation_result.outcomes
    door_matches = generation_result.door_matches
    profile_report = generation_result.profile_report

    profile_time = profile_start(state.profile)
    valid_mask = valid_map_mask(
        outcomes,
        torch.tensor(
            [getattr(generate_request, name) for name in VANILLA_AREA_CONDITION_FIELDS],
            dtype=torch.bool,
            device=outcomes.step_outcomes.vanilla_area_invalid.device,
        ),
        state.training_config.generation.min_area_size,
        state.training_config.generation.max_area_size,
    )
    valid_room_idx = episode_data.actions.room_idx[valid_mask]
    valid_room_area = episode_data.actions.room_area[valid_mask]
    valid_room_x = episode_data.actions.room_x[valid_mask]
    valid_room_y = episode_data.actions.room_y[valid_mask]
    valid_toilet_crossed_room_idx = outcomes.end_outcomes.toilet_crossed_room_idx[valid_mask]
    valid_door_matches = filter_door_matches(door_matches, valid_mask)
    add_serving_profile(
        serving_profiler,
        state.device,
        "python.serve.filter_valid_maps",
        profile_time,
    )

    profile_time = profile_start(state.profile)
    adjacency = build_room_adjacency(valid_room_idx, valid_door_matches, state.door_room_lookup)
    subarea = split_assignment_by_balanced_centers(adjacency, (valid_room_area,), (AREA_COUNT,))
    subsubarea = split_assignment_by_balanced_centers(
        adjacency, (valid_room_area, subarea), (AREA_COUNT, 2)
    )
    add_serving_profile(
        serving_profiler,
        state.device,
        "python.serve.subdivide_areas",
        profile_time,
    )

    profile_time = profile_start(state.profile)
    final_door_matches = valid_door_matches
    final_room_idx_list = tensor_to_list(valid_room_idx)
    final_room_x_list = tensor_to_list(valid_room_x)
    final_room_y_list = tensor_to_list(valid_room_y)
    final_area_list = tensor_to_list(valid_room_area)
    final_subarea_list = tensor_to_list(subarea)
    final_subsubarea_list = tensor_to_list(subsubarea)
    final_toilet_crossed_room_idx_list = tensor_to_list(valid_toilet_crossed_room_idx)
    if generate_request.small_map:
        assert generate_request.min_rooms is not None
        assert generate_request.max_rooms is not None
        assert generate_request.target_rooms is not None
        small_map_result = prune_small_maps(
            room_idx=final_room_idx_list,
            room_x=final_room_x_list,
            room_y=final_room_y_list,
            area=final_area_list,
            subarea=final_subarea_list,
            subsubarea=final_subsubarea_list,
            toilet_crossed_room_idx=final_toilet_crossed_room_idx_list,
            door_matches=final_door_matches,
            door_data=state.door_data,
            room_part_data=state.room_part_data,
            required_room_data=state.required_room_data,
            config=SmallMapConfig(
                min_rooms=generate_request.min_rooms,
                max_rooms=generate_request.max_rooms,
                target_rooms=generate_request.target_rooms,
                max_return_distance=generate_request.max_return_distance,
            ),
        )
        final_room_idx_list = small_map_result.room_idx
        final_room_x_list = small_map_result.room_x
        final_room_y_list = small_map_result.room_y
        final_area_list = small_map_result.area
        final_subarea_list = small_map_result.subarea
        final_subsubarea_list = small_map_result.subsubarea
        final_toilet_crossed_room_idx_list = small_map_result.toilet_crossed_room_idx
        small_map_index = torch.tensor(
            small_map_result.source_map_idx,
            device=state.device,
            dtype=torch.int64,
        )
        final_door_matches = select_door_matches(final_door_matches, small_map_index)
    final_room_id_list = response_room_ids(final_room_idx_list, state.rooms)
    num_generated = int(episode_data.actions.room_idx.shape[0])
    num_pre_valid = int(torch.sum(valid_mask).item())
    num_valid = len(final_room_idx_list)
    add_serving_profile(
        serving_profiler,
        state.device,
        "python.serve.prepare_response_tensors",
        profile_time,
    )

    profile_time = profile_start(state.profile)
    response = {
        "stats": {
            "num_generated": num_generated,
            "num_pre_valid": num_pre_valid,
            "num_valid": num_valid,
        },
        "rooms": {
            "id": final_room_id_list,
            "x": final_room_x_list,
            "y": final_room_y_list,
            "area": final_area_list,
            "subarea": final_subarea_list,
            "subsubarea": final_subsubarea_list,
        },
        "edges": response_edges(final_room_idx_list, final_door_matches, state.door_lookups),
        "toilet_crossing_room_placement_idx": response_toilet_crossing_room_placement_idx(
            final_room_idx_list,
            final_toilet_crossed_room_idx_list,
        ),
    }
    add_serving_profile(
        serving_profiler,
        state.device,
        "python.serve.build_response_json",
        profile_time,
    )
    add_serving_profile(serving_profiler, state.device, "python.serve.total", request_start)
    if state.profile:
        response["profile"] = profile_report + serving_profiler.report()
    logging.info("Response stats: %s", response["stats"])
    return response


def generate_response_data_uncached_validated(
    state: ServingState,
    generate_request: GenerateRequest,
) -> dict:
    serving_profiler = GenerationProfiler(state.profile)
    request_start = profile_start(state.profile)

    profile_time = profile_start(state.profile)
    configs = create_generate_configs(generate_request, state, state.envs, state.device)
    add_serving_profile(
        serving_profiler,
        state.device,
        "python.serve.create_generate_configs",
        profile_time,
    )

    generation_result = run_foreground_generation(
        state, configs, serving_profiler, prefetch_request_key(generate_request)
    )
    return build_generate_response_data(
        state,
        generate_request,
        serving_profiler,
        request_start,
        generation_result,
    )


def generate_batch_with_generation_lock_held(
    state: ServingState,
    generate_request: GenerateRequest,
    cancellation_event: threading.Event,
) -> dict:
    serving_profiler = GenerationProfiler(state.profile)
    request_start = profile_start(state.profile)

    profile_time = profile_start(state.profile)
    configs = create_generate_configs(generate_request, state, state.envs, state.device)
    add_serving_profile(
        serving_profiler,
        state.device,
        "python.serve.create_generate_configs",
        profile_time,
    )

    with torch.inference_mode():
        generation_result = run_generation_with_generation_lock_held(
            state,
            configs,
            serving_profiler,
            cancellation_event,
        )
    if cancellation_event.is_set():
        raise GenerationCancelled("generation pipeline cancelled")
    return build_generate_response_data(
        state,
        generate_request,
        serving_profiler,
        request_start,
        generation_result,
    )


def next_prefetch_refill_key(state: ServingState) -> str | None:
    prefetch = state.prefetch
    if prefetch.foreground_waiting:
        return None
    eligible = [
        key
        for key, queue in prefetch.queues.items()
        if queue.refill_batches > 0 and len(queue.maps) < queue.target
    ]
    if prefetch.priority_key in eligible:
        return prefetch.priority_key
    return (
        min(
            eligible,
            key=lambda key: (
                len(prefetch.queues[key].maps) / prefetch.queues[key].target,
                -prefetch.queues[key].last_used,
            ),
        )
        if eligible
        else None
    )


def run_prefetch_worker(state: ServingState) -> None:
    prefetch = state.prefetch
    while True:
        with prefetch.condition:
            while True:
                key = next_prefetch_refill_key(state)
                if key is not None and state.lock.acquire(blocking=False):
                    break
                prefetch.condition.wait()
            queue = prefetch.queues[key]
            cancellation_event = threading.Event()
            prefetch.active_key = key
            prefetch.active_cancellation = cancellation_event
            prefetch.priority_key = None
            queue.refill_batches -= 1
            log_queue_state(state, key, "Background batch started")
        try:
            response = generate_batch_with_generation_lock_held(
                state, queue.request, cancellation_event
            )
            maps = split_response_maps(response)
            with prefetch.condition:
                if prefetch.queues.get(key) is queue:
                    append_queue_maps(state, key, maps)
                    log_queue_state(state, key, "Background batch finished")
        except GenerationCancelled:
            logging.info("Background batch cancelled queue=%s", queue_id(key))
        except Exception:
            logging.exception("Background refill failed queue=%s", queue_id(key))
            with prefetch.condition:
                queue.refill_batches = 0
        finally:
            try:
                if cancellation_event.is_set() and is_gpu(state.device):
                    gpu_backend(state.device).synchronize(state.device)
            finally:
                with prefetch.condition:
                    prefetch.active_key = None
                    prefetch.active_cancellation = None
                    state.lock.release()
                    prefetch.condition.notify_all()


def combine_response_maps(maps: list[dict], stats: dict[str, int]) -> dict:
    return {
        "stats": stats | {"num_valid": len(maps)},
        "rooms": {
            name: [entry["rooms"][name] for entry in maps]
            for name in ("id", "x", "y", "area", "subarea", "subsubarea")
        },
        "edges": {
            name: [entry["edges"][name] for entry in maps]
            for name in (
                "from_room_placement_idx", "from_door_id",
                "to_room_placement_idx", "to_door_id",
            )
        },
        "toilet_crossing_room_placement_idx": [
            entry["toilet_crossing_room_placement_idx"] for entry in maps
        ],
    }


@app.post("/generate")
def generate_response():
    state = serving_state()
    body = request.get_json(silent=False)
    logging.info("Request body: %s", body)
    generate_request = GenerateRequest.model_validate(body)
    response_data = generate_response_data(state, generate_request)
    return jsonify(response_data)


def generate_response_data(
    state: ServingState,
    generate_request: GenerateRequest,
) -> dict:
    validate_generate_request(generate_request, state.rooms)
    key = prefetch_request_key(generate_request)
    count = generate_request.max_num_maps
    prefetch = state.prefetch
    stats = {"num_generated": 0, "num_pre_valid": 0}
    if prefetch is not None:
        if count > state.serving_config.prefetch_capacity:
            raise ValueError("max_num_maps must not exceed prefetch_capacity")
        with prefetch.condition:
            queue = touch_prefetch_queue(state, key, generate_request)
            log_queue_state(state, key, f"Generate request max_num_maps={count}")
            if queue.maps:
                maps = take_queue_maps(state, key, min(count, len(queue.maps)))
                log_queue_state(state, key, f"Generate cache hit returned={len(maps)}")
                return combine_response_maps(maps, stats)
            log_queue_state(state, key, "Generate cache miss")
    else:
        log_queue_state(state, key, f"Generate request max_num_maps={count} caching disabled")
    acquire_foreground_generation_lock(state, key)
    profiles = []
    available = []
    try:
        for _ in range(state.serving_config.foreground_max_batches):
            if prefetch is not None:
                with prefetch.condition:
                    if queue.maps:
                        break
            response = generate_batch_with_generation_lock_held(
                state, generate_request, threading.Event()
            )
            for name in stats:
                stats[name] += response["stats"][name]
            if state.profile:
                profiles.extend(response["profile"])
            maps = split_response_maps(response)
            if prefetch is not None:
                with prefetch.condition:
                    append_queue_maps(state, key, maps)
            else:
                available.extend(maps)
                if available:
                    break
        if prefetch is not None:
            with prefetch.condition:
                available = take_queue_maps(state, key, min(count, len(queue.maps)))
        response = combine_response_maps(available[:count], stats)
        if state.profile:
            response["profile"] = profiles
        return response
    finally:
        if prefetch is not None:
            with prefetch.condition:
                log_queue_state(state, key, "Generate request finished")
        else:
            log_queue_state(state, key, "Generate request finished caching disabled")
        release_generation_lock(state)


def request_prefetch_refill(state: ServingState, generate_request: GenerateRequest) -> dict:
    validate_generate_request(generate_request, state.rooms)
    key = prefetch_request_key(generate_request)
    prefetch = state.prefetch
    if prefetch is None:
        log_queue_state(state, key, "Refill request ignored: caching disabled")
        return {"queue": queue_id(key), "funded_batches": 0}
    if generate_request.max_num_maps > state.serving_config.prefetch_capacity:
        raise ValueError("max_num_maps must not exceed prefetch_capacity")
    with prefetch.condition:
        queue = touch_prefetch_queue(state, key, generate_request)
        funded = (
            state.serving_config.prefetch_refill_batches if len(queue.maps) < queue.target else 0
        )
        queue.refill_batches += funded
        log_queue_state(state, key, f"Refill request funded_batches={funded}")
        prefetch.condition.notify_all()
        return {"queue": queue_id(key), "funded_batches": funded}


@app.post("/refill")
def refill_response():
    generate_request = GenerateRequest.model_validate(request.get_json(silent=False))
    return jsonify(request_prefetch_refill(serving_state(), generate_request)), 202


@app.get("/health")
def health_response():
    return jsonify({"status": "ok"})


@app.errorhandler(ValueError)
def value_error_response(error: ValueError):
    return jsonify({"error": str(error)}), 400


@app.errorhandler(ValidationError)
def validation_error_response(error: ValidationError):
    return jsonify({"error": error.errors()}), 400


@app.errorhandler(BadRequest)
def bad_request_response(error: BadRequest):
    return jsonify({"error": error.description}), 400


async def run_http_server(state: ServingState) -> None:
    config = HypercornConfig()
    config.bind = [f"{state.serving_config.host}:{state.serving_config.port}"]
    config.accesslog = logging.getLogger("hypercorn.access")
    config.errorlog = logging.getLogger("hypercorn.error")
    # Queue operations assume only one HTTP request executes at a time.
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))
    try:
        await serve(app, config, mode="wsgi")
    finally:
        # Stop refills before Python waits for their generation executor threads.
        prefetch = state.prefetch
        if prefetch is not None:
            with prefetch.condition:
                for queue in prefetch.queues.values():
                    queue.refill_batches = 0
                if prefetch.active_cancellation is not None:
                    logging.info(
                        "Cancellation triggered by server shutdown queue=%s",
                        queue_id(prefetch.active_key),
                    )
                    prefetch.active_cancellation.set()
            with state.lock:
                pass


def main() -> None:
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
    args = parse_args()
    serving_config = load_serving_config(args.serving_config)
    model_export = load_model_input(args.model_export)
    state = create_serving_state(
        serving_config,
        model_export,
        args.seed,
        args.profile,
    )
    initialize_serving_state(state)
    run_warmup_requests(state)
    asyncio.run(run_http_server(state))


if __name__ == "__main__":
    main()
