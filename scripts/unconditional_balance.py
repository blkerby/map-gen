#!/usr/bin/env python3
"""Migrate v18 to v19 unconditional door costs and door/area failure outcomes.

Only final cost layers are calibrated, independently for the main and EMA models.
Controller success prices and the existing Toilet failure price are preserved.
New failure prices start at zero. Calibration is approximate and requires saved
experience; held-out maps check the costs actually used during generation.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch.nn.functional import linear

from env import Actions, Engine, FeatureSlot
from learn import generation_area_balance_targets
from loss import compute_balance_price_tables
from model import FrontierModel
from model_loading import create_balance_model, frontier_model_kwargs, without_prefix
from scripts.balance_v18 import create_balance_model_v18
from scripts.migrate_toilet_balance import CaptureHeadInput, migrate_optimizer_state
from serve import TRAINING_CHECKPOINT_FORMAT, validate_model_input_metadata
from train import create_adam_optimizer, create_main_optimizer
from train_config import Config, instantiate_scheduleable_config


SOURCE_FORMAT = "map-gen-training-session-checkpoint-v18"
HEADS = {
    "balance_score_output": "door_output",
    "frontier_balance_score_output": "frontier_door_invalid_output",
}


def migrate_controller(tensors, model):
    """Extend only final layers, and express Toilet failure relative to success."""
    extended = set()
    for branch in ("door_net", "area_net"):
        layer = f"{branch}.{len(getattr(model, branch)) - 1}"
        for suffix in ("weight", "bias"):
            name = f"{layer}.{suffix}"
            old = tensors[f"balance_model.{name}"]
            target = model.state_dict()[name]
            if old.shape[1:] != target.shape[1:] or old.shape[0] >= target.shape[0]:
                raise ValueError(f"unexpected v18 controller shape for {name}")
            padding = old.new_zeros((target.shape[0] - old.shape[0], *old.shape[1:]))
            tensors[f"balance_model.{name}"] = torch.cat((old, padding)).contiguous()
            extended.add(name)
    reset = set()
    layer = f"toilet_net.{len(model.toilet_net) - 1}"
    for suffix in ("weight", "bias"):
        name = f"{layer}.{suffix}"
        old = tensors[f"balance_model.{name}"]
        mean = (old[:model.num_rooms] * model.toilet_compatibility.reshape(
            model.num_rooms, *([1] * (old.ndim - 1)),
        )).sum(0) / model.toilet_compatibility.sum().clamp_min(1)
        tensors[f"balance_model.{name}"] = torch.cat((old[:-1], (old[-1] - mean).unsqueeze(0))).contiguous()
        reset.add(name)
    model.load_state_dict(without_prefix(tensors, "balance_model"))
    return reset, extended


@torch.no_grad()
def verify_controller(old, new, variables, engine):
    area = generation_area_balance_targets(engine.rooms, variables)
    before = compute_balance_price_tables(old(variables), area.probability, area.dual_mask)
    after = compute_balance_price_tables(new(variables), area.probability, area.dual_mask)
    for name in ("left", "right", "up", "down", "toilet_crossed_room", "toilet_failure", "room_area"):
        torch.testing.assert_close(getattr(before, name), getattr(after, name), atol=1e-6, rtol=1e-5)
    for name in ("door_failure", "room_area_failure"):
        assert torch.count_nonzero(getattr(after, name)) == 0


@torch.no_grad()
def fit_cost_head(inputs, targets, head, ridge):
    """Multi-output ridge fit with a per-output scalar-rescaling prior."""
    x, y = inputs.double(), targets.double()
    mean, scale = x.mean(0), x.std(0).clamp_min(1e-8)
    design = torch.cat(((x - mean) / scale, x.new_ones((len(x), 1))), dim=1)
    conditional = linear(inputs, head.weight, head.bias).double()
    rescale = (conditional * y).sum(0) / conditional.square().sum(0).clamp_min(1e-20)
    prior_weight = head.weight.double() * rescale.unsqueeze(1)
    prior_bias = head.bias.double() * rescale
    prior = torch.cat((prior_weight * scale, (prior_bias + prior_weight @ mean).unsqueeze(1)), dim=1).T
    gram = design.T @ design / len(x)
    coefficients = prior + torch.linalg.solve(
        gram + ridge * torch.eye(gram.shape[0], dtype=gram.dtype),
        design.T @ (y - design @ prior) / len(x),
    )
    weight = coefficients[:-1].T / scale
    bias = coefficients[-1] - weight @ mean
    return weight.float().contiguous(), bias.float(), rescale.float()


def weighted_metrics(predicted, target, weight):
    count = weight.sum()
    if count <= 0:
        raise ValueError("no unknown door outcomes in validation data")
    error = predicted - target
    return {
        "door_states": int(count),
        "target_rms": float((target.square() * weight).sum().div(count).sqrt()),
        "rmse": float((error.square() * weight).sum().div(count).sqrt()),
        "mean_error": float((error * weight).sum() / count),
        "max_absolute_error": float(error[weight > 0].abs().max()),
    }


def door_usage(features, outcomes, model):
    """Count global variant uses; frontier overrides and known outcomes are excluded."""
    unknown = outcomes.door_match < 0
    rows = features.frontier_features.row_snapshot_idx.long()
    doors = features.frontier_features.row_door_output_idx.long()
    valid = (rows >= 0) & (rows < len(unknown)) & (doors >= 0) & (doors < unknown.shape[1])
    local = torch.zeros(len(rows), dtype=torch.bool)
    local[valid] = unknown[rows[valid], doors[valid]]
    remaining = unknown.clone()
    remaining[rows[valid], doors[valid]] = False
    weight = torch.zeros((len(unknown), model.balance_score_output.out_features))
    weight.scatter_add_(1, model.door_variant_outcome_idx.expand(len(unknown), -1), remaining.float())
    return weight, local


@torch.no_grad()
def calibrate_heads(args, config, engine, models, old_controller, new_controller):
    count = args.calibration_maps + args.validation_maps
    generator = torch.Generator().manual_seed(args.seed)
    with safe_open(args.experience, framework="pt") as experience:
        available = experience.get_slice("room_idx").get_shape()[0]
        if count > available:
            raise ValueError(f"requested {count} maps, experience contains {available}")
        indices = torch.randperm(available, generator=generator)[:count]
        variables = experience.get_tensor("generation_variable_floats")[indices]
        actions = {key: experience.get_tensor(key)[indices] for key in ("room_idx", "room_x", "room_y", "room_area")}
        log_temperature = experience.get_tensor("temperature")[indices].log()
        log_candidates = (experience.get_tensor("recommended_candidates")[indices] + 1).log()
    if min(args.steps) < 1 or max(args.steps) > actions["room_idx"].shape[1]:
        raise ValueError("calibration steps must be within the recorded episode")
    verify_controller(old_controller, new_controller, variables, engine)
    env = engine.create_environment_group(
        map_size=config.map_size, num_envs=count,
        candidate_spatial_cell_size=config.generation.candidate_spatial_cell_size,
        area_bounding_box_width=config.generation.area_bounding_box_width,
        area_bounding_box_height=config.generation.area_bounding_box_height,
        seed=args.seed, frontier_neighbor_count=config.generation.frontier_neighbor_count,
        frontier_window_size=config.generation.frontier_window_size, num_threads=args.threads,
        frontier_neighbor_algorithm=config.generation.frontier_neighbor_algorithm,
    )
    slot = FeatureSlot(env, pin_memory=False)
    captures, handles, samples = {}, [], {}
    for name, model in models.items():
        for head_name in HEADS:
            key = (name, head_name)
            captures[key] = CaptureHeadInput()
            handles.append(getattr(model, head_name).register_forward_pre_hook(captures[key]))
            samples[key] = {field: [] for field in ("input", "target", "train", "weight")}
    try:
        for step in range(max(args.steps)):
            env.step(Actions(**{key: value[:, step] for key, value in actions.items()}))
            if step + 1 not in args.steps:
                continue
            for start in range(0, count, args.batch_size):
                end = min(start + args.batch_size, count)
                outcomes = env.get_current_feature_outcomes(torch.device("cpu"), start, end - start)
                features = env.extract_features(
                    slot, log_temperature[start:end], config.features.temperature,
                    log_candidates[start:end], config.features.recommended_candidates,
                    variables[start:end], True, outcomes, config.features.lookahead_outcomes,
                    start, end - start,
                )
                usage, local = door_usage(features, outcomes, models["main_model"])
                local_idx = local.nonzero().flatten()
                if len(local_idx) > args.frontier_samples_per_batch:
                    local_idx = local_idx[torch.randperm(len(local_idx), generator=generator)[:args.frontier_samples_per_batch]]
                rows = features.frontier_features.row_snapshot_idx.long()[local_idx]
                for name, model in models.items():
                    model(features, return_proposal_state=False)
                    for head_name, validity_name in HEADS.items():
                        key = (name, head_name)
                        x = captures[key].values.pop()
                        if head_name == "frontier_balance_score_output":
                            x = x[local_idx]
                            train = start + rows < args.calibration_maps
                            weight = torch.ones((len(x), 1))
                        else:
                            train = torch.arange(start, end) < args.calibration_maps
                            weight = usage
                        head, validity = getattr(model, head_name), getattr(model, validity_name)
                        target = linear(x, head.weight, head.bias) * (-linear(x, validity.weight, validity.bias)).sigmoid()
                        for field, value in (("input", x), ("target", target), ("train", train), ("weight", weight)):
                            samples[key][field].append(value.clone())
            print(f"Collected calibration prefixes at step {step + 1}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    report = {}
    for (name, head_name), sample in samples.items():
        data = {key: torch.cat(values) for key, values in sample.items()}
        x, y, train, usage = (data[key] for key in ("input", "target", "train", "weight"))
        if train.sum() <= x.shape[1] or (~train).sum() == 0:
            raise ValueError(f"insufficient calibration/validation states for {name}.{head_name}")
        head = getattr(models[name], head_name)
        old = head(x)
        weight, bias, rescale = fit_cost_head(x[train], y[train], head, args.ridge)
        head.weight.copy_(weight)
        head.bias.copy_(bias)
        predicted = head(x)
        result = {
            "scalar_rescale": rescale.tolist(),
            "training": weighted_metrics(predicted[train], y[train], usage[train]),
            "validation": weighted_metrics(predicted[~train], y[~train], usage[~train]),
            "validation_scalar_rescale": weighted_metrics(old[~train] * rescale, y[~train], usage[~train]),
            "validation_unmigrated_head": weighted_metrics(old[~train], y[~train], usage[~train]),
        }
        if not torch.isfinite(predicted).all():
            raise ValueError(f"non-finite predictions for {name}.{head_name}")
        if result["validation"]["rmse"] > result["validation_unmigrated_head"]["rmse"]:
            raise ValueError(f"calibration worsened held-out costs for {name}.{head_name}: {result}")
        report[f"{name}.{head_name}"] = result
    return {"heads": report, "episode_indices": indices.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--experience", type=Path, required=True)
    for name in ("calibration-maps", "validation-maps", "batch-size", "threads", "seed", "frontier-samples-per-batch"):
        parser.add_argument(f"--{name}", type=int, required=True)
    parser.add_argument("--steps", nargs="+", type=int, required=True)
    parser.add_argument("--ridge", type=float, required=True)
    args = parser.parse_args()
    if min(args.calibration_maps, args.validation_maps, args.batch_size, args.threads, args.frontier_samples_per_batch, args.ridge) <= 0:
        raise ValueError("sample counts, threads and ridge must be positive")
    report_path, temporary = args.output.with_suffix(".migration.json"), args.output.with_suffix(".safetensors.tmp")
    for path in (args.output, report_path, temporary):
        if path.exists():
            raise FileExistsError(path)
    torch.set_num_threads(args.threads)
    with safe_open(args.checkpoint, framework="pt") as checkpoint:
        metadata = checkpoint.metadata()
        if metadata is None or metadata["format"] != SOURCE_FORMAT:
            raise ValueError("migration requires a v18 training checkpoint")
        tensors = {key: checkpoint.get_tensor(key) for key in checkpoint.keys()}
    config = instantiate_scheduleable_config(Config.model_validate_json(metadata["config"]), int(metadata["num_episodes"]))
    rooms = json.loads(config.room_set.read_text())
    engine = Engine(rooms, config.features, config.generation.min_area_size, config.generation.max_area_size)
    old_controller = create_balance_model_v18(config, rooms, engine, torch.device("cpu")).eval()
    old_controller.load_state_dict(without_prefix(tensors, "balance_model"))
    controller = create_balance_model(config, rooms, engine, torch.device("cpu")).eval()
    reset, extended = migrate_controller(tensors, controller)
    migrate_optimizer_state(
        tensors, metadata, controller, create_adam_optimizer(controller.parameters(), config.balance_optimizer),
        "balance_optimizer", reset, extended,
    )
    models = {}
    for name in ("main_model", "ema_model"):
        model = FrontierModel(**frontier_model_kwargs(config, rooms, engine))
        model.load_state_dict(without_prefix(tensors, name))
        models[name] = model.eval()
    report = calibrate_heads(args, config, engine, models, old_controller, controller)
    for name, model in models.items():
        for head_name in HEADS:
            for suffix in ("weight", "bias"):
                tensors[f"{name}.{head_name}.{suffix}"] = getattr(getattr(model, head_name), suffix).detach().contiguous()
    migrate_optimizer_state(
        tensors, metadata, models["main_model"], create_main_optimizer(models["main_model"], config.optimizer),
        "optimizer", {f"{name}.{suffix}" for name in HEADS for suffix in ("weight", "bias")}, set(),
    )
    metadata["format"] = TRAINING_CHECKPOINT_FORMAT
    validate_model_input_metadata(args.output, metadata)
    report.update({
        "source": str(args.checkpoint), "output": str(args.output), "experience": str(args.experience),
        "source_format": SOURCE_FORMAT, "output_format": TRAINING_CHECKPOINT_FORMAT,
        "calibration_maps": args.calibration_maps, "validation_maps": args.validation_maps,
        "batch_size": args.batch_size, "threads": args.threads,
        "steps": args.steps, "seed": args.seed, "ridge": args.ridge,
        "frontier_samples_per_batch": args.frontier_samples_per_batch,
        "initial_door_area_failure_price": 0.0,
        "optimizer_changes": (
            "Reset main global/frontier door cost final layers; append zero door/area failure moments. "
            "Reset Toilet controller final-layer state because its coordinate change has no exact Adam "
            "second-moment mapping. Preserve all hidden-layer weights and optimizer states."
        ),
    })
    metadata["unconditional_balance_migration"] = json.dumps(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, temporary, metadata=metadata)
    os.replace(temporary, args.output)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    for name, result in report["heads"].items():
        print(name, json.dumps({key: value for key, value in result.items() if key != "scalar_rescale"}))
    print(f"Saved {args.output} and {report_path}")


if __name__ == "__main__":
    main()
