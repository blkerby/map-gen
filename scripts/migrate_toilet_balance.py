#!/usr/bin/env python3
"""Migrate v16 conditional Toilet costs to v18 independent balance controllers.

The frozen main and EMA encoders are calibrated independently on replay prefixes.
Only their final Toilet cost heads are fitted. The new controller failure price
starts at zero for every conditioning vector; all existing prices are preserved.
Run from the repository root in the map-gen environment.
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

from env import AREA_COUNT, Actions, Engine, FeatureSlot
from learn import generation_area_balance_targets
from loss import compute_balance_price_tables
from model import FrontierModel
from model_loading import frontier_model_kwargs, without_prefix
from train import (
    create_adam_optimizer,
    create_main_optimizer,
    load_named_optimizer_checkpoint_state,
    named_checkpoint_optimizers,
)
from train_config import Config, instantiate_scheduleable_config
from scripts.balance_v18 import create_balance_model_v18
from scripts.split_balance_controller import (
    shared_balance_model,
    split_shared_balance_state,
)


SOURCE_FORMAT = "map-gen-training-session-checkpoint-v16"
TRAINING_CHECKPOINT_FORMAT = "map-gen-training-session-checkpoint-v18"
HEAD_NAME = "toilet_balance_score_output"


class CaptureHeadInput:
    def __init__(self):
        self.values = []

    def __call__(self, module, args):
        self.values.append(args[0].detach().reshape(-1, module.in_features).clone())


def append_failure_output(tensors, balance_model):
    """Append an affine copy of the feasible-success mean, giving centered f=0."""
    layer_name = f"net.{len(balance_model.door_net) - 1}"
    area_width = balance_model.num_room_connection_variants * AREA_COUNT
    old_width = (
        balance_model.door_net[-1].out_features + balance_model.num_rooms + area_width
    )
    toilet_start = old_width - area_width - balance_model.num_rooms
    compatible = balance_model.toilet_compatibility
    if not compatible.any():
        raise ValueError("migration requires a room set with feasible Toilet crossings")
    for suffix in ("weight", "bias"):
        key = f"balance_model.{layer_name}.{suffix}"
        old = tensors[key]
        if old.shape[0] != old_width:
            raise ValueError(f"unexpected v16 controller shape: {key} {old.shape}")
        new_row = old[toilet_start:toilet_start + balance_model.num_rooms][compatible].mean(0)
        tensors[key] = torch.cat((old, new_row.unsqueeze(0))).contiguous()
    return {f"{layer_name}.weight", f"{layer_name}.bias"}


def migrate_optimizer_state(tensors, metadata, model, optimizer, prefix, reset_names, extend_names):
    """Map saved parameter IDs through the actual optimizer's parameter ordering."""
    parameter_names = {id(param): name for name, param in model.named_parameters()}
    parts = named_checkpoint_optimizers(optimizer)
    if set(json.loads(metadata[f"{prefix}_names"])) != set(parts):
        raise ValueError(f"unexpected optimizer parts for {prefix}")
    changed = set()
    for part_name, part in parts.items():
        groups = json.loads(metadata[f"{prefix}_{part_name}_param_groups"])
        scalar_key = f"{prefix}_{part_name}_scalar_state"
        scalar_state = json.loads(metadata[scalar_key])
        for saved_group, current_group in zip(groups, part.param_groups, strict=True):
            for param_id, param in zip(saved_group["params"], current_group["params"], strict=True):
                name = parameter_names[id(param)]
                if name not in reset_names | extend_names:
                    continue
                state_prefix = f"{prefix}.{part_name}.state.{param_id}."
                keys = [key for key in tensors if key.startswith(state_prefix)]
                if not keys:
                    raise ValueError(f"missing optimizer state for {name}")
                for key in keys:
                    old = tensors[key]
                    if name in reset_names:
                        tensors[key] = torch.zeros_like(old)
                    elif old.ndim:
                        if old.shape[1:] != param.shape[1:] or old.shape[0] >= param.shape[0]:
                            raise ValueError(f"unexpected optimizer state shape: {key}")
                        padding = old.new_zeros((param.shape[0] - old.shape[0], *old.shape[1:]))
                        tensors[key] = torch.cat((old, padding)).contiguous()
                if str(param_id) in scalar_state and name in reset_names:
                    scalar_state[str(param_id)] = {
                        key: type(value)(0) for key, value in scalar_state[str(param_id)].items()
                    }
                changed.add(name)
        metadata[scalar_key] = json.dumps(scalar_state)
    if changed != reset_names | extend_names:
        raise ValueError(f"could not migrate optimizer state for {(reset_names | extend_names) - changed}")
    load_named_optimizer_checkpoint_state(optimizer, tensors, metadata, prefix)


@torch.no_grad()
def fit_head(inputs, targets, old_head, ridge):
    """Ridge fit in standardized coordinates, regularized toward a scalar rescale."""
    x = inputs.double()
    y = targets.double()
    mean = x.mean(0)
    scale = x.std(0).clamp_min(1e-8)
    design = torch.cat(((x - mean) / scale, torch.ones((len(x), 1))), dim=1)
    conditional = old_head(inputs).flatten().double()
    rescale = (conditional @ y) / (conditional @ conditional).clamp_min(1e-20)
    prior_weight = old_head.weight.flatten().double() * rescale
    prior_bias = old_head.bias.flatten().double() * rescale
    prior = torch.cat((prior_weight * scale, prior_bias + prior_weight @ mean))
    gram = design.T @ design / len(x)
    coefficients = prior + torch.linalg.solve(
        gram + ridge * torch.eye(gram.shape[0], dtype=gram.dtype),
        design.T @ (y - design @ prior) / len(x),
    )
    weight = coefficients[:-1] / scale
    bias = coefficients[-1] - weight @ mean
    return weight.float().unsqueeze(0), bias.float().reshape(1), float(rescale)


def fit_metrics(predicted, target):
    error = predicted - target
    return {
        "states": target.numel(),
        "target_rms": float(target.square().mean().sqrt()),
        "rmse": float(error.square().mean().sqrt()),
        "mean_error": float(error.mean()),
        "absolute_error_p95": float(error.abs().quantile(0.95)),
        "max_absolute_error": float(error.abs().max()),
    }


@torch.no_grad()
def calibrate_heads(args, config, engine, balance_model, models):
    count = args.calibration_maps + args.validation_maps
    with safe_open(args.experience, framework="pt") as experience:
        available = experience.get_slice("room_idx").get_shape()[0]
        if count > available:
            raise ValueError(f"requested {count} maps, experience contains {available}")
        indices = torch.randperm(available, generator=torch.Generator().manual_seed(args.seed))[:count]
        variables = experience.get_tensor("generation_variable_floats")[indices]
        actions = {key: experience.get_tensor(key)[indices] for key in ("room_idx", "room_x", "room_y", "room_area")}
        log_temperature = experience.get_tensor("temperature")[indices].log()
        log_candidates = (experience.get_tensor("recommended_candidates")[indices] + 1).log()
    episode_length = actions["room_idx"].shape[1]
    if min(args.steps) < 1 or max(args.steps) > episode_length:
        raise ValueError("calibration steps must be within the recorded episode")
    env = engine.create_environment_group(
        map_size=config.map_size,
        num_envs=count,
        candidate_spatial_cell_size=config.generation.candidate_spatial_cell_size,
        area_bounding_box_width=config.generation.area_bounding_box_width,
        area_bounding_box_height=config.generation.area_bounding_box_height,
        seed=args.seed,
        frontier_neighbor_count=config.generation.frontier_neighbor_count,
        frontier_window_size=config.generation.frontier_window_size,
        num_threads=args.threads,
        frontier_neighbor_algorithm=config.generation.frontier_neighbor_algorithm,
    )
    slot = FeatureSlot(env, pin_memory=False)
    captures = {name: CaptureHeadInput() for name in models}
    handles = {name: getattr(model, HEAD_NAME).register_forward_pre_hook(captures[name]) for name, model in models.items()}
    samples = {name: {key: [] for key in ("input", "target", "train")} for name in models}
    crossing_prices = []
    for start in range(0, count, args.batch_size):
        batch_variables = variables[start:start + args.batch_size]
        area = generation_area_balance_targets(engine.rooms, batch_variables)
        tables = compute_balance_price_tables(balance_model(batch_variables), area.probability, area.dual_mask)
        torch.testing.assert_close(tables.toilet_failure, torch.zeros_like(tables.toilet_failure), atol=1e-6, rtol=0)
        crossing_prices.append(tables.toilet_crossed_room)
    crossing_prices = torch.cat(crossing_prices)
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
            # Known outcomes are overridden exactly during generation in both formats.
            unknown = outcomes.toilet_invalid.flatten() < 0
            crossed = features.global_features.toilet_crossed_room_idx.flatten().long()
            exact = crossing_prices[start:end].gather(1, crossed.clamp_min(0).unsqueeze(1)).flatten()
            for name, model in models.items():
                prediction = model(features, return_proposal_state=False)
                conditional = torch.where(crossed >= 0, exact, prediction.toilet_balance_score.flatten())
                old_cost = conditional * (-prediction.toilet_invalid.flatten()).sigmoid()
                samples[name]["input"].append(captures[name].values.pop()[unknown])
                samples[name]["target"].append(old_cost[unknown].clone())
                samples[name]["train"].append((torch.arange(start, end) < args.calibration_maps)[unknown])
        print(f"Collected calibration prefixes at step {step + 1}", flush=True)
    report = {}
    for name, model in models.items():
        handles[name].remove()
        data = {key: torch.cat(values) for key, values in samples[name].items()}
        x, y, train = data["input"], data["target"], data["train"]
        if train.sum() <= x.shape[1] or (~train).sum() == 0:
            raise ValueError("insufficient unknown-outcome states for calibration and validation")
        head = getattr(model, HEAD_NAME)
        old = head(x).flatten()
        weight, bias, rescale = fit_head(x[train], y[train], head, args.ridge)
        head.weight.copy_(weight)
        head.bias.copy_(bias)
        predicted = head(x).flatten()
        report[name] = {
            "scalar_rescale": rescale,
            "training": fit_metrics(predicted[train], y[train]),
            "validation": fit_metrics(predicted[~train], y[~train]),
            "validation_scalar_rescale": fit_metrics(old[~train] * rescale, y[~train]),
            "validation_unmigrated_head": fit_metrics(old[~train], y[~train]),
        }
        if not torch.isfinite(predicted).all():
            raise ValueError(f"non-finite migrated predictions in {name}")
        if report[name]["validation"]["rmse"] >= report[name]["validation_unmigrated_head"]["rmse"]:
            raise ValueError(f"calibration did not improve held-out error for {name}")
    return {"models": report, "episode_indices": indices.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--experience", type=Path, required=True)
    parser.add_argument("--calibration-maps", type=int, required=True)
    parser.add_argument("--validation-maps", type=int, required=True)
    parser.add_argument("--steps", nargs="+", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ridge", type=float, required=True)
    args = parser.parse_args()
    if min(args.calibration_maps, args.validation_maps, args.batch_size, args.threads, args.ridge) <= 0:
        raise ValueError("sample counts, batch size, threads and ridge must be positive")
    report_path = args.output.with_suffix(".migration.json")
    temp_path = args.output.with_suffix(".safetensors.tmp")
    for path in (args.output, report_path, temp_path):
        if path.exists():
            raise FileExistsError(path)
    torch.set_num_threads(args.threads)
    with safe_open(args.checkpoint, framework="pt") as checkpoint:
        metadata = checkpoint.metadata()
        if metadata is None or metadata["format"] != SOURCE_FORMAT:
            raise ValueError("migration requires a v16 training checkpoint")
        tensors = {key: checkpoint.get_tensor(key) for key in checkpoint.keys()}
    config = instantiate_scheduleable_config(Config.model_validate_json(metadata["config"]), int(metadata["num_episodes"]))
    rooms = json.loads(config.room_set.read_text())
    engine = Engine(rooms, config.features, config.generation.min_area_size, config.generation.max_area_size)
    balance_model = create_balance_model_v18(config, rooms, engine, torch.device("cpu"))
    extended_names = append_failure_output(tensors, balance_model)
    shared = shared_balance_model(balance_model)
    migrate_optimizer_state(
        tensors, metadata, shared,
        create_adam_optimizer(shared.parameters(), config.balance_optimizer),
        "balance_optimizer", set(), extended_names,
    )
    split_shared_balance_state(tensors, metadata, balance_model)
    balance_model.eval()
    models = {}
    for name in ("main_model", "ema_model"):
        model = FrontierModel(**frontier_model_kwargs(config, rooms, engine))
        model.load_state_dict(without_prefix(tensors, name))
        model.eval()
        models[name] = model
    report = calibrate_heads(args, config, engine, balance_model, models)
    for name, model in models.items():
        for suffix in ("weight", "bias"):
            tensors[f"{name}.{HEAD_NAME}.{suffix}"] = getattr(getattr(model, HEAD_NAME), suffix).detach().contiguous()
    migrate_optimizer_state(
        tensors, metadata, models["main_model"],
        create_main_optimizer(models["main_model"], config.optimizer), "optimizer",
        {f"{HEAD_NAME}.weight", f"{HEAD_NAME}.bias"}, set(),
    )
    load_named_optimizer_checkpoint_state(
        create_adam_optimizer(balance_model.parameters(), config.balance_optimizer),
        tensors, metadata, "balance_optimizer",
    )
    metadata["format"] = TRAINING_CHECKPOINT_FORMAT
    report.update({
        "source": str(args.checkpoint), "output": str(args.output),
        "experience": str(args.experience), "source_format": SOURCE_FORMAT,
        "output_format": TRAINING_CHECKPOINT_FORMAT,
        "calibration_maps": args.calibration_maps, "validation_maps": args.validation_maps,
        "steps": args.steps, "seed": args.seed, "ridge": args.ridge,
        "initial_failure_price": 0.0,
        "optimizer_changes": (
            "Reset main Toilet cost head state; append zero failure moments; "
            "copy shared hidden state and split output state into independent branches."
        ),
    })
    metadata["toilet_balance_migration"] = json.dumps(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, temp_path, metadata=metadata)
    os.replace(temp_path, args.output)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["models"], indent=2))
    print(f"Saved {args.output} and {report_path}")


if __name__ == "__main__":
    main()
