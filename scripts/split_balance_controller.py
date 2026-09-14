#!/usr/bin/env python3
"""Migrate v17 shared balance controllers to independent v18 price networks.

Copy hidden parameters and Adam state into each branch, and partition output
rows and their optimizer state. Main/EMA models and their optimizers are unchanged.
Run from the repository root in the map-gen environment.
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from env import Engine
from model import BalanceModel, balance_price_network
from model_loading import without_prefix
from train import (
    create_adam_optimizer,
    create_generate_config,
    load_named_optimizer_checkpoint_state,
)
from train_config import Config, instantiate_scheduleable_config
from scripts.balance_v18 import create_balance_model_v18

TRAINING_CHECKPOINT_FORMAT = "map-gen-training-session-checkpoint-v18"


SOURCE_FORMAT = "map-gen-training-session-checkpoint-v17"
BRANCHES = ("door_net", "toilet_net", "area_net")


def shared_balance_model(model: BalanceModel) -> torch.nn.Module:
    """Reconstruct only the legacy parameter layout, for explicit migration."""
    shared = torch.nn.Module()
    shared.add_module(
        "net",
        balance_price_network(
            hidden_width=model.door_net[0].out_features,
            num_layers=(len(model.door_net) - 1) // 2,
            output_width=sum(getattr(model, branch)[-1].out_features for branch in BRANCHES),
        ),
    )
    return shared


def branch_output_rows(model: BalanceModel) -> dict[str, torch.Tensor]:
    door_end = model.door_net[-1].out_features
    toilet_end = door_end + model.num_rooms
    area_end = toilet_end + model.area_net[-1].out_features
    # v17 appends the failure output after all room-area outputs.
    return {
        "door_net": torch.arange(door_end),
        "toilet_net": torch.cat((torch.arange(door_end, toilet_end), torch.tensor([area_end]))),
        "area_net": torch.arange(toilet_end, area_end),
    }


def split_shared_balance_state(
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, str],
    model: BalanceModel,
) -> None:
    """Transform a validated v17 layout in place, with independent tensor storage."""
    if json.loads(metadata["balance_optimizer_names"]) != ["adam"]:
        raise ValueError("balance controller migration requires its Adam optimizer")
    groups = json.loads(metadata["balance_optimizer_adam_param_groups"])
    if len(groups) != 1:
        raise ValueError("expected one balance optimizer parameter group")
    scalar_state = json.loads(metadata["balance_optimizer_adam_scalar_state"])
    shared = shared_balance_model(model)
    legacy_state = without_prefix(tensors, "balance_model")
    shared.load_state_dict({key: value for key, value in legacy_state.items() if key.startswith("net.")})
    buffers = dict(model.named_buffers())
    expected_keys = set(shared.state_dict()) | (set(model.state_dict()) & set(buffers))
    if set(legacy_state) != expected_keys:
        raise ValueError("checkpoint balance tensors do not match the v17 layout")
    source_ids = dict(zip(dict(shared.named_parameters()), groups[0]["params"], strict=True))
    last_layer = len(model.door_net) - 1
    output_rows = branch_output_rows(model)
    new_tensors = {}
    new_scalar_state = {}
    for new_id, (name, parameter) in enumerate(model.named_parameters()):
        branch, suffix = name.split(".", 1)
        source_name = f"net.{suffix}"
        source = legacy_state[source_name]
        rows = (
            output_rows[branch]
            if suffix.startswith(f"{last_layer}.")
            else torch.arange(source.shape[0])
        )
        value = source.index_select(0, rows)
        if value.shape != parameter.shape:
            raise ValueError(f"migrated parameter shape mismatch: {name}")
        new_tensors[f"balance_model.{name}"] = value
        source_id = source_ids[source_name]
        prefix = f"balance_optimizer.adam.state.{source_id}."
        state = {key[len(prefix):]: value for key, value in tensors.items() if key.startswith(prefix)}
        scalars = scalar_state.get(str(source_id), {})
        if not {"step", "exp_avg", "exp_avg_sq"} <= state.keys() | scalars.keys():
            raise ValueError(f"incomplete Adam state for {source_name}")
        for state_name, value in state.items():
            if value.ndim and value.shape != source.shape:
                raise ValueError(f"unexpected Adam state shape for {source_name}.{state_name}")
            new_tensors[f"balance_optimizer.adam.state.{new_id}.{state_name}"] = (
                value.index_select(0, rows) if value.ndim else value.clone()
            )
        if scalars:
            new_scalar_state[str(new_id)] = copy.deepcopy(scalars)
    for key in list(tensors):
        if key.startswith(("balance_model.net.", "balance_optimizer.adam.state.")):
            del tensors[key]
    tensors.update(new_tensors)
    groups[0]["params"] = list(range(len(list(model.parameters()))))
    metadata["balance_optimizer_adam_param_groups"] = json.dumps(groups)
    metadata["balance_optimizer_adam_scalar_state"] = json.dumps(new_scalar_state)
    model.load_state_dict(without_prefix(tensors, "balance_model"))


@torch.no_grad()
def verify_split_outputs(
    shared: torch.nn.Module,
    model: BalanceModel,
    variables: torch.Tensor,
) -> dict[str, float]:
    raw = shared.net(variables)
    errors = {}
    for branch, rows in branch_output_rows(model).items():
        expected = raw.index_select(1, rows)
        actual = getattr(model, branch)(variables)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        errors[branch] = float((actual - expected).abs().max()) if actual.numel() else 0.0
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threads", type=int, required=True)
    args = parser.parse_args()
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    report_path = args.output.with_suffix(".migration.json")
    temp_path = args.output.with_suffix(".safetensors.tmp")
    for path in (args.output, report_path, temp_path):
        if path.exists():
            raise FileExistsError(path)
    torch.set_num_threads(args.threads)
    with safe_open(args.checkpoint, framework="pt") as checkpoint:
        metadata = checkpoint.metadata()
        if metadata is None or metadata["format"] != SOURCE_FORMAT:
            raise ValueError("migration requires a v17 training checkpoint")
        tensors = {key: checkpoint.get_tensor(key) for key in checkpoint.keys()}
    config = instantiate_scheduleable_config(
        Config.model_validate_json(metadata["config"]), int(metadata["num_episodes"])
    )
    rooms = json.loads(config.room_set.read_text())
    engine = Engine(rooms, config.features, config.generation.min_area_size, config.generation.max_area_size)
    model = create_balance_model_v18(config, rooms, engine, torch.device("cpu"))
    shared = shared_balance_model(model)
    shared.load_state_dict({
        key: value for key, value in without_prefix(tensors, "balance_model").items()
        if key.startswith("net.")
    })
    split_shared_balance_state(tensors, metadata, model)
    optimizer = create_adam_optimizer(model.parameters(), config.balance_optimizer)
    load_named_optimizer_checkpoint_state(optimizer, tensors, metadata, "balance_optimizer")
    torch.manual_seed(18)
    generation = create_generate_config(config, rooms, len(rooms), 32, torch.device("cpu"), False)
    errors = verify_split_outputs(shared, model, generation.generation_variable_floats)
    report = {
        "source": str(args.checkpoint), "output": str(args.output),
        "source_format": SOURCE_FORMAT, "output_format": TRAINING_CHECKPOINT_FORMAT,
        "maximum_absolute_output_error": errors,
        "hidden_parameters": "Independent copies of existing shared hidden layers in each branch.",
        "optimizer_state": "Hidden Adam state copied; output state partitioned; step counts preserved.",
        "main_and_ema": "Unchanged, including optimizer state.",
    }
    metadata["format"] = TRAINING_CHECKPOINT_FORMAT
    metadata["balance_branch_migration"] = json.dumps(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, temp_path, metadata=metadata)
    os.replace(temp_path, args.output)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
