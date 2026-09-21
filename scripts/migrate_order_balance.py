#!/usr/bin/env python3
"""Upgrade a v19 checkpoint with a zero-output area-start-order controller.

Example (from the repository root):
  conda run -n map-gen python scripts/migrate_order_balance.py SOURCE OUTPUT \
      --order-beta 1 --order-balance-weight 1 --seed 0

Existing weights, optimizer moments, counters, and run identity are preserved.
New parameters have no optimizer history, just as in a freshly created optimizer.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from env import Engine
from model import FrontierModel
from model_loading import create_balance_model, frontier_model_kwargs, without_prefix
from train import (
    TRAINING_CHECKPOINT_FORMAT,
    create_adam_optimizer,
    create_main_optimizer,
    load_named_optimizer_checkpoint_state,
    named_checkpoint_optimizers,
)
from train_config import Config, instantiate_scheduleable_config, validate_config

SOURCE_FORMAT = "map-gen-training-session-checkpoint-v19"


def add_model_parameters(tensors, model, prefix: str, added_prefix: str) -> set[str]:
    expected = model.state_dict()
    old = without_prefix(tensors, prefix)
    added = {name for name in expected if name.startswith(added_prefix)}
    if set(old) != set(expected) - added:
        raise ValueError(
            f"unexpected {prefix} v19 keys: missing={set(expected) - added - set(old)}, extra={set(old) - set(expected) | (set(old) & added)}"
        )
    for name in added:
        old[name] = expected[name].detach().cpu().contiguous().clone()
        tensors[f"{prefix}.{name}"] = old[name]
    model.load_state_dict(old, strict=True)
    return added


def extend_optimizer_groups(
    tensors, metadata, model, optimizer, prefix: str, added: set[str]
) -> None:
    """Keep every old state ID; insert fresh IDs in the new parameter ordering."""
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    parts = named_checkpoint_optimizers(optimizer)
    if set(json.loads(metadata[f"{prefix}_names"])) != set(parts):
        raise ValueError(f"unexpected optimizer parts for {prefix}")
    for part_name, part in parts.items():
        key = f"{prefix}_{part_name}_param_groups"
        saved_groups = json.loads(metadata[key])
        next_id = max((i for group in saved_groups for i in group["params"]), default=-1) + 1
        for saved, current in zip(saved_groups, part.param_groups, strict=True):
            old_parameters = [p for p in current["params"] if names[id(p)] not in added]
            if len(old_parameters) != len(saved["params"]):
                raise ValueError(f"unexpected parameter count for {prefix}.{part_name}")
            old_ids = iter(saved["params"])
            ids = []
            for parameter in current["params"]:
                if names[id(parameter)] in added:
                    ids.append(next_id)
                    next_id += 1
                else:
                    ids.append(next(old_ids))
            saved["params"] = ids
        metadata[key] = json.dumps(saved_groups)
    load_named_optimizer_checkpoint_state(optimizer, tensors, metadata, prefix)
    # Loading accepts differently shaped moments; check them explicitly.
    for part in parts.values():
        for parameter, state in part.state.items():
            for name, value in state.items():
                if torch.is_tensor(value) and value.ndim and value.shape != parameter.shape:
                    raise ValueError(
                        f"optimizer moment {name} has shape {value.shape}, expected {parameter.shape}"
                    )


def migrate_checkpoint(
    source: Path, output: Path, order_beta: float, order_balance_weight: float, seed: int
) -> dict:
    if output.exists():
        raise FileExistsError(output)
    with safe_open(source, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata()
        if metadata is None or metadata["format"] != SOURCE_FORMAT:
            raise ValueError(f"expected a {SOURCE_FORMAT} checkpoint")
        tensors = {name: checkpoint.get_tensor(name) for name in checkpoint.keys()}
    config_data = json.loads(metadata["config"])
    if (
        "order_beta" in config_data["balance_train"]
        or "order_balance_weight" in config_data["train"]
    ):
        raise ValueError("source already contains order-balancing config fields")
    config_data["balance_train"]["order_beta"] = order_beta
    config_data["train"]["order_balance_weight"] = order_balance_weight
    config = Config.model_validate(config_data)
    validate_config(config)
    step_config = instantiate_scheduleable_config(config, int(metadata["num_episodes"]))
    room_path = Path(config.room_set)
    if not room_path.is_absolute():
        room_path = Path(__file__).resolve().parents[1] / room_path
    rooms = json.loads(room_path.read_text())
    torch.manual_seed(seed)
    engine = Engine(
        rooms,
        step_config.features,
        step_config.generation.min_area_size,
        step_config.generation.max_area_size,
    )
    model = FrontierModel(**frontier_model_kwargs(step_config, rooms, engine))
    added_main = add_model_parameters(tensors, model, "main_model", "order_balance_score_output.")
    optimizer = create_main_optimizer(model, step_config.optimizer)
    extend_optimizer_groups(tensors, metadata, model, optimizer, "optimizer", added_main)
    # A new head starts at zero in both networks; no calibration is needed.
    added_ema = add_model_parameters(tensors, model, "ema_model", "order_balance_score_output.")
    for suffix in ("weight", "bias"):
        assert torch.count_nonzero(tensors[f"main_model.order_balance_score_output.{suffix}"]) == 0
        assert torch.count_nonzero(tensors[f"ema_model.order_balance_score_output.{suffix}"]) == 0
    balance = create_balance_model(step_config, rooms, engine, torch.device("cpu"))
    added_balance = add_model_parameters(tensors, balance, "balance_model", "order_net.")
    balance_optimizer = create_adam_optimizer(balance.parameters(), step_config.balance_optimizer)
    extend_optimizer_groups(
        tensors, metadata, balance, balance_optimizer, "balance_optimizer", added_balance
    )
    assert torch.count_nonzero(balance.order_net[-1].weight) == 0
    assert torch.count_nonzero(balance.order_net[-1].bias) == 0
    report = {
        "source": str(source),
        "source_format": SOURCE_FORMAT,
        "output_format": TRAINING_CHECKPOINT_FORMAT,
        "seed": seed,
        "order_beta": order_beta,
        "order_balance_weight": order_balance_weight,
        "added_main": sorted(added_main),
        "added_ema": sorted(added_ema),
        "added_balance": sorted(added_balance),
    }
    metadata["format"] = TRAINING_CHECKPOINT_FORMAT
    metadata["config"] = config.model_dump_json()
    metadata["order_balance_migration"] = json.dumps(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent, suffix=".safetensors", delete=False
    ) as temp:
        temporary = Path(temp.name)
    try:
        save_file(tensors, temporary, metadata=metadata)
        os.link(temporary, output)  # Atomically publish without replacing an existing file.
    finally:
        temporary.unlink()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--order-beta", type=float, required=True)
    parser.add_argument("--order-balance-weight", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            migrate_checkpoint(
                args.source, args.output, args.order_beta, args.order_balance_weight, args.seed
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
