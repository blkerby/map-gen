#!/usr/bin/env python3
"""Upgrade a v22 checkpoint with a zero-output room-placement-step controller.

Example (from the repository root):
  conda run -n map-gen python scripts/migrate_step_balance.py SOURCE OUTPUT \
      --step-beta 1 --step-price-scale 1 --step-balance-weight 1 --seed 0

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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from env import Engine
from model import FrontierModel
from model_loading import create_balance_model, frontier_model_kwargs
from train import (
    TRAINING_CHECKPOINT_FORMAT,
    create_adam_optimizer,
    create_main_optimizer,
    validate_checkpoint_metadata,
)
from train_config import Config, instantiate_scheduleable_config, validate_config
from scripts.checkpoint_migration import add_model_parameters, extend_optimizer_groups

SOURCE_FORMAT = "map-gen-training-session-checkpoint-v22"
TARGET_FORMAT = "map-gen-training-session-checkpoint-v23"


def migrate_checkpoint(
    source: Path, output: Path, step_beta: float, step_price_scale: float,
    step_balance_weight: float, seed: int
) -> dict:
    if TRAINING_CHECKPOINT_FORMAT != TARGET_FORMAT:
        raise ValueError("this migration requires the v23 model implementation")
    if output.exists():
        raise FileExistsError(output)
    with safe_open(source, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata()
        if metadata is None or metadata["format"] != SOURCE_FORMAT:
            raise ValueError(f"expected a {SOURCE_FORMAT} checkpoint")
        tensors = {name: checkpoint.get_tensor(name) for name in checkpoint.keys()}
    config_data = json.loads(metadata["config"])
    if (
        "step_beta" in config_data["balance_train"]
        or "step_price_scale" in config_data["balance_train"]
        or "step_balance_weight" in config_data["train"]
    ):
        raise ValueError("source already contains step-balancing config fields")
    config_data["balance_train"]["step_beta"] = step_beta
    config_data["balance_train"]["step_price_scale"] = step_price_scale
    config_data["train"]["step_balance_weight"] = step_balance_weight
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
    added_main = add_model_parameters(tensors, model, "main_model", "step_balance_score_output.")
    optimizer = create_main_optimizer(model, step_config.optimizer)
    extend_optimizer_groups(tensors, metadata, model, optimizer, "optimizer", added_main)
    # A new head starts at zero in both networks; no calibration is needed.
    added_ema = add_model_parameters(tensors, model, "ema_model", "step_balance_score_output.")
    for suffix in ("weight", "bias"):
        assert torch.count_nonzero(tensors[f"main_model.step_balance_score_output.{suffix}"]) == 0
        assert torch.count_nonzero(tensors[f"ema_model.step_balance_score_output.{suffix}"]) == 0
    balance = create_balance_model(step_config, rooms, engine, torch.device("cpu"))
    added_balance = add_model_parameters(tensors, balance, "balance_model", "step_net.")
    balance_optimizer = create_adam_optimizer(balance.parameters(), step_config.balance_optimizer)
    extend_optimizer_groups(
        tensors, metadata, balance, balance_optimizer, "balance_optimizer", added_balance
    )
    assert torch.count_nonzero(balance.step_net[-1].weight) == 0
    assert torch.count_nonzero(balance.step_net[-1].bias) == 0
    with safe_open(source, framework="pt", device="cpu") as checkpoint:
        for name in checkpoint.keys():
            if not torch.equal(tensors[name], checkpoint.get_tensor(name)):
                raise ValueError(f"migration changed an existing tensor: {name}")
    report = {
        "source": str(source),
        "source_format": SOURCE_FORMAT,
        "output_format": TRAINING_CHECKPOINT_FORMAT,
        "seed": seed,
        "step_beta": step_beta,
        "step_price_scale": step_price_scale,
        "step_balance_weight": step_balance_weight,
        "added_main": sorted(added_main),
        "added_ema": sorted(added_ema),
        "added_balance": sorted(added_balance),
    }
    metadata["format"] = TRAINING_CHECKPOINT_FORMAT
    metadata["config"] = config.model_dump_json()
    metadata["step_balance_migration"] = json.dumps(report)
    validate_checkpoint_metadata(output, metadata)
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
    parser.add_argument("--step-beta", type=float, required=True)
    parser.add_argument("--step-price-scale", type=float, required=True)
    parser.add_argument("--step-balance-weight", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(
        json.dumps(
            migrate_checkpoint(
                args.source, args.output, args.step_beta, args.step_price_scale,
                args.step_balance_weight, args.seed
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
