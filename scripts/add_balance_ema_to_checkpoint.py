#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import safetensors.torch
from safetensors import safe_open


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from train import TRAINING_CHECKPOINT_FORMAT  # noqa: E402
from train_config import Config, validate_config  # noqa: E402


SOURCE_FORMAT = "map-gen-training-session-checkpoint-v9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add balance EMA weights to a version 9 training checkpoint and write a "
            "version 10 checkpoint. The EMA is initialized from the raw balance model."
        )
    )
    parser.add_argument("config", type=Path, help="Version 10 training config JSON.")
    parser.add_argument("input", type=Path, help="Version 9 training checkpoint.")
    parser.add_argument("output", type=Path, help="Version 10 training checkpoint.")
    return parser.parse_args()


def add_balance_ema_to_checkpoint(
    config_path: Path,
    input_path: Path,
    output_path: Path,
) -> None:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output checkpoint paths must differ")
    if output_path.exists():
        raise FileExistsError(f"output checkpoint already exists: {output_path}")

    config = Config.model_validate_json(config_path.read_text())
    validate_config(config)
    with safe_open(input_path, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata()
        if metadata is None:
            raise ValueError(f"checkpoint metadata missing in {input_path}")
        if metadata.get("format") != SOURCE_FORMAT:
            raise ValueError(
                f"unsupported checkpoint format in {input_path}: {metadata.get('format')!r}"
            )
        tensors = {name: checkpoint.get_tensor(name) for name in checkpoint.keys()}

    balance_names = [name for name in tensors if name.startswith("balance_model.")]
    if not balance_names:
        raise ValueError(f"checkpoint balance_model tensor group missing in {input_path}")
    if any(name.startswith("balance_ema_model.") for name in tensors):
        raise ValueError(f"checkpoint already contains balance_ema_model tensors: {input_path}")
    for name in balance_names:
        ema_name = name.replace("balance_model.", "balance_ema_model.", 1)
        tensors[ema_name] = tensors[name].clone()

    output_metadata = dict(metadata)
    output_metadata["format"] = TRAINING_CHECKPOINT_FORMAT
    output_metadata["config"] = config.model_dump_json()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    safetensors.torch.save_file(tensors, temporary_path, metadata=output_metadata)
    os.replace(temporary_path, output_path)


def main() -> None:
    args = parse_args()
    add_balance_ema_to_checkpoint(args.config, args.input, args.output)
    print(f"Wrote checkpoint: {args.output}")


if __name__ == "__main__":
    main()
