#!/usr/bin/env python3
"""Export inference-ready weights.

From the repository root:
  python scripts/export_model.py CHECKPOINT OUTPUT --backbone-dtype bfloat16

Use --backbone-dtype float32 for an entirely float32 model export. Prediction
blocks retain float32 weights in both modes.
"""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import torch
import safetensors.torch
from safetensors import safe_open

from env import Engine
from model_loading import (
    INFERENCE_BACKBONE_DTYPES,
    MODEL_EXPORT_FORMAT,
    create_inference_models,
    inference_backbone_dtype,
)
from train_config import Config, validate_config


TRAINING_CHECKPOINT_FORMAT = "map-gen-training-session-checkpoint-v25"
MODEL_PREFIX_MAP = {
    "ema_model": "ema_model",
    "balance_model": "balance_model",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the EMA model and balance model weights from a training checkpoint "
            "to a smaller safetensors file."
        ),
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Training checkpoint safetensors file to export from.",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output safetensors file containing only ema_model.* and balance_model.* tensors.",
    )
    parser.add_argument(
        "--backbone-dtype",
        choices=tuple(INFERENCE_BACKBONE_DTYPES),
        required=True,
        help="Saved backbone dtype; prediction blocks always retain float32 weights.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it already exists.",
    )
    return parser.parse_args()


def validate_checkpoint_metadata(path: Path, metadata: dict[str, str] | None) -> dict[str, str]:
    if metadata is None:
        raise ValueError(f"checkpoint metadata missing in {path}")
    if metadata["format"] != TRAINING_CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported checkpoint format in {path}")
    for field in ("config", "num_episodes", "aim_run_hash"):
        if field not in metadata:
            raise ValueError(f"checkpoint metadata field {field!r} missing in {path}")
    return metadata


def exported_model_name(name: str) -> tuple[str, str] | None:
    for checkpoint_prefix, export_prefix in MODEL_PREFIX_MAP.items():
        if name.startswith(f"{checkpoint_prefix}."):
            return checkpoint_prefix, name.replace(checkpoint_prefix, export_prefix, 1)
    return None


def export_model_tensors(
    checkpoint_path: Path,
    output_path: Path,
    backbone_dtype: str,
    overwrite: bool,
) -> None:
    dtype = inference_backbone_dtype(backbone_dtype)
    if checkpoint_path.resolve() == output_path.resolve():
        raise ValueError("model export must not overwrite the training checkpoint")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output file already exists: {output_path}")

    with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
        checkpoint_metadata = validate_checkpoint_metadata(
            checkpoint_path,
            checkpoint.metadata(),
        )
        tensors = {}
        tensor_counts = dict.fromkeys(MODEL_PREFIX_MAP, 0)
        for name in checkpoint.keys():
            export_name = exported_model_name(name)
            if export_name is None:
                continue
            checkpoint_prefix, output_name = export_name
            tensors[output_name] = checkpoint.get_tensor(name)
            tensor_counts[checkpoint_prefix] += 1

    missing_prefixes = [prefix for prefix, count in tensor_counts.items() if count == 0]
    if missing_prefixes:
        raise ValueError(
            f"checkpoint missing model tensor group(s): {', '.join(missing_prefixes)}"
        )

    config = Config.model_validate_json(checkpoint_metadata["config"])
    validate_config(config)
    rooms = json.loads(Path(config.room_set).read_text())
    engine = Engine(
        rooms, config.features, config.generation.min_area_size, config.generation.max_area_size
    )
    models = create_inference_models(
        config, rooms, engine, tensors, torch.float32, torch.device("cpu")
    )
    tensors = {}
    for prefix, model in zip(MODEL_PREFIX_MAP.values(), models, strict=True):
        model.configure_inference_dtype(dtype)
        tensors.update(
            {
                f"{prefix}.{name}": value.detach().contiguous()
                for name, value in model.state_dict().items()
            }
        )

    metadata = {
        "format": MODEL_EXPORT_FORMAT,
        "backbone_dtype": backbone_dtype,
        "config": checkpoint_metadata["config"],
        "source_format": checkpoint_metadata["format"],
        "source_num_episodes": checkpoint_metadata["num_episodes"],
        "source_aim_run_hash": checkpoint_metadata["aim_run_hash"],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    safetensors.torch.save_file(tensors, temp_path, metadata=metadata)
    os.replace(temp_path, output_path)

    print(
        f"Exported {len(tensors)} tensor(s) from {checkpoint_path} to {output_path}: "
        f"{', '.join(f'{prefix}={tensor_counts[prefix]}' for prefix in MODEL_PREFIX_MAP)}"
    )


def main() -> int:
    args = parse_args()
    export_model_tensors(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        backbone_dtype=args.backbone_dtype,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
