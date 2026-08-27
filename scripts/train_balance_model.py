#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import safetensors.torch
import torch
from safetensors import safe_open


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from env import (  # noqa: E402
    Engine,
    forced_special_room_mask,
)
from experience import (  # noqa: E402
    EXPERIENCE_FORMAT,
    REQUIRED_BALANCE_EXPERIENCE_TENSORS,
    load_balance_experience,
)
from learn import (  # noqa: E402
    VANILLA_AREA_CONDITION_INDICES,
    episode_room_area,
    set_optimizer_lrs,
    train_balance_batch,
)
from model_loading import create_balance_model  # noqa: E402
from train import (  # noqa: E402
    TRAINING_CHECKPOINT_FORMAT,
    create_adam_optimizer,
    prefixed_state_dict,
    save_named_optimizer_checkpoint_state,
)
from train_config import (  # noqa: E402
    Config,
    instantiate_scheduleable_config,
    validate_config,
)


REQUIRED_CHECKPOINT_METADATA = (
    "format",
    "config",
    "aim_run_hash",
    "num_episodes",
    "experience_num_files",
)
COPIED_CHECKPOINT_PREFIXES = ("main_model.", "ema_model.", "optimizer.")


def parse_ema_beta(value: str) -> float:
    beta = float(value)
    if not 0.0 <= beta < 1.0:
        raise argparse.ArgumentTypeError("must be greater than or equal to 0 and less than 1")
    return beta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a new balance model by replaying an inclusive range of experience "
            "files in numeric order, then replace the balance model in a full checkpoint."
        ),
    )
    parser.add_argument("config", type=Path, help="Training config JSON.")
    parser.add_argument("input_checkpoint", type=Path, help="Checkpoint to copy from.")
    parser.add_argument("experience_dir", type=Path, help="Directory containing N.safetensors.")
    parser.add_argument("first_file", type=int, help="First experience file number (inclusive).")
    parser.add_argument("last_file", type=int, help="Last experience file number (inclusive).")
    parser.add_argument("output_checkpoint", type=Path, help="Completed output checkpoint.")
    parser.add_argument(
        "--loss-ema-beta",
        type=parse_ema_beta,
        required=True,
        help="Exponential moving average beta for reported balance loss (0 <= beta < 1).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device (default: cuda when available, otherwise cpu).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output checkpoint if it already exists.",
    )
    return parser.parse_args()


def experience_paths(directory: Path, first_file: int, last_file: int) -> list[Path]:
    if first_file < 0:
        raise ValueError("first_file must be nonnegative")
    if last_file < first_file:
        raise ValueError("last_file must be greater than or equal to first_file")
    paths = [directory / f"{file_num}.safetensors" for file_num in range(first_file, last_file + 1)]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"experience file does not exist: {missing[0]}")
    return paths


def validate_checkpoint(path: Path) -> int:
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata()
        if metadata is None:
            raise ValueError(f"checkpoint metadata missing in {path}")
        missing_metadata = [name for name in REQUIRED_CHECKPOINT_METADATA if name not in metadata]
        if missing_metadata:
            raise ValueError(
                f"checkpoint metadata field(s) missing in {path}: {', '.join(missing_metadata)}"
            )
        if metadata["format"] != TRAINING_CHECKPOINT_FORMAT:
            raise ValueError(f"unsupported checkpoint format in {path}: {metadata['format']}")
        tensor_names = list(checkpoint.keys())
        missing_prefixes = [
            prefix[:-1]
            for prefix in COPIED_CHECKPOINT_PREFIXES
            if not any(name.startswith(prefix) for name in tensor_names)
        ]
        if missing_prefixes:
            raise ValueError(
                f"checkpoint tensor group(s) missing in {path}: {', '.join(missing_prefixes)}"
            )
        try:
            num_episodes = int(metadata["num_episodes"])
        except ValueError as error:
            raise ValueError(
                f"checkpoint num_episodes is not an integer in {path}: {metadata['num_episodes']}"
            ) from error
        if num_episodes < 0:
            raise ValueError(f"checkpoint num_episodes must be nonnegative in {path}")
        return num_episodes


def experience_episode_count(path: Path) -> int:
    with safe_open(path, framework="pt", device="cpu") as experience:
        metadata = experience.metadata()
        if metadata is None or metadata.get("format") != EXPERIENCE_FORMAT:
            raise ValueError(f"unsupported experience format in {path}")
        missing = [
            name
            for name in REQUIRED_BALANCE_EXPERIENCE_TENSORS
            if name not in experience.keys()
        ]
        if missing:
            raise ValueError(f"{path} missing tensor(s): {', '.join(missing)}")
        shape = experience.get_slice("room_idx").get_shape()
    if len(shape) != 2:
        raise ValueError(f"{path} room_idx must be two-dimensional, got {tuple(shape)}")
    return shape[0]


def interpolate_schedule_episode(
    processed_episodes: int,
    total_replay_episodes: int,
    checkpoint_episodes: int,
) -> int:
    return round(processed_episodes / total_replay_episodes * checkpoint_episodes)


def write_checkpoint(
    input_path: Path,
    output_path: Path,
    config_json: str,
    balance_model: torch.nn.Module,
    balance_optimizer: torch.optim.Optimizer,
) -> None:
    with safe_open(input_path, framework="pt", device="cpu") as checkpoint:
        metadata = dict(checkpoint.metadata() or {})
        tensors = {
            name: checkpoint.get_tensor(name)
            for name in checkpoint.keys()
            if not name.startswith(("balance_model.", "balance_optimizer."))
        }

    for name in list(metadata):
        if name.startswith("balance_optimizer_"):
            del metadata[name]
    metadata["config"] = config_json
    tensors.update(prefixed_state_dict("balance_model", balance_model))
    save_named_optimizer_checkpoint_state(
        tensors,
        metadata,
        balance_optimizer,
        "balance_optimizer",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    safetensors.torch.save_file(tensors, temporary_path, metadata=metadata)
    os.replace(temporary_path, output_path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
    )
    if args.output_checkpoint.exists() and not args.overwrite:
        raise FileExistsError(f"output checkpoint already exists: {args.output_checkpoint}")
    checkpoint_episodes = validate_checkpoint(args.input_checkpoint)
    paths = experience_paths(args.experience_dir, args.first_file, args.last_file)

    config = Config.model_validate_json(args.config.read_text())
    validate_config(config)
    rooms = json.loads(config.room_set.read_text())
    episode_counts = {path: experience_episode_count(path) for path in paths}
    for path, episode_count in episode_counts.items():
        if episode_count % config.balance_train.batch_size != 0:
            raise ValueError(
                f"{path} has {episode_count} episodes, which is not divisible by "
                f"balance_train.batch_size={config.balance_train.batch_size}"
            )
    total_replay_episodes = sum(episode_counts.values())
    if total_replay_episodes == 0:
        raise ValueError("experience range contains no episodes")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.set_float32_matmul_precision("high")

    engine = Engine(
        rooms,
        config.features,
        config.generation.min_area_size,
        config.generation.max_area_size,
    )
    balance_model = create_balance_model(config, rooms, engine, device)
    initial_config = instantiate_scheduleable_config(config, 0)
    balance_optimizer = create_adam_optimizer(
        balance_model.parameters(),
        config.balance_optimizer,
        initial_config.balance_optimizer,
    )

    processed_episodes = 0
    optimizer_step = 0
    ema_loss = 0.0
    ema_weight = 0.0
    logging.info(
        "step\tfile\tbatch\tepisodes\tlr\tbalance_loss\tbalance_loss_ema"
    )
    for path in paths:
        actions, variables = load_balance_experience(path, len(rooms))
        episode_count = actions.room_idx.shape[0]
        if episode_count != episode_counts[path]:
            raise ValueError(f"{path} episode count changed while training")
        logging.info("Computing balance targets for %s", path)
        door_matches, toilet_crossed_room_idx = engine.compute_balance_targets(actions, device)
        variables = variables.to(device)
        door_matches = door_matches.to(device)
        toilet_crossed_room_idx = toilet_crossed_room_idx.to(device)
        room_area = episode_room_area(actions, len(rooms)).to(device)
        vanilla_area_constraint_mask = variables[:, VANILLA_AREA_CONDITION_INDICES].to(torch.bool)
        room_area_mask = ~forced_special_room_mask(rooms, vanilla_area_constraint_mask)
        record_weight = torch.ones(episode_count, dtype=torch.float32, device=device)
        for start in range(0, episode_count, config.balance_train.batch_size):
            end = start + config.balance_train.batch_size
            schedule_episode = interpolate_schedule_episode(
                processed_episodes + end - start,
                total_replay_episodes,
                checkpoint_episodes,
            )
            step_config = instantiate_scheduleable_config(config, schedule_episode)
            set_optimizer_lrs(balance_optimizer, step_config.balance_optimizer)
            loss = train_balance_batch(
                generation_variable_floats=variables[start:end],
                door_matches=door_matches.slice(start, end),
                toilet_crossed_room_idx=toilet_crossed_room_idx[start:end],
                room_area=room_area[start:end],
                room_area_mask=room_area_mask[start:end],
                record_weight=record_weight[start:end],
                balance_model=balance_model,
                balance_optimizer=balance_optimizer,
            )
            processed_episodes += end - start
            optimizer_step += 1
            ema_loss = (
                args.loss_ema_beta * ema_loss + (1.0 - args.loss_ema_beta) * loss
            )
            ema_weight = (
                args.loss_ema_beta * ema_weight + (1.0 - args.loss_ema_beta)
            )
            lr = balance_optimizer.param_groups[0]["lr"]
            logging.info(
                "%s\t%s\t%s\t%s\t%.9g\t%.9g\t%.9g",
                optimizer_step,
                path.stem,
                start // config.balance_train.batch_size + 1,
                processed_episodes,
                lr,
                loss,
                ema_loss / ema_weight,
            )

    if optimizer_step == 0:
        raise RuntimeError("no balance optimizer steps were completed")
    write_checkpoint(
        args.input_checkpoint,
        args.output_checkpoint,
        config.model_dump_json(),
        balance_model,
        balance_optimizer,
    )
    logging.info("Wrote checkpoint: %s", args.output_checkpoint)


if __name__ == "__main__":
    main()
