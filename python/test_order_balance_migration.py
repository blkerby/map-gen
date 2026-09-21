import json
from pathlib import Path
import sys
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env import Engine
from model import FrontierModel
from model_loading import create_balance_model, frontier_model_kwargs, without_prefix
from scripts.migrate_order_balance import SOURCE_FORMAT, migrate_checkpoint
from test_room_area_plumbing import one_tile_room
from train import (
    TRAINING_CHECKPOINT_FORMAT,
    create_main_optimizer,
    create_adam_optimizer,
    load_named_optimizer_checkpoint_state,
    named_checkpoint_optimizers,
    prefixed_state_dict,
    save_named_optimizer_checkpoint_state,
)
from train_config import Config, instantiate_scheduleable_config


def make_source(root: Path, optimizer_type: str) -> tuple[Path, dict, dict]:
    rooms = [one_tile_room(f"Room {i}", "left" if i % 2 else "right") for i in range(6)]
    room_path = root / "rooms.json"
    room_path.write_text(json.dumps(rooms))
    data = json.loads(Path("configs/debug.json").read_text())
    data["room_set"] = str(room_path)
    data["model"]["embedding_width"] = 8
    data["model"]["global_embedding_width"] = 8
    data["model"]["hidden_width"] = 16
    data["balance_model"]["hidden_width"] = 8
    if optimizer_type == "muon":
        data["optimizer"] = {
            "type": "muon",
            "adam": {"lr": 0.001, "beta1": 0.9, "beta2": 0.99},
            "muon": {
                "lr": 0.001,
                "momentum": 0.95,
                "nesterov": True,
                "backend": "newtonschulz5",
                "backend_steps": 1,
            },
        }
    config = instantiate_scheduleable_config(Config.model_validate(data), 256)
    engine = Engine(
        rooms, config.features, config.generation.min_area_size, config.generation.max_area_size
    )
    main = FrontierModel(**frontier_model_kwargs(config, rooms, engine))
    balance = create_balance_model(config, rooms, engine, torch.device("cpu"))
    # Reconstruct the exact pre-order parameter layout before creating optimizers.
    del main.order_balance_score_output
    del balance.order_net
    optimizer = create_main_optimizer(main, config.optimizer)
    balance_optimizer = create_adam_optimizer(balance.parameters(), config.balance_optimizer)
    for model, opt in ((main, optimizer), (balance, balance_optimizer)):
        for index, parameter in enumerate(model.parameters()):
            parameter.grad = torch.full_like(parameter, (index + 1) / 1000)
        opt.step()
        for part in named_checkpoint_optimizers(opt).values():
            part.zero_grad(set_to_none=True)
    tensors = prefixed_state_dict("main_model", main)
    tensors.update(
        {key: value.clone() for key, value in prefixed_state_dict("ema_model", main).items()}
    )
    tensors.update(prefixed_state_dict("balance_model", balance))
    del data["balance_train"]["order_beta"]
    del data["train"]["order_balance_weight"]
    metadata = {
        "format": SOURCE_FORMAT,
        "config": json.dumps(data),
        "num_episodes": "256",
        "experience_num_files": "1",
        "aim_run_hash": "synthetic-test",
    }
    save_named_optimizer_checkpoint_state(tensors, metadata, optimizer, "optimizer")
    save_named_optimizer_checkpoint_state(
        tensors, metadata, balance_optimizer, "balance_optimizer"
    )
    original_ids = {}
    for model, opt, prefix in (
        (main, optimizer, "optimizer"),
        (balance, balance_optimizer, "balance_optimizer"),
    ):
        names = {id(parameter): name for name, parameter in model.named_parameters()}
        for part_name, part in named_checkpoint_optimizers(opt).items():
            original_ids[f"{prefix}.{part_name}"] = {
                names[id(parameter)]: saved_id
                for saved, group in zip(
                    part.state_dict()["param_groups"], part.param_groups, strict=True
                )
                for saved_id, parameter in zip(saved["params"], group["params"], strict=True)
            }
    metadata["test_original_parameter_ids"] = json.dumps(original_ids)
    source = root / "source.safetensors"
    save_file(tensors, source, metadata=metadata)
    return source, tensors, metadata


def check_migration(optimizer_type: str) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source, old_tensors, old_metadata = make_source(root, optimizer_type)
        output = root / "migrated.safetensors"
        migrate_checkpoint(source, output, 0.5, 2.0, 17)
        with safe_open(output, framework="pt") as checkpoint:
            metadata = checkpoint.metadata()
            tensors = {key: checkpoint.get_tensor(key) for key in checkpoint.keys()}
        assert metadata["format"] == TRAINING_CHECKPOINT_FORMAT
        for key in ("num_episodes", "experience_num_files", "aim_run_hash"):
            assert metadata[key] == old_metadata[key]
        for key, value in old_tensors.items():
            torch.testing.assert_close(tensors[key], value, rtol=0, atol=0)
        for prefix in ("optimizer", "balance_optimizer"):
            for part in json.loads(old_metadata[f"{prefix}_names"]):
                assert (
                    metadata[f"{prefix}_{part}_scalar_state"]
                    == old_metadata[f"{prefix}_{part}_scalar_state"]
                )
        config = instantiate_scheduleable_config(
            Config.model_validate_json(metadata["config"]), 256
        )
        assert config.balance_train.order_beta == 0.5
        assert config.train.order_balance_weight == 2.0
        rooms = json.loads(Path(config.room_set).read_text())
        engine = Engine(
            rooms,
            config.features,
            config.generation.min_area_size,
            config.generation.max_area_size,
        )
        main = FrontierModel(**frontier_model_kwargs(config, rooms, engine))
        main.load_state_dict(without_prefix(tensors, "main_model"), strict=True)
        balance = create_balance_model(config, rooms, engine, torch.device("cpu"))
        balance.load_state_dict(without_prefix(tensors, "balance_model"), strict=True)
        optimizer = create_main_optimizer(main, config.optimizer)
        controller_optimizer = create_adam_optimizer(
            balance.parameters(), config.balance_optimizer
        )
        for model, opt, prefix, new_prefix in (
            (main, optimizer, "optimizer", "order_balance_score_output."),
            (balance, controller_optimizer, "balance_optimizer", "order_net."),
        ):
            load_named_optimizer_checkpoint_state(opt, tensors, metadata, prefix)
            names = {id(p): name for name, p in model.named_parameters()}
            original_ids = json.loads(old_metadata["test_original_parameter_ids"])
            for part_name, part in named_checkpoint_optimizers(opt).items():
                for group in part.param_groups:
                    for parameter in group["params"]:
                        if names[id(parameter)].startswith(new_prefix):
                            assert not part.state.get(parameter)
                        else:
                            assert part.state[parameter]
                            old_id = original_ids[f"{prefix}.{part_name}"][names[id(parameter)]]
                            for state_name, value in part.state[parameter].items():
                                if torch.is_tensor(value):
                                    expected = old_tensors[
                                        f"{prefix}.{part_name}.state.{old_id}.{state_name}"
                                    ]
                                    torch.testing.assert_close(value, expected, rtol=0, atol=0)
            # An actual resumed optimizer step must initialize every new parameter.
            for parameter in model.parameters():
                parameter.grad = torch.full_like(parameter, 0.125)
            opt.step()
        for prefix in ("main_model", "ema_model"):
            for suffix in ("weight", "bias"):
                assert (
                    tensors[f"{prefix}.order_balance_score_output.{suffix}"].count_nonzero() == 0
                )
        try:
            migrate_checkpoint(source, output, 1.0, 1.0, 0)
        except FileExistsError:
            pass
        else:
            raise AssertionError("migration must not overwrite output")
        with safe_open(source, framework="pt") as checkpoint:
            assert checkpoint.metadata()["format"] == SOURCE_FORMAT
            assert set(checkpoint.keys()) == set(old_tensors)


def test_adam_migration() -> None:
    check_migration("adam")


def test_muon_migration() -> None:
    check_migration("muon")


if __name__ == "__main__":
    torch.set_num_threads(2)
    test_adam_migration()
    test_muon_migration()
