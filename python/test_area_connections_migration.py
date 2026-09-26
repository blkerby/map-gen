import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import optimizers
from env import Engine
from model import FrontierModel
from model_loading import create_balance_model, frontier_model_kwargs, without_prefix
from scripts.migrate_area_connections import (
    NEW_HEAD,
    REWARD_FIELD,
    SOURCE_FORMAT,
    TARGET_FORMAT,
    global_added_columns,
    migrate_checkpoint,
)
from test_room_area_plumbing import one_tile_room
from train import (
    create_adam_optimizer,
    create_main_optimizer,
    load_named_optimizer_checkpoint_state,
    named_checkpoint_optimizers,
    prefixed_state_dict,
    save_named_optimizer_checkpoint_state,
)
from train_config import Config, GENERATION_VARIABLE_FLOAT_FIELDS, instantiate_scheduleable_config


def remove_columns(model: torch.nn.Module, columns: dict[str, list[int]]) -> None:
    for name, removed in columns.items():
        module_name, _ = name.rsplit(".", 1)
        module = model.get_submodule(module_name)
        keep = torch.ones(module.in_features, dtype=torch.bool)
        keep[removed] = False
        module.weight = torch.nn.Parameter(module.weight[:, keep].detach().clone())
        module.in_features = int(keep.sum())


def initialize_optimizer_state(model, optimizer) -> None:
    for index, parameter in enumerate(model.parameters()):
        parameter.grad = torch.full_like(parameter, (index + 1) / 1000)
    # Use the actual Muon math without compiling tiny synthetic fixtures.
    with patch(
        "optimizers.zeropower_via_newtonschulz5",
        optimizers.zeropower_via_newtonschulz5._torchdynamo_orig_callable,
    ):
        optimizer.step()
    for part in named_checkpoint_optimizers(optimizer).values():
        part.zero_grad(set_to_none=True)


def check_migration(optimizer_type: str, generation_features: bool, lookahead: int) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        room_path = root / "rooms.json"
        rooms = [one_tile_room(str(i), "left" if i % 2 else "right") for i in range(6)]
        room_path.write_text(json.dumps(rooms))
        data = json.loads(Path("configs/debug.json").read_text())
        data["room_set"] = str(room_path)
        data["model"].update(embedding_width=8, global_embedding_width=8, hidden_width=8)
        data["balance_model"]["hidden_width"] = 8
        data["features"]["generation_variable_floats"] = generation_features
        data["features"]["lookahead_outcomes"] = lookahead
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
        engine = Engine(rooms, config.features, 1, 100)
        main = FrontierModel(**frontier_model_kwargs(config, rooms, engine))
        balance = create_balance_model(config, rooms, engine, torch.device("cpu"))
        columns = global_added_columns(config, rooms, engine)
        main_columns = {"global_mlp.weight": columns} if columns else {}
        reward_column = GENERATION_VARIABLE_FLOAT_FIELDS.index(REWARD_FIELD)
        balance_columns = {
            f"{family}_net.0.weight": [reward_column]
            for family in ("door", "toilet", "area", "order", "step")
        }
        del main.area_connection_output
        remove_columns(main, main_columns)
        remove_columns(balance, balance_columns)
        optimizer = create_main_optimizer(main, config.optimizer)
        balance_optimizer = create_adam_optimizer(balance.parameters(), config.balance_optimizer)
        initialize_optimizer_state(main, optimizer)
        initialize_optimizer_state(balance, balance_optimizer)
        tensors = prefixed_state_dict("main_model", main)
        tensors.update(
            {key: value.clone() for key, value in prefixed_state_dict("ema_model", main).items()}
        )
        tensors.update(prefixed_state_dict("balance_model", balance))
        del data["generation"][REWARD_FIELD]
        del data["train"]["area_distinct_crossing_weight"]
        metadata = {
            "format": SOURCE_FORMAT,
            "config": json.dumps(data),
            "num_episodes": "256",
            "experience_num_files": "3",
            "aim_run_hash": "area-migration-test",
        }
        expansions = {
            f"{prefix}.{name}": indices
            for prefix in ("main_model", "ema_model")
            for name, indices in main_columns.items()
        }
        expansions.update(
            {f"balance_model.{name}": indices for name, indices in balance_columns.items()}
        )
        for model, opt, prefix, inputs in (
            (main, optimizer, "optimizer", main_columns),
            (balance, balance_optimizer, "balance_optimizer", balance_columns),
        ):
            save_named_optimizer_checkpoint_state(tensors, metadata, opt, prefix)
            names = {id(parameter): name for name, parameter in model.named_parameters()}
            for part_name, part in named_checkpoint_optimizers(opt).items():
                for saved, group in zip(
                    part.state_dict()["param_groups"], part.param_groups, strict=True
                ):
                    for parameter_id, parameter in zip(
                        saved["params"], group["params"], strict=True
                    ):
                        name = names[id(parameter)]
                        if name in inputs:
                            state_prefix = f"{prefix}.{part_name}.state.{parameter_id}."
                            for key, value in tensors.items():
                                if key.startswith(state_prefix) and value.ndim:
                                    expansions[key] = inputs[name]
        source = root / "source.safetensors"
        output = root / "output.safetensors"
        save_file(tensors, source, metadata=metadata)
        original_bytes = source.read_bytes()
        migrate_checkpoint(source, output)
        with safe_open(output, framework="pt") as checkpoint:
            migrated_metadata = checkpoint.metadata()
            migrated = {name: checkpoint.get_tensor(name) for name in checkpoint.keys()}
        assert migrated_metadata["format"] == TARGET_FORMAT
        for key in ("num_episodes", "experience_num_files", "aim_run_hash"):
            assert migrated_metadata[key] == metadata[key]
        for key, old in tensors.items():
            new = migrated[key]
            if key in expansions:
                keep = torch.ones(new.shape[1], dtype=torch.bool)
                keep[expansions[key]] = False
                torch.testing.assert_close(new[:, keep], old, rtol=0, atol=0)
                assert not new[:, ~keep].count_nonzero()
                inputs = torch.randn((2, new.shape[1]))
                torch.testing.assert_close(inputs @ new.T, inputs[:, keep] @ old.T)
            else:
                torch.testing.assert_close(new, old, rtol=0, atol=0)
        new_config = instantiate_scheduleable_config(
            Config.model_validate_json(migrated_metadata["config"]), 256
        )
        assert new_config.generation.reward_area_distinct_crossing == 0
        assert new_config.train.area_distinct_crossing_weight == 1
        new_main = FrontierModel(**frontier_model_kwargs(new_config, rooms, engine))
        new_main.load_state_dict(without_prefix(migrated, "main_model"), strict=True)
        new_balance = create_balance_model(new_config, rooms, engine, torch.device("cpu"))
        new_balance.load_state_dict(without_prefix(migrated, "balance_model"), strict=True)
        for prefix in ("main_model", "ema_model"):
            assert not migrated[f"{prefix}.{NEW_HEAD}weight"].count_nonzero()
            assert not migrated[f"{prefix}.{NEW_HEAD}bias"].count_nonzero()
        new_optimizer = create_main_optimizer(new_main, new_config.optimizer)
        new_balance_optimizer = create_adam_optimizer(
            new_balance.parameters(), new_config.balance_optimizer
        )
        for model, opt, prefix in (
            (new_main, new_optimizer, "optimizer"),
            (new_balance, new_balance_optimizer, "balance_optimizer"),
        ):
            load_named_optimizer_checkpoint_state(opt, migrated, migrated_metadata, prefix)
            for part in named_checkpoint_optimizers(opt).values():
                for parameter, state in part.state.items():
                    for value in state.values():
                        if torch.is_tensor(value) and value.ndim:
                            assert value.shape == parameter.shape
            initialize_optimizer_state(model, opt)
        assert source.read_bytes() == original_bytes
        try:
            migrate_checkpoint(source, output)
        except FileExistsError:
            pass
        else:
            raise AssertionError("must not overwrite an existing output")
        try:
            migrate_checkpoint(output, root / "twice.safetensors")
        except ValueError:
            pass
        else:
            raise AssertionError("must reject an already migrated checkpoint")


class AreaConnectionsMigrationTest(unittest.TestCase):
    def test_migration_preserves_parameters_and_optimizer_history(self) -> None:
        for optimizer, generation_features, lookahead in (
            ("adam", True, 8),
            ("muon", True, 8),
            ("adam", False, 0),
            ("adam", False, 8),
            ("adam", True, 0),
        ):
            with self.subTest(
                optimizer=optimizer, generation_features=generation_features, lookahead=lookahead
            ):
                check_migration(optimizer, generation_features, lookahead)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
