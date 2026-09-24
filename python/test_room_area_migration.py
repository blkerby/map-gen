from dataclasses import fields
import json
from pathlib import Path
import sys

import tempfile
import unittest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env import Actions, Engine, FeatureSlot
from model import FrontierModel
from model_loading import create_balance_model, frontier_model_kwargs, without_prefix
from scripts.migrate_room_area import SOURCE_FORMAT, TARGET_FORMAT, migrate_checkpoint
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


def make_source(root: Path, optimizer_type: str):
    rooms = [one_tile_room(f"Room {i}", "right" if i % 2 == 0 else "left") for i in range(6)]
    room_path = root / "rooms.json"
    room_path.write_text(json.dumps(rooms))
    data = json.loads(Path("configs/debug.json").read_text())
    data["room_set"] = str(room_path)
    data["features"]["room_area"] = False
    data["model"].update(embedding_width=8, global_embedding_width=8, hidden_width=16)
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
    else:
        data["optimizer"] = {"type": "adam", "lr": 0.001, "beta1": 0.9, "beta2": 0.99}
    config = instantiate_scheduleable_config(Config.model_validate(data), 256)
    engine = Engine(
        rooms, config.features, config.generation.min_area_size, config.generation.max_area_size
    )
    main = FrontierModel(**frontier_model_kwargs(config, rooms, engine))
    balance = create_balance_model(config, rooms, engine, torch.device("cpu"))
    optimizer = create_main_optimizer(main, config.optimizer)
    balance_optimizer = create_adam_optimizer(balance.parameters(), config.balance_optimizer)
    for model, opt in [(main, optimizer), (balance, balance_optimizer)]:
        for index, parameter in enumerate(model.parameters()):
            parameter.grad = torch.full_like(parameter, (index + 1) / 1000)
        opt.step()
        for part in named_checkpoint_optimizers(opt).values():
            part.zero_grad(set_to_none=True)
    tensors = prefixed_state_dict("main_model", main)
    tensors.update({k: v.clone() for k, v in prefixed_state_dict("ema_model", main).items()})
    tensors.update(prefixed_state_dict("balance_model", balance))
    del data["features"]["room_area"]
    metadata = dict(
        format=SOURCE_FORMAT,
        config=json.dumps(data),
        num_episodes="256",
        experience_num_files="17",
        aim_run_hash="room-area-test",
    )
    save_named_optimizer_checkpoint_state(tensors, metadata, optimizer, "optimizer")
    save_named_optimizer_checkpoint_state(
        tensors, metadata, balance_optimizer, "balance_optimizer"
    )
    path = root / "source.safetensors"
    save_file(tensors, path, metadata=metadata)
    return path, tensors, metadata, main, rooms


def extract_features(engine, config):
    env = engine.create_environment_group(
        map_size=config.map_size,
        num_envs=2,
        candidate_spatial_cell_size=config.generation.candidate_spatial_cell_size,
        area_bounding_box_width=config.generation.area_bounding_box_width,
        area_bounding_box_height=config.generation.area_bounding_box_height,
        seed=0,
        num_threads=2,
        frontier_neighbor_count=config.generation.frontier_neighbor_count,
        frontier_window_size=config.generation.frontier_window_size,
        frontier_neighbor_algorithm=config.generation.frontier_neighbor_algorithm,
    )
    env.step(
        Actions(
            room_idx=torch.tensor([0, 0], dtype=torch.uint8),
            room_x=torch.tensor([4, 4], dtype=torch.int8),
            room_y=torch.tensor([4, 4], dtype=torch.int8),
            room_area=torch.tensor([2, 4], dtype=torch.uint8),
        )
    )
    return env.extract_features(
        FeatureSlot(env=env, pin_memory=False),
        torch.zeros(2),
        config.features.temperature,
        torch.zeros(2),
        config.features.recommended_candidates,
        torch.zeros((2, len(GENERATION_VARIABLE_FLOAT_FIELDS))),
        config.features.generation_variable_floats,
        env.get_current_feature_outcomes(torch.device("cpu"), 0, 2),
        config.features.lookahead_outcomes,
        0,
        2,
    )


def check_migration(tmp_path: Path, optimizer_type: str):
    torch.manual_seed(12)
    torch.set_num_threads(2)
    source, old, old_meta, old_model, rooms = make_source(tmp_path, optimizer_type)
    output = tmp_path / "migrated.safetensors"
    report = migrate_checkpoint(source, output)
    with safe_open(output, framework="pt") as cp:
        metadata = cp.metadata()
        tensors = {k: cp.get_tensor(k) for k in cp.keys()}
    assert metadata["format"] == TARGET_FORMAT
    for key in old_meta:
        if key not in ("format", "config"):
            assert metadata[key] == old_meta[key]
    assert set(tensors) == set(old)
    expanded = set(report["expanded_tensors"])
    assert len(expanded) == (4 if optimizer_type == "adam" else 3)
    for key, before in old.items():
        after = tensors[key]
        if key in expanded:
            assert after.shape[1] == before.shape[1] + len(rooms) * 6
            torch.testing.assert_close(after[:, : before.shape[1]], before, rtol=0, atol=0)
            assert torch.count_nonzero(after[:, before.shape[1] :]) == 0
        else:
            torch.testing.assert_close(after, before, rtol=0, atol=0)
    config = instantiate_scheduleable_config(Config.model_validate_json(metadata["config"]), 256)
    assert config.features.room_area
    engine = Engine(
        rooms, config.features, config.generation.min_area_size, config.generation.max_area_size
    )
    features = extract_features(engine, config)
    old_model.eval()
    with torch.no_grad():
        before = old_model(features, return_proposal_state=True)
    for prefix in ["main_model", "ema_model"]:
        migrated = FrontierModel(**frontier_model_kwargs(config, rooms, engine))
        migrated.load_state_dict(without_prefix(tensors, prefix), strict=True)
        migrated.eval()
        with torch.no_grad():
            after = migrated(features, return_proposal_state=True)
        for field in fields(before):
            a, b = getattr(before, field.name), getattr(after, field.name)
            if a is not None:
                torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            old_model.proposal_output(before.proposal_state),
            migrated.proposal_output(after.proposal_state),
            rtol=1e-5,
            atol=1e-6,
        )
    migrated.load_state_dict(without_prefix(tensors, "main_model"), strict=True)
    optimizer = create_main_optimizer(migrated, config.optimizer)
    load_named_optimizer_checkpoint_state(optimizer, tensors, metadata, "optimizer")
    old_width = old_model.global_mlp.weight.shape[1]
    # A real model loss reaches the zero-initialized columns on the first update.
    migrated(features, return_proposal_state=False).phantoon_area_invalid.sum().backward()
    assert torch.count_nonzero(migrated.global_mlp.weight.grad[:, old_width:]) > 0
    optimizer.step()
    assert torch.count_nonzero(migrated.global_mlp.weight[:, old_width:]) > 0
    with unittest.TestCase().assertRaises(FileExistsError):
        migrate_checkpoint(source, output)
    with unittest.TestCase().assertRaisesRegex(ValueError, "expected.*v21"):
        migrate_checkpoint(output, tmp_path / "again.safetensors")
    # Normal loading remains strict; only the explicit migration fills the new field.
    with unittest.TestCase().assertRaisesRegex(ValueError, "room_area"):
        Config.model_validate_json(old_meta["config"])
    with safe_open(source, framework="pt") as cp:
        assert cp.metadata() == old_meta
        for key in cp.keys():
            torch.testing.assert_close(cp.get_tensor(key), old[key], rtol=0, atol=0)


class RoomAreaMigrationTest(unittest.TestCase):
    def test_adam_migration(self):
        with tempfile.TemporaryDirectory() as directory:
            check_migration(Path(directory), "adam")

    def test_muon_migration(self):
        with tempfile.TemporaryDirectory() as directory:
            check_migration(Path(directory), "muon")


if __name__ == "__main__":
    unittest.main()
