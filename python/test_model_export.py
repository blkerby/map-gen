import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import safetensors.torch
import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.export_model import export_model_tensors  # noqa: E402
from env import Engine  # noqa: E402
from generate import GenerationProfiler  # noqa: E402
from model import (  # noqa: E402
    Float32Linear,
    FrontierModel,
    MissingConnectQueryHead,
    ProposalOutput,
    SaveRefillUtilityQueryHead,
)
from model_loading import MODEL_EXPORT_FORMAT, create_balance_model, frontier_model_kwargs  # noqa: E402
from serve import (  # noqa: E402
    ServingConfig,
    create_generate_configs,
    create_serving_state,
    load_model_input,
    run_foreground_generation,
    warmup_generate_request,
)
from test_room_area_plumbing import one_tile_room  # noqa: E402
from train import (  # noqa: E402
    TRAINING_CHECKPOINT_FORMAT,
    TrainingSession,
    create_adam_optimizer,
    create_main_optimizer,
    named_checkpoint_optimizers,
)
from train_config import (  # noqa: E402
    Config,
    VANILLA_AREA_CONDITION_FIELDS,
    instantiate_scheduleable_config,
)


def step_optimizers(session: TrainingSession, gradient_scale: float) -> None:
    for model, optimizer in (
        (session.main_model, session.main_optimizer),
        (session.balance_model, session.balance_optimizer),
    ):
        for index, parameter in enumerate(model.parameters()):
            parameter.grad = torch.full_like(parameter, gradient_scale * (index + 1) / 1000)
        optimizer.step()
        for part in named_checkpoint_optimizers(optimizer).values():
            part.zero_grad(set_to_none=True)


class ModelExportTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.rooms = [one_tile_room(f"Room {i}", "left" if i % 2 else "right") for i in range(6)]
        room_path = self.root / "rooms.json"
        room_path.write_text(json.dumps(self.rooms))
        data = json.loads(Path("configs/debug.json").read_text())
        data["room_set"] = str(room_path)
        data["map_size"] = [8, 8]
        data["model"].update(
            embedding_width=8,
            global_embedding_width=8,
            hidden_width=16,
            proposal_hidden_widths=[8],
            missing_connect_query_hidden_width=8,
            missing_connect_query_frontier_width=4,
            missing_connect_query_distance_width=2,
            utility_query_hidden_width=8,
            utility_query_frontier_width=4,
        )
        data["balance_model"].update(hidden_width=8, num_layers=2)
        for name, value in data["features"].items():
            if type(value) is int and value > 0:
                data["features"][name] = 2
        data["features"].update(save_utility_query=True, refill_utility_query=True)
        data["generation"].update(num_threads=1, pipeline_groups=1)
        data["train"]["pipeline_groups"] = 1
        self.config = instantiate_scheduleable_config(Config.model_validate(data), 256)
        self.engine = Engine(
            self.rooms,
            self.config.features,
            self.config.generation.min_area_size,
            self.config.generation.max_area_size,
        )
        torch.manual_seed(37)
        self.session = self.make_session()
        # Nonzero, unrounded weights expose accidental bfloat16 conversion of heads.
        with torch.no_grad():
            for model in (
                self.session.main_model,
                self.session.ema_model,
                self.session.balance_model,
            ):
                for parameter in model.parameters():
                    parameter.uniform_(-0.3, 0.3)
        step_optimizers(self.session, gradient_scale=1.0)
        self.checkpoint_path = self.root / "checkpoint.safetensors"
        self.session.save_checkpoint(self.checkpoint_path)
        self.export_path = self.root / "model.safetensors"

    def make_session(self) -> TrainingSession:
        main_model = FrontierModel(**frontier_model_kwargs(self.config, self.rooms, self.engine))
        balance_model = create_balance_model(
            self.config, self.rooms, self.engine, torch.device("cpu")
        )
        session = TrainingSession(
            args=Mock(),
            config=self.config,
            run_path=str(self.root),
            rooms=self.rooms,
            device=torch.device("cpu"),
            generation_devices=[torch.device("cpu")],
            engine=self.engine,
            train_batch_envs=[],
            main_model=main_model,
            ema_model=copy.deepcopy(main_model),
            balance_model=balance_model,
            main_optimizer=create_main_optimizer(main_model, self.config.optimizer),
            balance_optimizer=create_adam_optimizer(
                balance_model.parameters(), self.config.balance_optimizer
            ),
            loss_config=Mock(),
            experience=SimpleNamespace(num_files=3),
            train_batch_prefetcher=Mock(),
            generation_executors=[],
            num_episodes=256,
        )
        session.aim_run = SimpleNamespace(hash="synthetic-export-test")
        return session

    def make_serving_state(self, path: Path, device: str):
        config = ServingConfig.model_validate_json(
            Path("configs/zebes_serving_cpu.json").read_text()
        ).model_copy(
            update={
                "room_set": Path(self.config.room_set),
                "device": device,
                "compile_model": False,
                "num_environments": 2,
                "num_threads": 1,
                "gpu_prefetch_batches": 0,
                "autocast": True,
            }
        )
        return create_serving_state(config, load_model_input(path), seed=0, profile=False)

    def test_mixed_export_preserves_float32_prediction_weights(self) -> None:
        export_model_tensors(
            self.checkpoint_path, self.export_path, backbone_dtype="bfloat16", overwrite=False
        )
        loaded = load_model_input(self.export_path)
        self.assertEqual(loaded.backbone_dtype, torch.bfloat16)
        with safe_open(self.export_path, framework="pt", device="cpu") as exported:
            self.assertEqual(exported.metadata()["format"], MODEL_EXPORT_FORMAT)
            self.assertEqual(exported.metadata()["backbone_dtype"], "bfloat16")
        self.assertFalse(any(name.startswith("main_model.") for name in loaded.tensors))
        self.assertFalse(any(name.startswith("optimizer.") for name in loaded.tensors))
        for name in (
            "frontier_node_mlp.weight",
            "source_message_layers.0.weight",
            "pair_message_layers.0.weight",
            "message_output_layers.0.1.weight",
            "update_layers.0.0.weight",
            "global_mlp.weight",
        ):
            self.assertEqual(loaded.tensors[f"ema_model.{name}"].dtype, torch.bfloat16)
        for family in ("door", "toilet", "area", "order", "step"):
            for index in (0, 2):
                self.assertEqual(
                    loaded.tensors[f"balance_model.{family}_net.{index}.weight"].dtype,
                    torch.bfloat16,
                )
        protected_types = (
            Float32Linear,
            ProposalOutput,
            MissingConnectQueryHead,
            SaveRefillUtilityQueryHead,
        )
        for prefix, model in (
            ("ema_model", self.session.ema_model),
            ("balance_model", self.session.balance_model),
        ):
            for module_name, module in model.named_modules():
                if isinstance(module, protected_types) or module_name == "pooled_mlp":
                    for name, original in module.state_dict().items():
                        saved = loaded.tensors[f"{prefix}.{module_name}.{name}"]
                        self.assertEqual(saved.dtype, torch.float32)
                        torch.testing.assert_close(saved, original, rtol=0, atol=0)
                        if original.numel() > 0:
                            self.assertFalse(torch.equal(original, original.bfloat16().float()))
            for name, original in model.state_dict().items():
                saved = loaded.tensors[f"{prefix}.{name}"]
                torch.testing.assert_close(saved, original.to(saved.dtype), rtol=0, atol=0)
        state = self.make_serving_state(self.export_path, "cpu")
        for prefix, model in (("ema_model", state.model), ("balance_model", state.balance_model)):
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(
                    tensor, loaded.tensors[f"{prefix}.{name}"], rtol=0, atol=0
                )
            self.assertFalse(model.training)
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        for direction in ("left", "right", "up", "down"):
            self.assertEqual(
                getattr(state.balance_model, f"{direction}_probability").dtype, torch.float32
            )

    def test_float32_export_and_training_checkpoint_load_without_conversion(self) -> None:
        export_model_tensors(
            self.checkpoint_path, self.export_path, backbone_dtype="float32", overwrite=False
        )
        checkpoint = load_model_input(self.checkpoint_path)
        exported = load_model_input(self.export_path)
        self.assertEqual(checkpoint.backbone_dtype, torch.float32)
        self.assertEqual(exported.backbone_dtype, torch.float32)
        self.assertEqual(set(checkpoint.tensors), set(exported.tensors))
        for name, tensor in checkpoint.tensors.items():
            torch.testing.assert_close(exported.tensors[name], tensor, rtol=0, atol=0)
        state = self.make_serving_state(self.checkpoint_path, "cpu")
        self.assertTrue(all(p.dtype == torch.float32 for p in state.model.parameters()))
        self.assertTrue(all(p.dtype == torch.float32 for p in state.balance_model.parameters()))

    def check_generation(self, device: str, compiled: bool) -> None:
        export_model_tensors(
            self.checkpoint_path, self.export_path, backbone_dtype="bfloat16", overwrite=False
        )
        state = self.make_serving_state(self.export_path, device)
        if compiled:
            backend = "inductor" if device.startswith("cuda") else "aot_eager"
            state.model = torch.compile(state.model, backend=backend)
            state.balance_model = torch.compile(state.balance_model, backend=backend)
        request = warmup_generate_request().model_copy(
            update={
                **dict.fromkeys(VANILLA_AREA_CONDITION_FIELDS, False),
                "episode_length": 3,
                "recommended_candidates": 2,
                "shortlist_candidates": 2,
                "num_scored_invalid_candidates": 1,
                "guarantee_phantoon_counterpart": False,
            }
        )
        configs = create_generate_configs(request, state, state.envs, state.device)
        result = run_foreground_generation(state, configs, GenerationProfiler(False))
        self.assertEqual(result.episode_data.actions.room_idx.shape, (2, 3))

    def test_mixed_export_generates_eager(self) -> None:
        self.check_generation("cpu", compiled=False)

    def test_mixed_export_generates_compiled(self) -> None:
        self.check_generation("cpu", compiled=True)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_mixed_export_generates_cuda_compiled(self) -> None:
        self.check_generation("cuda:0", compiled=True)

    def test_export_rejects_missing_or_inconsistent_precision(self) -> None:
        export_model_tensors(
            self.checkpoint_path, self.export_path, backbone_dtype="bfloat16", overwrite=False
        )
        with safe_open(self.export_path, framework="pt", device="cpu") as exported:
            metadata = exported.metadata()
            tensors = {name: exported.get_tensor(name) for name in exported.keys()}
        corrupt_path = self.root / "corrupt.safetensors"
        no_precision = dict(metadata)
        del no_precision["backbone_dtype"]
        safetensors.torch.save_file(tensors, corrupt_path, metadata=no_precision)
        with self.assertRaisesRegex(ValueError, "backbone_dtype.*missing"):
            load_model_input(corrupt_path)
        for name, wrong_dtype in (
            ("ema_model.door_output.weight", torch.bfloat16),
            ("ema_model.pooled_mlp.0.weight", torch.bfloat16),
            ("ema_model.global_mlp.weight", torch.float32),
            ("balance_model.door_net.4.weight", torch.bfloat16),
        ):
            with self.subTest(name=name):
                corrupt = dict(tensors)
                corrupt[name] = tensors[name].to(wrong_dtype)
                safetensors.torch.save_file(corrupt, corrupt_path, metadata=metadata)
                with self.assertRaisesRegex(ValueError, "inference tensor.*dtype"):
                    self.make_serving_state(corrupt_path, "cpu")

    def test_export_does_not_change_training_checkpoint_or_resume(self) -> None:
        original_hash = hashlib.sha256(self.checkpoint_path.read_bytes()).digest()
        export_model_tensors(
            self.checkpoint_path, self.export_path, backbone_dtype="bfloat16", overwrite=False
        )
        self.assertEqual(hashlib.sha256(self.checkpoint_path.read_bytes()).digest(), original_hash)
        resumed = self.make_session()
        original_parameters = list(resumed.main_model.parameters())
        metadata = resumed.load_checkpoint(self.checkpoint_path)
        self.assertEqual(metadata["format"], TRAINING_CHECKPOINT_FORMAT)
        self.assertEqual(TRAINING_CHECKPOINT_FORMAT, "map-gen-training-session-checkpoint-v25")
        self.assertEqual(resumed.num_episodes, self.session.num_episodes)
        self.assertEqual(resumed.experience.num_files, self.session.experience.num_files)
        self.assertTrue(
            all(
                before is after
                for before, after in zip(
                    original_parameters, resumed.main_model.parameters(), strict=True
                )
            )
        )
        for before, after in (
            (self.session.main_model, resumed.main_model),
            (self.session.ema_model, resumed.ema_model),
            (self.session.balance_model, resumed.balance_model),
        ):
            actual = after.state_dict()
            self.assertEqual(set(before.state_dict()), set(actual))
            for name, expected in before.state_dict().items():
                torch.testing.assert_close(actual[name], expected, rtol=0, atol=0)
            self.assertTrue(all(p.dtype == torch.float32 for p in after.parameters()))
        # Matching the next update checks that optimizer moments and counters survived too.
        step_optimizers(self.session, gradient_scale=-0.5)
        step_optimizers(resumed, gradient_scale=-0.5)
        for before, after in (
            (self.session.main_model, resumed.main_model),
            (self.session.balance_model, resumed.balance_model),
        ):
            actual = after.state_dict()
            for name, expected in before.state_dict().items():
                torch.testing.assert_close(actual[name], expected, rtol=0, atol=0)

    def test_export_cannot_overwrite_training_checkpoint(self) -> None:
        original = self.checkpoint_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "must not overwrite the training checkpoint"):
            export_model_tensors(
                self.checkpoint_path,
                self.checkpoint_path,
                backbone_dtype="bfloat16",
                overwrite=True,
            )
        self.assertEqual(self.checkpoint_path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
