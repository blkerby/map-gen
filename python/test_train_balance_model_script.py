import argparse
import tempfile
import unittest
import sys
from pathlib import Path

import safetensors.torch
import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.train_balance_model import (  # noqa: E402
    TRAINING_CHECKPOINT_FORMAT,
    interpolate_schedule_episode,
    parse_ema_beta,
    write_checkpoint,
)


class TrainBalanceModelScriptTest(unittest.TestCase):
    def test_interpolate_schedule_episode(self) -> None:
        self.assertEqual(interpolate_schedule_episode(0, 100, 600), 0)
        self.assertEqual(interpolate_schedule_episode(50, 100, 600), 300)
        self.assertEqual(interpolate_schedule_episode(100, 100, 600), 600)

    def test_parse_ema_beta(self) -> None:
        self.assertEqual(parse_ema_beta("0"), 0.0)
        self.assertEqual(parse_ema_beta("0.99"), 0.99)
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_ema_beta("1")

    def test_write_checkpoint_replaces_balance_components(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.safetensors"
            output_path = Path(directory) / "output.safetensors"
            safetensors.torch.save_file(
                {
                    "main_model.weight": torch.ones(1),
                    "ema_model.weight": torch.ones(1),
                    "optimizer.adam.state.0.exp_avg": torch.ones(1),
                    "balance_model.old": torch.ones(1),
                    "balance_optimizer.adam.state.0.old": torch.ones(1),
                },
                input_path,
                metadata={
                    "format": TRAINING_CHECKPOINT_FORMAT,
                    "config": "old config",
                    "aim_run_hash": "run",
                    "num_episodes": "1",
                    "experience_num_files": "1",
                    "optimizer_names": '["adam"]',
                    "balance_optimizer_names": '["old"]',
                    "balance_optimizer_old": "old metadata",
                },
            )
            model = torch.nn.Linear(1, 1)
            ema_model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            model(torch.ones([1, 1])).sum().backward()
            optimizer.step()

            write_checkpoint(
                input_path,
                output_path,
                "new config",
                model,
                ema_model,
                optimizer,
            )

            with safe_open(output_path, framework="pt", device="cpu") as checkpoint:
                names = set(checkpoint.keys())
                metadata = checkpoint.metadata()
            self.assertIn("main_model.weight", names)
            self.assertIn("balance_model.weight", names)
            self.assertIn("balance_ema_model.weight", names)
            self.assertNotIn("balance_model.old", names)
            self.assertFalse(any(name.endswith(".old") for name in names))
            self.assertIsNotNone(metadata)
            self.assertEqual(metadata["config"], "new config")
            self.assertEqual(metadata["balance_optimizer_names"], '["adam"]')
            self.assertNotIn("balance_optimizer_old", metadata)


if __name__ == "__main__":
    unittest.main()
