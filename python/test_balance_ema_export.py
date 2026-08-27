import sys
import tempfile
from pathlib import Path

import safetensors.torch
import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.export_model import export_model_tensors  # noqa: E402
from serve import load_model_input  # noqa: E402
from train import TRAINING_CHECKPOINT_FORMAT  # noqa: E402


def test_export_and_serving_use_balance_ema() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint_path = root / "checkpoint.safetensors"
        export_path = root / "model.safetensors"
        config_json = Path("configs/debug.json").read_text()
        safetensors.torch.save_file(
            {
                "ema_model.weight": torch.tensor([1.0]),
                "balance_model.weight": torch.tensor([2.0]),
                "balance_ema_model.weight": torch.tensor([3.0]),
            },
            checkpoint_path,
            metadata={
                "format": TRAINING_CHECKPOINT_FORMAT,
                "config": config_json,
                "aim_run_hash": "run",
                "num_episodes": "256",
            },
        )

        export_model_tensors(checkpoint_path, export_path, overwrite=False)

        with safe_open(export_path, framework="pt", device="cpu") as exported:
            assert "balance_ema_model.weight" not in exported.keys()
            torch.testing.assert_close(
                exported.get_tensor("balance_model.weight"),
                torch.tensor([3.0]),
            )
        loaded_checkpoint = load_model_input(checkpoint_path)
        torch.testing.assert_close(
            loaded_checkpoint.tensors["balance_model.weight"],
            torch.tensor([3.0]),
        )


if __name__ == "__main__":
    test_export_and_serving_use_balance_ema()
