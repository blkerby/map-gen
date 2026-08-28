import json
import sys
import tempfile
from pathlib import Path

import safetensors.torch
import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.add_balance_ema_to_checkpoint import (  # noqa: E402
    SOURCE_FORMAT,
    add_balance_ema_to_checkpoint,
)
from train import TRAINING_CHECKPOINT_FORMAT  # noqa: E402


def test_add_balance_ema_to_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config_path = root / "config.json"
        input_path = root / "input.safetensors"
        output_path = root / "output.safetensors"
        config_path.write_text(Path("configs/debug.json").read_text())
        safetensors.torch.save_file(
            {
                "main_model.weight": torch.tensor([1.0]),
                "balance_model.weight": torch.tensor([2.0]),
                "balance_optimizer.state": torch.tensor([3.0]),
            },
            input_path,
            metadata={
                "format": SOURCE_FORMAT,
                "config": "old config",
                "aim_run_hash": "run",
                "num_episodes": "256",
                "experience_num_files": "1",
            },
        )

        add_balance_ema_to_checkpoint(config_path, input_path, output_path)

        with safe_open(output_path, framework="pt", device="cpu") as checkpoint:
            metadata = checkpoint.metadata()
            balance_model = checkpoint.get_tensor("balance_model.weight")
            balance_ema_model = checkpoint.get_tensor("balance_ema_model.weight")
        assert metadata is not None
        assert metadata["format"] == TRAINING_CHECKPOINT_FORMAT
        assert (
            json.loads(metadata["config"])["balance_train"]["ema_half_life_episodes"]
            == 35400.338684
        )
        torch.testing.assert_close(balance_model, balance_ema_model)


if __name__ == "__main__":
    test_add_balance_ema_to_checkpoint()
