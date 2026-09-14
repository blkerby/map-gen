import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.migrate_toilet_balance import append_failure_output, fit_head, migrate_optimizer_state
from scripts.balance_v18 import BalanceModelV18 as BalanceModel
from train import save_named_optimizer_checkpoint_state


def test_new_controller_failure_price_preserves_success_prices() -> None:
    model = BalanceModel(
        left_count=1, right_count=1, up_count=0, down_count=0,
        door_output_variant_idx=torch.tensor([0, 1]),
        door_room_idx=torch.tensor([0, 1]),
        door_variant_compatibility=torch.ones((2, 2), dtype=torch.bool),
        room_connection_variant_idx=torch.tensor([0, 1]),
        num_room_connection_variants=2,
        toilet_compatibility=torch.tensor([True, True]),
        hidden_width=4, num_layers=1,
    )
    tensors = {
        "balance_model.net.2.weight": torch.randn(16, 4),
        "balance_model.net.2.bias": torch.randn(16),
    }
    original = {key: value.clone() for key, value in tensors.items()}
    append_failure_output(tensors, model)
    for key, value in tensors.items():
        torch.testing.assert_close(value[:-1], original[key], rtol=0, atol=0)
    inputs = torch.randn(10, 4)
    raw = torch.nn.functional.linear(inputs, tensors["balance_model.net.2.weight"], tensors["balance_model.net.2.bias"])
    # Two pair outputs precede the two Toilet room outputs.
    torch.testing.assert_close(raw[:, -1], raw[:, 2:4].mean(-1))


def test_optimizer_resets_only_migrated_head_and_extends_controller_moments() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model(torch.randn(4, 3)).square().mean().backward()
    optimizer.step()
    tensors, metadata = {}, {}
    save_named_optimizer_checkpoint_state(tensors, metadata, optimizer, "optimizer")
    original = {key: value.clone() for key, value in tensors.items()}
    migrate_optimizer_state(tensors, metadata, model, optimizer, "optimizer", {"1.weight", "1.bias"}, set())
    for key, value in tensors.items():
        if ".state.0." in key or ".state.1." in key:
            torch.testing.assert_close(value, original[key], rtol=0, atol=0)
        else:
            assert torch.count_nonzero(value) == 0
    # Expanding an output retains its old moments and starts the new row at zero.
    model[0] = torch.nn.Linear(3, 3)
    model[1] = torch.nn.Linear(3, 1)
    expanded_optimizer = torch.optim.Adam(model[0].parameters(), lr=0.01)
    small_optimizer = torch.optim.Adam(torch.nn.Linear(3, 2).parameters(), lr=0.01)
    for param in small_optimizer.param_groups[0]["params"]:
        param.grad = torch.ones_like(param)
    small_optimizer.step()
    tensors, metadata = {}, {}
    save_named_optimizer_checkpoint_state(tensors, metadata, small_optimizer, "balance_optimizer")
    original = {key: value.clone() for key, value in tensors.items()}
    migrate_optimizer_state(tensors, metadata, model[0], expanded_optimizer, "balance_optimizer", set(), {"weight", "bias"})
    for key, value in tensors.items():
        if value.ndim:
            torch.testing.assert_close(value[:-1], original[key], rtol=0, atol=0)
            assert torch.count_nonzero(value[-1]) == 0
        else:
            torch.testing.assert_close(value, original[key], rtol=0, atol=0)


def test_head_calibration_recovers_linear_expected_cost() -> None:
    torch.manual_seed(91)
    x = torch.randn(2048, 8)
    head = torch.nn.Linear(8, 1)
    expected_weight = torch.randn(1, 8)
    target = (x @ expected_weight.T).flatten() + 0.2
    weight, bias, _ = fit_head(x[:1024], target[:1024], head, 1e-6)
    prediction = torch.nn.functional.linear(x[1024:], weight, bias).flatten()
    torch.testing.assert_close(prediction, target[1024:], atol=2e-5, rtol=2e-5)


if __name__ == "__main__":
    test_new_controller_failure_price_preserves_success_prices()
    test_optimizer_resets_only_migrated_head_and_extends_controller_moments()
    test_head_calibration_recovers_linear_expected_cost()
