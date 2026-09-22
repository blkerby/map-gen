import json
from pathlib import Path

import torch
from pydantic import ValidationError

from area_order import (
    candidate_order_balance_score,
    candidate_order_price,
    episode_area_order,
    proposal_order_price,
    remaining_order_price,
)
from env import Actions
from loss import balance_family_loss, balance_objective_terms, terminal_balance_cost
from train_config import Config, validate_config


def test_start_order_ignores_repeats_and_dummy_actions() -> None:
    actions = Actions(
        room_idx=torch.tensor([[0, 1, 9, 2, 3, 4, 5, 6]]),
        room_x=torch.zeros(1, 8),
        room_y=torch.zeros(1, 8),
        room_area=torch.tensor([[3, 3, 0, 1, 5, 1, 0, 255]]),
    )
    assert episode_area_order(actions, 7).tolist() == [[3, 1, 5, 0, -1, -1]]


def test_immediate_price_only_when_area_starts() -> None:
    prices = torch.arange(36.0).view(1, 6, 6)
    candidates = Actions(
        room_idx=torch.tensor([[0, 0, 0, 1]]),
        room_x=torch.zeros(1, 4),
        room_y=torch.zeros(1, 4),
        room_area=torch.tensor([[3, 3, 1, 255]]),
    )
    sizes = torch.tensor(
        [[[0, 0, 0, 2, 0, 0], [0, 0, 0, 4, 0, 0], [0, 2, 0, 4, 0, 0], [0, 0, 0, 4, 0, 0]]]
    )
    immediate = candidate_order_price(prices, sizes, candidates, torch.tensor([2]))
    torch.testing.assert_close(immediate, torch.tensor([[3.0, 0.0, 7.0, 0.0]]))
    # The future head describes the state after the candidate; add its cost once.
    total = candidate_order_balance_score(
        torch.full((1, 4), 10.0),
        prices,
        torch.ones(1, 6),
        sizes,
        torch.zeros(1, 4, 2, dtype=torch.bool),
        candidates,
        torch.tensor([2]),
    )
    torch.testing.assert_close(total, immediate + 10)


def test_future_target_excludes_all_started_ranks_and_includes_failure() -> None:
    prices = torch.arange(36.0).view(1, 6, 6)
    failures = torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]])
    terminal = terminal_balance_cost(prices, failures, torch.tensor([[3, 1, 5, -1, -1, -1]]))
    used = torch.tensor([[False, False, False, True, False, False]])
    torch.testing.assert_close(remaining_order_price(terminal, used), torch.tensor([174.0]))
    torch.testing.assert_close(
        remaining_order_price(terminal, torch.ones_like(used)), torch.zeros(1)
    )


def test_completed_episode_overrides_future_prediction() -> None:
    candidates = Actions(
        room_idx=torch.tensor([[0, 0]]),
        room_x=torch.zeros(1, 2),
        room_y=torch.zeros(1, 2),
        room_area=torch.tensor([[5, 1]]),
    )
    sizes = torch.tensor([[[2, 2, 2, 2, 2, 1], [3, 1, 0, 0, 0, 0]]])
    total = candidate_order_balance_score(
        torch.full((1, 2), 999.0),
        torch.ones(1, 6, 6),
        torch.full((1, 6), 5.0),
        sizes,
        torch.ones(1, 2, 3, dtype=torch.bool),
        candidates,
        torch.tensor([1]),
    )
    # Sixth area: just the immediate price. Two areas reached: four failure costs.
    torch.testing.assert_close(total, torch.tensor([[1.0, 21.0]]))


def test_proposal_cost_matches_next_rank_and_masks_used_areas() -> None:
    prices = torch.arange(36.0).view(1, 6, 6).expand(2, -1, -1)
    used = torch.tensor([[False, False, False, True, False, False], [True] * 6])
    torch.testing.assert_close(
        proposal_order_price(prices, used),
        torch.tensor([[6.0, 7.0, 8.0, 0.0, 10.0, 11.0], [0.0] * 6]),
    )


def test_controller_pushes_overrepresented_choices_and_failures_up() -> None:
    raw = torch.zeros(1, 6, 6, requires_grad=True)
    failures = torch.zeros(1, 6, requires_grad=True)
    centered = raw - raw.mean(-1, keepdim=True)
    observed = torch.tensor([[3, 1, 5, 0, -1, -1]])
    loss = balance_family_loss(
        [
            balance_objective_terms(
                centered, failures, observed, torch.ones(1, 6, dtype=torch.bool)
            )
        ],
        beta=1.0,
        price_scale=1.0,
        record_weight=torch.ones(1),
    )
    loss.backward()
    assert raw.grad[0, 0, 3] < 0  # Gradient descent raises the observed price.
    assert raw.grad[0, 0, 0] > 0
    assert failures.grad[0, 4] < 0
    assert failures.grad[0, 0] == 0
    torch.testing.assert_close(centered.mean(-1), torch.zeros(1, 6))


def test_order_config_is_required_and_validated() -> None:
    original = json.loads(Path("configs/debug.json").read_text())
    for section, field in (("balance_train", "order_beta"), ("train", "order_balance_weight")):
        data = json.loads(json.dumps(original))
        del data[section][field]
        try:
            Config.model_validate(data)
        except ValidationError:
            pass
        else:
            raise AssertionError(f"{section}.{field} must be required")
        data[section][field] = -1
        try:
            validate_config(Config.model_validate(data))
        except ValueError as error:
            assert field in str(error)
        else:
            raise AssertionError(f"{section}.{field} must reject negative values")
    original["features"]["area_state"] = False
    try:
        validate_config(Config.model_validate(original))
    except ValueError as error:
        assert "area_state" in str(error)
    else:
        raise AssertionError("order balancing requires area state features")


if __name__ == "__main__":
    test_start_order_ignores_repeats_and_dummy_actions()
    test_immediate_price_only_when_area_starts()
    test_future_target_excludes_all_started_ranks_and_includes_failure()
    test_completed_episode_overrides_future_prediction()
    test_proposal_cost_matches_next_rank_and_masks_used_areas()
    test_controller_pushes_overrepresented_choices_and_failures_up()
    test_order_config_is_required_and_validated()
