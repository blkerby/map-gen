import torch

from loss import balance_family_loss, balance_objective_terms, compute_balance_loss
from test_balance_variants import empty_door_matches, example_predictions, uniform_area_targets


def test_quartic_prices_include_failures_and_respect_masks_and_weights() -> None:
    successes = torch.tensor(
        [[[1.0, -1.0], [9.0, -9.0]], [[2.0, -2.0], [9.0, -9.0]]],
        requires_grad=True,
    )
    failures = torch.tensor([[1.0, 9.0], [2.0, 9.0]], requires_grad=True)
    terms = balance_objective_terms(
        successes,
        failures,
        outcome=torch.tensor([[0, -1], [-1, -1]]),
        enabled=torch.tensor([[True, False], [True, False]]),
    )
    loss = balance_family_loss(
        [terms], beta=0.5, price_scale=1.0, record_weight=torch.tensor([1.0, 3.0])
    )
    # Active prices are (1, -1, 1) and (2, -2, 2). Their regularization
    # costs are 1.125 and 9; observed costs are 1 and 2, respectively.
    torch.testing.assert_close(loss, torch.tensor(5.28125))
    loss.backward()
    torch.testing.assert_close(
        successes.grad,
        torch.tensor([[[0.0, -0.25], [0.0, 0.0]], [[3.75, -3.75], [0.0, 0.0]]]),
    )
    torch.testing.assert_close(failures.grad, torch.tensor([[0.25, 0.0], [3.0, 0.0]]))


def test_each_balance_family_uses_its_own_quartic_scale_for_failures() -> None:
    preds = example_predictions(requires_grad=True)
    with torch.no_grad():
        preds.door_failure.fill_(1.0)
        preds.toilet_failure.fill_(1.0)
        preds.room_area_failure.fill_(1.0)
        preds.area_order_failure.fill_(1.0)
    probability, mask = uniform_area_targets()
    loss = compute_balance_loss(
        preds=preds,
        door_matches=empty_door_matches(),
        toilet_crossed_room_idx=torch.tensor([-1]),
        room_area=torch.full((1, 2), -1, dtype=torch.int64),
        area_order=torch.full((1, 6), -1, dtype=torch.int64),
        area_probability=probability,
        area_dual_mask=mask,
        record_weight=torch.ones(1),
        door_beta=1.0,
        toilet_beta=1.0,
        area_beta=1.0,
        order_beta=1.0,
        door_price_scale=0.5,
        toilet_price_scale=1.0,
        area_price_scale=2.0,
        order_price_scale=4.0,
    )
    loss.backward()
    # At price=beta=1 with every group failing, the observed and quadratic
    # gradients cancel. Only 1 / (c^2 * number_of_groups) remains.
    torch.testing.assert_close(preds.door_failure.grad, torch.full((1, 8), 0.5))
    torch.testing.assert_close(preds.toilet_failure.grad, torch.ones(1))
    torch.testing.assert_close(preds.room_area_failure.grad, torch.full((1, 2), 0.125))
    torch.testing.assert_close(preds.area_order_failure.grad, torch.full((1, 6), 1 / 96))


if __name__ == "__main__":
    test_quartic_prices_include_failures_and_respect_masks_and_weights()
    test_each_balance_family_uses_its_own_quartic_scale_for_failures()
