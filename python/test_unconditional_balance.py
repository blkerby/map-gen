import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env import AREA_COUNT
from loss import (
    balance_family_loss,
    balance_objective_terms,
    compute_balance_loss,
    compute_balance_price_tables,
    compute_balance_score_targets,
    compute_room_area_balance_score_targets,
    compute_step_balance_score_targets,
    terminal_balance_cost,
)
from scripts.balance_v18 import BalanceModelV19 as BalanceModel
from scripts.balance_v18 import BalanceModelV18
from scripts.migrate_toilet_balance import migrate_optimizer_state
from scripts.unconditional_balance import fit_cost_head, migrate_controller
from test_balance_variants import empty_door_matches, example_predictions, uniform_area_targets
from train import save_named_optimizer_checkpoint_state
from train_config import GENERATION_VARIABLE_FLOAT_FIELDS


def test_terminal_failures_train_all_families() -> None:
    preds = example_predictions(requires_grad=True)
    probability, mask = uniform_area_targets()
    loss = compute_balance_loss(
        area_order=torch.full((preds.room_area.shape[0], 6), -1, dtype=torch.int64),
        order_beta=1.0,
        door_price_scale=1.0,
        toilet_price_scale=1.0,
        area_price_scale=1.0,
        order_price_scale=1.0,
        preds=preds, door_matches=empty_door_matches(),
        toilet_crossed_room_idx=torch.tensor([-1]), room_area=torch.tensor([[-1, -1]]),
        area_probability=probability, area_dual_mask=mask, record_weight=torch.ones(1),
        door_beta=0.2, toilet_beta=0.3, area_beta=0.4,
    )
    loss.backward()
    torch.testing.assert_close(preds.door_failure.grad, torch.full((1, 8), -1 / 8))
    torch.testing.assert_close(preds.room_area_failure.grad, torch.full((1, 2), -1 / 2))
    torch.testing.assert_close(preds.toilet_failure.grad, torch.tensor([-1.0]))


def test_failure_has_zero_target_and_fixed_normalization() -> None:
    # Two groups, one always fails. beta=0.5 and c=1 give equilibrium f=1:
    # beta * f * (1 + (f / c)^2) = p_failure = 1. The other group's success
    # frequency does not change its denominator.
    successes = torch.zeros((4, 2, 2), requires_grad=True)
    failure = torch.full((4, 2), 1.0, requires_grad=True)
    outcomes = torch.tensor([[-1, 0], [-1, 0], [-1, 0], [-1, -1]])
    terms = balance_objective_terms(successes, failure, outcomes, torch.ones((4, 2), dtype=torch.bool))
    loss = balance_family_loss([terms], beta=0.5, price_scale=1.0, record_weight=torch.ones(4))
    loss.backward()
    torch.testing.assert_close(failure.grad.sum(0), torch.tensor([0.0, 0.375]))
    assert terms.group_count.tolist() == [2, 2, 2, 2]


def test_known_failures_and_unknown_costs_are_distinct() -> None:
    preds = example_predictions(requires_grad=True)
    preds.door_failure = torch.arange(1.0, 9.0).unsqueeze(0).requires_grad_()
    preds.room_area_failure = torch.tensor([[9.0, 10.0]], requires_grad=True)
    probability, mask = uniform_area_targets()
    tables = compute_balance_price_tables(preds, probability, mask)
    terminal = compute_balance_score_targets(tables, empty_door_matches())
    torch.testing.assert_close(terminal, preds.door_failure)
    assert not terminal.requires_grad
    areas = compute_room_area_balance_score_targets(tables, torch.tensor([[-1, 0]]))
    torch.testing.assert_close(areas, torch.tensor([[9.0, 0.0]]))
    assert not areas.requires_grad
    # Three right/left partners and one vertical partner. -1 is unknown;
    # the partner count marks a known failure, which must override predictions.
    step_match = torch.tensor([[[3, -1, 0, 3, -1, 2, 1, -1]]])
    costs, known = compute_step_balance_score_targets(tables, step_match)
    torch.testing.assert_close(costs[0, 0, [0, 3, 6]], torch.tensor([1.0, 4.0, 7.0]))
    assert known.tolist() == [[[True, False, True, True, False, True, True, False]]]
    assert not costs.requires_grad
    # A direction with no possible targets can still represent terminal failure.
    empty = terminal_balance_cost(torch.empty((1, 2, 0)), torch.tensor([[2.0, 3.0]]), torch.tensor([[-1, -1]]))
    torch.testing.assert_close(empty, torch.tensor([[2.0, 3.0]]))


def test_disabled_groups_have_no_failure_price_or_gradient() -> None:
    preds = example_predictions(requires_grad=True)
    preds.left_compatibility[0] = False
    probability, mask = uniform_area_targets()
    mask[0, 1] = False
    tables = compute_balance_price_tables(preds, probability, mask)
    (tables.door_failure.sum() + tables.room_area_failure.sum()).backward()
    assert preds.door_failure.grad[0, 0] == 0
    assert preds.room_area_failure.grad.tolist() == [[1.0, 0.0]]


def model_kwargs():
    return dict(
        left_count=2, right_count=2, up_count=0, down_count=0,
        door_output_variant_idx=torch.tensor([0, 0, 1, 1]),
        door_room_idx=torch.tensor([0, 1, 2, 2]),
        door_variant_compatibility=torch.ones((2, 2), dtype=torch.bool),
        room_connection_variant_idx=torch.tensor([0, 0, 1]),
        num_room_connection_variants=2,
        toilet_compatibility=torch.tensor([True, True, False]),
        hidden_width=8, num_layers=2,
    )


def test_migration_preserves_prices_hidden_layers_and_moments() -> None:
    torch.manual_seed(19)
    old = BalanceModelV18(**model_kwargs())
    with torch.no_grad():
        for branch in (old.door_net, old.toilet_net, old.area_net):
            branch[-1].weight.normal_()
            branch[-1].bias.normal_()
    optimizer = torch.optim.Adam(old.parameters(), lr=0.001)
    variables = torch.randn(16, len(GENERATION_VARIABLE_FLOAT_FIELDS))
    for branch in (old.door_net, old.toilet_net, old.area_net):
        branch(variables).square().sum().backward()
    optimizer.step()
    tensors = {f"balance_model.{name}": value.clone() for name, value in old.state_dict().items()}
    metadata = {}
    save_named_optimizer_checkpoint_state(tensors, metadata, optimizer, "balance_optimizer")
    new = BalanceModel(**model_kwargs())
    reset, extended = migrate_controller(tensors, new)
    new_optimizer = torch.optim.Adam(new.parameters(), lr=0.001)
    migrate_optimizer_state(tensors, metadata, new, new_optimizer, "balance_optimizer", reset, extended)
    before, after = old(variables), new(variables)
    probability = torch.rand((16, 3, AREA_COUNT))
    probability /= probability.sum(-1, keepdim=True)
    mask = torch.rand((16, 3)) > 0.2
    before_tables = compute_balance_price_tables(before, probability, mask)
    after_tables = compute_balance_price_tables(after, probability, mask)
    for name in ("left", "right", "up", "down", "toilet_crossed_room", "toilet_failure", "room_area"):
        torch.testing.assert_close(getattr(before_tables, name), getattr(after_tables, name))
    assert after.door_failure.count_nonzero() == after.room_area_failure.count_nonzero() == 0
    old_parameters = dict(old.named_parameters())
    for name, parameter in new.named_parameters():
        source = old_parameters[name]
        if name in reset:
            assert all(state.count_nonzero() == 0 for state in new_optimizer.state[parameter].values())
        else:
            torch.testing.assert_close(parameter[:source.shape[0]], source, atol=0, rtol=0)
            for key, value in new_optimizer.state[parameter].items():
                if value.ndim:
                    torch.testing.assert_close(value[:source.shape[0]], optimizer.state[source][key], atol=0, rtol=0)
                    assert value[source.shape[0]:].count_nonzero() == 0
                else:
                    torch.testing.assert_close(value, optimizer.state[source][key], atol=0, rtol=0)
    # Failure rows preserve variant sharing and gradients stay in their branch.
    for family, output in (("door", after.door_failure), ("toilet", after.toilet_failure), ("area", after.room_area_failure)):
        new.zero_grad(set_to_none=True)
        output.sum().backward(retain_graph=True)
        for branch in ("door", "toilet", "area"):
            assert all(p.grad is None for p in getattr(new, branch + "_net").parameters()) == (branch != family)
    with torch.no_grad():
        new.door_net[-1].bias[-2:] = torch.tensor([2.0, 3.0])
        new.area_net[-1].bias[-2:] = torch.tensor([4.0, 5.0])
    after = new(variables)
    torch.testing.assert_close(after.door_failure[0], torch.tensor([2.0, 2.0, 3.0, 3.0]))
    torch.testing.assert_close(after.room_area_failure[0], torch.tensor([4.0, 4.0, 5.0]))


def test_calibration_changes_only_final_cost_layer() -> None:
    torch.manual_seed(20)
    x = torch.randn(1000, 8)
    head = torch.nn.Linear(8, 3)
    weight, bias, scale = fit_cost_head(x, head(x).detach() * torch.tensor([0.2, 0.5, 0.8]), head, 0.01)
    torch.testing.assert_close(weight, head.weight * torch.tensor([[0.2], [0.5], [0.8]]))
    torch.testing.assert_close(bias, head.bias * torch.tensor([0.2, 0.5, 0.8]))
    torch.testing.assert_close(scale, torch.tensor([0.2, 0.5, 0.8]))


if __name__ == "__main__":
    test_terminal_failures_train_all_families()
    test_failure_has_zero_target_and_fixed_normalization()
    test_known_failures_and_unknown_costs_are_distinct()
    test_disabled_groups_have_no_failure_price_or_gradient()
    test_migration_preserves_prices_hidden_layers_and_moments()
    test_calibration_changes_only_final_cost_layer()
