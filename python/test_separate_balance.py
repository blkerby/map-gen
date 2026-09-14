import copy
import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env import AREA_COUNT, DoorMatches
from loss import compute_balance_loss, compute_balance_price_tables
from scripts.balance_v18 import BalanceModelV18 as BalanceModel
from train import load_named_optimizer_checkpoint_state, save_named_optimizer_checkpoint_state
from train_config import GENERATION_VARIABLE_FLOAT_FIELDS
from scripts.split_balance_controller import (
    shared_balance_model,
    split_shared_balance_state,
    verify_split_outputs,
)
from scripts.migrate_toilet_balance import append_failure_output, migrate_optimizer_state


def example_model() -> BalanceModel:
    return BalanceModel(
        left_count=1,
        right_count=1,
        up_count=0,
        down_count=0,
        door_output_variant_idx=torch.tensor([0, 1]),
        door_room_idx=torch.tensor([0, 1]),
        door_variant_compatibility=torch.ones((2, 2), dtype=torch.bool),
        room_connection_variant_idx=torch.tensor([0, 1, 2]),
        num_room_connection_variants=3,
        toilet_compatibility=torch.tensor([True, True, False]),
        hidden_width=8,
        num_layers=2,
    )


def checkpoint_fixture():
    torch.manual_seed(18)
    model = example_model()
    shared = shared_balance_model(model)
    with torch.no_grad():
        shared.net[-1].weight.normal_(0, 0.1)
        shared.net[-1].bias.normal_(0, 0.1)
    variables = torch.randn(4, len(GENERATION_VARIABLE_FLOAT_FIELDS))
    optimizer = torch.optim.Adam(shared.parameters(), lr=0.001)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        shared.net(variables).square().sum().backward()
        optimizer.step()
    tensors = {
        f"balance_model.{name}": value.clone()
        for name, value in model.state_dict().items()
        if not name.startswith(("door_net.", "toilet_net.", "area_net."))
    }
    tensors.update({f"balance_model.{name}": value.clone() for name, value in shared.state_dict().items()})
    tensors["main_model.unchanged"] = torch.randn(3)
    metadata = {}
    save_named_optimizer_checkpoint_state(tensors, metadata, optimizer, "balance_optimizer")
    return model, shared, variables, tensors, metadata


def test_migration_preserves_outputs_moments_and_independent_storage() -> None:
    model, shared, variables, tensors, metadata = checkpoint_fixture()
    old = {key: value.clone() for key, value in tensors.items()}
    split_shared_balance_state(tensors, metadata, model)
    verify_split_outputs(shared, model, variables)
    torch.testing.assert_close(tensors["main_model.unchanged"], old["main_model.unchanged"], rtol=0, atol=0)
    assert not any(key.startswith("balance_model.net.") for key in tensors)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    load_named_optimizer_checkpoint_state(optimizer, tensors, metadata, "balance_optimizer")
    for branch in (model.door_net, model.toilet_net, model.area_net):
        for name, param in branch[:-1].named_parameters():
            torch.testing.assert_close(param, dict(shared.net.named_parameters())[name], rtol=0, atol=0)
            source_id = list(dict(shared.named_parameters())).index(f"net.{name}")
            for state_name in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    optimizer.state[param][state_name],
                    old[f"balance_optimizer.adam.state.{source_id}.{state_name}"],
                    rtol=0, atol=0,
                )
    # Check the noncontiguous Toilet output mapping explicitly: rooms, then failure.
    rows = torch.tensor([2, 3, 4, 23])
    for suffix, source_id in (("weight", 4), ("bias", 5)):
        parameter = getattr(model.toilet_net[-1], suffix)
        torch.testing.assert_close(
            parameter, old[f"balance_model.net.4.{suffix}"][rows], rtol=0, atol=0
        )
        for state_name in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                optimizer.state[parameter][state_name],
                old[f"balance_optimizer.adam.state.{source_id}.{state_name}"][rows],
                rtol=0, atol=0,
            )
    weights = [network[0].weight for network in (model.door_net, model.toilet_net, model.area_net)]
    assert len({weight.data_ptr() for weight in weights}) == 3
    assert len({optimizer.state[weight]["exp_avg"].data_ptr() for weight in weights}) == 3
    # Missing source state is rejected by migration, rather than initialized silently.
    model, _, _, tensors, metadata = checkpoint_fixture()
    del tensors["balance_optimizer.adam.state.0.exp_avg"]
    try:
        split_shared_balance_state(tensors, metadata, model)
    except ValueError as error:
        assert "incomplete Adam state" in str(error)
    else:
        raise AssertionError("migration must reject incomplete Adam state")


def test_v16_failure_initialization_can_be_followed_by_branch_migration() -> None:
    model, shared, variables, tensors, metadata = checkpoint_fixture()
    for key, value in list(tensors.items()):
        if key.startswith("balance_model.net.4.") or (
            key.startswith(("balance_optimizer.adam.state.4.", "balance_optimizer.adam.state.5."))
            and value.ndim
        ):
            tensors[key] = value[:-1].clone()
    extended = append_failure_output(tensors, model)
    migrate_optimizer_state(
        tensors, metadata, shared, torch.optim.Adam(shared.parameters(), lr=0.001),
        "balance_optimizer", set(), extended,
    )
    split_shared_balance_state(tensors, metadata, model)
    preds = model(variables)
    tables = compute_balance_price_tables(
        preds,
        torch.full((4, 3, AREA_COUNT), 1.0 / AREA_COUNT),
        torch.ones((4, 3), dtype=torch.bool),
    )
    torch.testing.assert_close(tables.toilet_failure, torch.zeros(4), atol=1e-6, rtol=0)


def test_failure_learning_does_not_change_other_controller_updates() -> None:
    model, _, variables, tensors, metadata = checkpoint_fixture()
    split_shared_balance_state(tensors, metadata, model)
    models = [model, copy.deepcopy(model)]
    optimizers = [torch.optim.Adam(m.parameters(), lr=0.001) for m in models]
    for optimizer in optimizers:
        load_named_optimizer_checkpoint_state(optimizer, copy.deepcopy(tensors), metadata, "balance_optimizer")
    matches = DoorMatches(
        left=torch.zeros((4, 1), dtype=torch.int64),
        right=torch.zeros((4, 1), dtype=torch.int64),
        up=torch.empty((4, 0), dtype=torch.int64),
        down=torch.empty((4, 0), dtype=torch.int64),
    )
    probability = torch.full((4, 3, AREA_COUNT), 1.0 / AREA_COUNT)
    area_mask = torch.ones((4, 3), dtype=torch.bool)
    for _ in range(8):
        for include_failure, current, optimizer in zip((False, True), models, optimizers, strict=True):
            optimizer.zero_grad(set_to_none=True)
            preds = current(variables)
            if not include_failure:
                preds = replace(
                    preds,
                    toilet_failure=torch.zeros_like(preds.toilet_failure),
                )
            loss = compute_balance_loss(
                preds=preds,
                door_matches=matches,
                toilet_crossed_room_idx=torch.tensor([-1, -1, -1, 0]),
                room_area=torch.tensor([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]),
                area_probability=probability,
                area_dual_mask=area_mask,
                record_weight=torch.ones(4),
                door_beta=0.15,
                toilet_beta=0.2,
                area_beta=0.1,
            )
            loss.backward()
            optimizer.step()
    for branch in ("door_net", "area_net"):
        for name, parameter in getattr(models[0], branch).named_parameters():
            other = dict(getattr(models[1], branch).named_parameters())[name]
            torch.testing.assert_close(parameter, other, atol=0, rtol=0)
            for state_name in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    optimizers[0].state[parameter][state_name],
                    optimizers[1].state[other][state_name], atol=0, rtol=0,
                )
    failures = [
        compute_balance_price_tables(current(variables), probability, area_mask).toilet_failure
        for current in models
    ]
    assert (failures[1] - failures[0]).mean() > 0.001
    # Each price family's gradient reaches only its own network.
    current = models[1]
    for family in ("door", "toilet", "area"):
        current.zero_grad(set_to_none=True)
        preds = current(variables)
        outputs = {"door": preds.left, "toilet": preds.toilet_failure, "area": preds.room_area}
        outputs[family].square().sum().backward()
        for branch in ("door", "toilet", "area"):
            parameters = list(getattr(current, f"{branch}_net").parameters())
            assert all(parameter.grad is None for parameter in parameters) == (branch != family)


if __name__ == "__main__":
    test_migration_preserves_outputs_moments_and_independent_storage()
    test_v16_failure_initialization_can_be_followed_by_branch_migration()
    test_failure_learning_does_not_change_other_controller_updates()
