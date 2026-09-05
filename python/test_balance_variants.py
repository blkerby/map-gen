from types import SimpleNamespace
from unittest.mock import Mock

import torch

from env import AREA_COUNT, DoorMatches, compute_area_balance_targets
from loss import (
    compute_balance_loss,
    compute_balance_price_tables,
    compute_proposal_area_balance_score_residual,
    compute_proposal_area_balance_score_table,
    compute_proposal_balance_score_residual,
    compute_proposal_balance_score_table,
)
from model import BalanceModel, BalancePredictions
from model_loading import create_balance_model
from train_config import GENERATION_VARIABLE_FLOAT_FIELDS


def example_door_variant_compatibility() -> torch.Tensor:
    compatibility = torch.zeros((6, 6), dtype=torch.bool)
    for source, target in (
        (0, 2),
        (1, 2),
        (1, 3),
        (2, 0),
        (2, 1),
        (3, 1),
        (4, 5),
        (5, 4),
    ):
        compatibility[source, target] = True
    return compatibility


def example_predictions(requires_grad: bool = False) -> BalancePredictions:
    return BalancePredictions(
        left=torch.tensor([[[0.0, 1.0], [2.0, -1.0]]], requires_grad=requires_grad),
        right=torch.tensor([[[0.5, -0.5], [1.0, 0.0]]], requires_grad=requires_grad),
        up=torch.zeros((1, 1, 1), requires_grad=requires_grad),
        down=torch.zeros((1, 1, 1), requires_grad=requires_grad),
        toilet_crossed_room=torch.zeros((1, 2), requires_grad=requires_grad),
        room_area=torch.zeros((1, 2, AREA_COUNT), requires_grad=requires_grad),
        left_door_variant_idx=torch.tensor([0, 0, 1]),
        right_door_variant_idx=torch.tensor([0, 1, 1]),
        up_door_variant_idx=torch.tensor([0]),
        down_door_variant_idx=torch.tensor([0]),
        left_global_door_variant_idx=torch.tensor([0, 1]),
        right_global_door_variant_idx=torch.tensor([2, 3]),
        up_global_door_variant_idx=torch.tensor([4]),
        down_global_door_variant_idx=torch.tensor([5]),
        door_variant_compatibility=example_door_variant_compatibility(),
        toilet_compatibility=torch.tensor([True, False]),
    )


def uniform_area_targets(batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.full((batch_size, 2, AREA_COUNT), 1.0 / AREA_COUNT),
        torch.ones((batch_size, 2), dtype=torch.bool),
    )


def empty_door_matches() -> DoorMatches:
    return DoorMatches(
        left=torch.full((1, 3), -1),
        right=torch.full((1, 3), -1),
        up=torch.full((1, 1), -1),
        down=torch.full((1, 1), -1),
    )


def test_balance_model_outputs_direction_local_variant_pairs() -> None:
    model = BalanceModel(
        left_count=3,
        right_count=3,
        up_count=1,
        down_count=1,
        door_output_variant_idx=torch.tensor([10, 10, 11, 20, 21, 21, 30, 40]),
        door_variant_compatibility=torch.ones((41, 41), dtype=torch.bool),
        room_connection_variant_idx=torch.tensor([0, 0, 1]),
        num_room_connection_variants=2,
        toilet_compatibility=torch.tensor([True, False, False]),
        hidden_width=4,
        num_layers=1,
    )
    with torch.no_grad():
        model.net[-1].bias[-2 * AREA_COUNT :] = torch.arange(2 * AREA_COUNT)
    preds = model(torch.zeros((1, len(GENERATION_VARIABLE_FLOAT_FIELDS))))

    assert preds.left.shape == (1, 2, 2)
    assert preds.right.shape == (1, 2, 2)
    assert preds.room_area.shape == (1, 3, AREA_COUNT)
    assert torch.equal(preds.room_area[0, 0], preds.room_area[0, 1])
    assert not torch.equal(preds.room_area[0, 0], preds.room_area[0, 2])
    assert preds.toilet_compatibility.tolist() == [True, False, False]
    assert "toilet_compatibility" not in model.state_dict()
    tables = compute_balance_price_tables(
        preds,
        torch.full((1, 3, AREA_COUNT), 1.0 / AREA_COUNT),
        torch.ones((1, 3), dtype=torch.bool),
    )
    # A single feasible crossing has zero centered price, regardless of other outputs.
    torch.testing.assert_close(tables.toilet_crossed_room, torch.zeros((1, 3)))


def test_toilet_compatibility_uses_crossing_columns() -> None:
    rooms = [
        {"doors": [[{"direction": "left"}]], "toilet_crossing_x": [0]},
        {"doors": [[{"direction": "right"}]], "toilet_crossing_x": []},
        {"doors": [], "toilet_crossing_x": [1]},
        {"doors": [], "toilet_crossing_x": [0], "special_type": "toilet"},
    ]
    engine = Mock()
    engine.get_output_metadata.return_value = SimpleNamespace(
        door=[(0, 0), (1, 1)],
        door_variant_compatibility=torch.ones((2, 2), dtype=torch.bool),
        room_connection_variant_idx=[0, 1, 2, 3],
        num_room_connection_variants=4,
    )
    model = create_balance_model(
        config=SimpleNamespace(balance_model=SimpleNamespace(hidden_width=4, num_layers=1)),
        rooms=rooms,
        engine=engine,
        device=torch.device("cpu"),
    )
    preds = model(torch.zeros((1, len(GENERATION_VARIABLE_FLOAT_FIELDS))))
    assert preds.toilet_compatibility.tolist() == [True, False, True, False]
    preds.toilet_crossed_room = torch.tensor([[2.0, 100.0, 4.0, 200.0]], requires_grad=True)
    tables = compute_balance_price_tables(
        preds,
        torch.full((1, 4, AREA_COUNT), 1.0 / AREA_COUNT),
        torch.ones((1, 4), dtype=torch.bool),
    )
    torch.testing.assert_close(
        tables.toilet_crossed_room, torch.tensor([[-1.0, 0.0, 1.0, 0.0]])
    )
    tables.toilet_crossed_room.square().sum().backward()
    assert preds.toilet_crossed_room.grad[0, 1] == 0
    assert preds.toilet_crossed_room.grad[0, 3] == 0


def test_area_targets_apply_preferences_and_mask_forced_rooms() -> None:
    rooms = [
        {"water": 1},
        {"heat": 3},
        {"special_type": "ship"},
        {},
    ]
    target_rooms = torch.full((1, AREA_COUNT), 4.0 / AREA_COUNT)
    force_mask = torch.tensor([[True, False, False, False, False, False]])
    preference = torch.tensor([[[0.50, 0.60, 0.75], [0.40, 0.55, 0.70]]])

    targets = compute_area_balance_targets(rooms, target_rooms, force_mask, preference)

    torch.testing.assert_close(targets.probability.sum(dim=-1), torch.ones((1, 4)))
    torch.testing.assert_close(targets.probability[0, 0, 4], torch.tensor(0.50))
    torch.testing.assert_close(targets.probability[0, 1, 2], torch.tensor(0.70))
    assert targets.probability[0, 2].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert targets.dual_mask.tolist() == [[True, True, False, True]]
    torch.testing.assert_close(targets.effective_area_rooms.sum(), torch.tensor(4.0))
    assert targets.effective_area_rooms[0, 4] > target_rooms[0, 4]
    assert targets.effective_area_rooms[0, 2] > target_rooms[0, 2]


def test_prices_are_centered_masked_and_include_area_prior() -> None:
    preds = example_predictions()
    area_probability, area_mask = uniform_area_targets()
    area_probability[0, 0] = torch.tensor([0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
    area_mask[0, 1] = False
    tables = compute_balance_price_tables(
        preds,
        area_probability,
        area_mask,
    )

    left_compatibility = torch.tensor(
        [[True, False, False], [True, False, False], [True, True, True]]
    )
    assert torch.count_nonzero(tables.left[:, ~left_compatibility]) == 0
    torch.testing.assert_close(tables.left[0, 2].mean(), torch.tensor(0.0))
    assert tables.toilet_crossed_room[0, 1] == 0.0
    assert tables.room_area[0, 0, 0] < tables.room_area[0, 0, 1]
    assert torch.count_nonzero(tables.room_area[0, 1]) == 0
    torch.testing.assert_close(
        torch.sum(tables.room_area[0, 0] * area_probability[0, 0]),
        torch.tensor(0.0),
    )


def test_forced_one_hot_area_target_has_finite_zero_price() -> None:
    targets = compute_area_balance_targets(
        rooms=[{"special_type": "ship"}, {}],
        target_area_rooms=torch.ones((1, AREA_COUNT)),
        vanilla_area_constraint_mask=torch.tensor([[True, False, False, False, False, False]]),
        preferred_probability=torch.full((1, 2, 3), 1.0 / AREA_COUNT),
    )
    tables = compute_balance_price_tables(
        example_predictions(),
        targets.probability,
        targets.dual_mask,
    )

    assert torch.all(torch.isfinite(tables.room_area))
    assert torch.count_nonzero(tables.room_area[0, 0]) == 0


def test_dual_gradient_uses_probability_error_scale() -> None:
    preds = example_predictions(requires_grad=True)
    with torch.no_grad():
        preds.left.zero_()
        preds.right.zero_()
        preds.toilet_crossed_room.zero_()
        preds.room_area.zero_()
    preds.toilet_compatibility[:] = True
    door_matches = empty_door_matches()
    door_matches.left[0, 2] = 0
    area_probability, area_mask = uniform_area_targets()
    area_mask[0, 1] = False

    loss = compute_balance_loss(
        preds=preds,
        door_matches=door_matches,
        toilet_crossed_room_idx=torch.tensor([0]),
        room_area=torch.tensor([[0, -1]]),
        area_probability=area_probability,
        area_dual_mask=area_mask,
        record_weight=torch.ones(1),
        door_beta=1.0,
        toilet_beta=1.0,
        area_beta=1.0,
    )
    loss.backward()

    torch.testing.assert_close(preds.left.grad[0, 1], torch.tensor([-2.0 / 3.0, 2.0 / 3.0]))
    torch.testing.assert_close(preds.toilet_crossed_room.grad[0], torch.tensor([-0.5, 0.5]))
    torch.testing.assert_close(
        preds.room_area.grad[0, 0],
        torch.tensor([-5.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]),
    )


def test_area_beta_does_not_regularize_fixed_prior() -> None:
    preds = example_predictions(requires_grad=True)
    area_probability, area_mask = uniform_area_targets()
    area_probability[0, 0] = torch.tensor([0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
    area_mask[0, 1] = False
    with torch.no_grad():
        preds.room_area.zero_()
    loss = compute_balance_loss(
        preds=preds,
        door_matches=empty_door_matches(),
        toilet_crossed_room_idx=torch.tensor([-1]),
        room_area=torch.full((1, 2), -1),
        area_probability=area_probability,
        area_dual_mask=area_mask,
        record_weight=torch.ones(1),
        door_beta=1.0,
        toilet_beta=1.0,
        area_beta=1.0,
    )
    loss.backward()

    assert torch.count_nonzero(preds.room_area.grad) == 0


def test_prices_are_unbounded_and_beta_pulls_corrections_toward_zero() -> None:
    preds = example_predictions(requires_grad=True)
    with torch.no_grad():
        preds.left.mul_(100.0)
    area_probability, area_mask = uniform_area_targets()
    area_mask.zero_()

    tables = compute_balance_price_tables(
        preds,
        area_probability,
        area_mask,
    )
    assert tables.left.abs().max() > 20.0
    loss = compute_balance_loss(
        preds=preds,
        door_matches=empty_door_matches(),
        toilet_crossed_room_idx=torch.tensor([-1]),
        room_area=torch.full((1, 2), -1),
        area_probability=area_probability,
        area_dual_mask=area_mask,
        record_weight=torch.ones(1),
        door_beta=1.0,
        toilet_beta=1.0,
        area_beta=1.0,
    )
    loss.backward()

    assert torch.sum(preds.left.grad * preds.left) > 0.0


def test_infeasible_toilet_observation_is_rejected() -> None:
    preds = example_predictions()
    area_probability, area_mask = uniform_area_targets()
    area_mask.zero_()
    try:
        compute_balance_loss(
            preds=preds,
            door_matches=empty_door_matches(),
            toilet_crossed_room_idx=torch.tensor([1]),
            room_area=torch.full((1, 2), -1),
            area_probability=area_probability,
            area_dual_mask=area_mask,
            record_weight=torch.ones(1),
            door_beta=1.0,
            toilet_beta=1.0,
            area_beta=1.0,
        )
    except ValueError as error:
        assert "infeasible" in str(error)
    else:
        raise AssertionError("the Toilet room itself must not be a balance target")


def test_proposal_price_residual_is_negative_price_without_gain() -> None:
    preds = example_predictions()
    area_probability, area_mask = uniform_area_targets()
    tables = compute_balance_price_tables(
        preds,
        area_probability,
        area_mask,
    )
    proposal = compute_proposal_balance_score_table(preds, tables, 6)
    residual = compute_proposal_balance_score_residual(
        proposal,
        frontier_door_variant=torch.tensor([1]),
        row_snapshot_idx=torch.tensor([0]),
    )
    torch.testing.assert_close(residual[0].reshape(6, AREA_COUNT)[:, 0], -proposal[0, 1])

    area_proposal = compute_proposal_area_balance_score_table(
        tables.room_area,
        ~area_mask,
        door_room_idx=torch.tensor([0, 1, 0, 1, 0, 1]),
        door_output_variant_idx=torch.arange(6),
        num_door_variants=6,
    )
    area_residual = compute_proposal_area_balance_score_residual(
        area_proposal,
        row_snapshot_idx=torch.tensor([0]),
    )
    torch.testing.assert_close(area_residual, -area_proposal)


def main() -> None:
    test_balance_model_outputs_direction_local_variant_pairs()
    test_toilet_compatibility_uses_crossing_columns()
    test_area_targets_apply_preferences_and_mask_forced_rooms()
    test_prices_are_centered_masked_and_include_area_prior()
    test_forced_one_hot_area_target_has_finite_zero_price()
    test_dual_gradient_uses_probability_error_scale()
    test_area_beta_does_not_regularize_fixed_prior()
    test_prices_are_unbounded_and_beta_pulls_corrections_toward_zero()
    test_infeasible_toilet_observation_is_rejected()
    test_proposal_price_residual_is_negative_price_without_gain()


if __name__ == "__main__":
    main()
