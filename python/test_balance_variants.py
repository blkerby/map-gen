from types import SimpleNamespace
from unittest.mock import Mock

import torch

from env import AREA_COUNT, DoorMatches, compute_area_balance_targets
from loss import (
    compute_balance_loss,
    compute_balance_price_tables,
    compute_toilet_balance_score_targets,
    compute_room_area_balance_score_targets,
    compute_proposal_area_balance_score_residual,
    compute_proposal_area_balance_score_table,
    compute_proposal_balance_score_residual,
    compute_proposal_balance_score_table,
)
from model import BalanceModel, BalancePredictions, compatible_proposal_door_pairs
from model_loading import create_balance_model
from train_config import GENERATION_VARIABLE_FLOAT_FIELDS


def example_predictions(requires_grad: bool = False) -> BalancePredictions:
    return BalancePredictions(
        left=torch.tensor([[[0.0, 1.0], [2.0, -1.0]]], requires_grad=requires_grad),
        right=torch.tensor([[[0.5, -0.5], [1.0, 0.0]]], requires_grad=requires_grad),
        up=torch.zeros((1, 1, 1), requires_grad=requires_grad),
        down=torch.zeros((1, 1, 1), requires_grad=requires_grad),
        toilet_crossed_room=torch.zeros((1, 2), requires_grad=requires_grad),
        toilet_failure=torch.zeros(1, requires_grad=requires_grad),
        door_failure=torch.zeros((1, 8), requires_grad=requires_grad),
        room_area_failure=torch.zeros((1, 2), requires_grad=requires_grad),
        room_area=torch.zeros((1, 2, AREA_COUNT), requires_grad=requires_grad),
        area_order=torch.zeros((1, AREA_COUNT, AREA_COUNT), requires_grad=requires_grad),
        area_order_failure=torch.zeros((1, AREA_COUNT), requires_grad=requires_grad),
        left_door_variant_idx=torch.tensor([0, 0, 1]),
        right_door_variant_idx=torch.tensor([0, 1, 1]),
        up_door_variant_idx=torch.tensor([0]),
        down_door_variant_idx=torch.tensor([0]),
        left_global_door_variant_idx=torch.tensor([0, 1]),
        right_global_door_variant_idx=torch.tensor([2, 3]),
        up_global_door_variant_idx=torch.tensor([4]),
        down_global_door_variant_idx=torch.tensor([5]),
        left_compatibility=torch.tensor(
            [[True, True, False], [True, True, False], [True, True, True]]
        ),
        right_compatibility=torch.tensor(
            [[True, True, True], [True, True, True], [False, False, True]]
        ),
        up_compatibility=torch.ones((1, 1), dtype=torch.bool),
        down_compatibility=torch.ones((1, 1), dtype=torch.bool),
        toilet_compatibility=torch.tensor([True, False]),
        left_probability=torch.tensor([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]]),
        right_probability=torch.tensor([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]]),
        up_probability=torch.ones((1, 1)),
        down_probability=torch.ones((1, 1)),
        horizontal_proposal_door_pairs=torch.tensor([[0, 0, 2, 2], [0, 1, 0, 1]]),
        vertical_proposal_door_pairs=torch.tensor([[0], [0]]),
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
        door_room_idx=torch.tensor([0, 0, 0, 1, 1, 1, 0, 1]),
        door_variant_compatibility=torch.ones((41, 41), dtype=torch.bool),
        room_connection_variant_idx=torch.tensor([0, 0, 1]),
        num_room_connection_variants=2,
        toilet_compatibility=torch.tensor([True, False, False]),
        hidden_width=4,
        num_layers=1,
    )
    with torch.no_grad():
        model.area_net[-1].bias[:2 * AREA_COUNT] = torch.arange(2 * AREA_COUNT)
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


def test_concrete_door_masks_exclude_same_room_and_preserve_other_instances() -> None:
    model = BalanceModel(
        left_count=3,
        right_count=3,
        up_count=1,
        down_count=1,
        door_output_variant_idx=torch.tensor([0, 0, 1, 2, 2, 3, 4, 5]),
        door_room_idx=torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]),
        door_variant_compatibility=torch.ones((6, 6), dtype=torch.bool),
        room_connection_variant_idx=torch.tensor([0, 0, 1]),
        num_room_connection_variants=2,
        toilet_compatibility=torch.zeros(3, dtype=torch.bool),
        hidden_width=4,
        num_layers=1,
    )
    preds = model(torch.zeros((1, len(GENERATION_VARIABLE_FLOAT_FIELDS))))
    expected = ~torch.eye(3, dtype=torch.bool)
    assert torch.equal(preds.left_compatibility, expected)
    assert torch.equal(preds.right_compatibility, expected.T)
    assert preds.up_compatibility.all()
    assert preds.down_compatibility.all()
    assert "door_variant_compatibility" not in dict(model.named_buffers())
    for direction in ("left", "right", "up", "down"):
        assert f"{direction}_compatibility" in model.state_dict()

    preds.left = torch.tensor([[[100.0, 10.0], [2.0, 6.0]]], requires_grad=True)
    area_probability = torch.full((1, 3, AREA_COUNT), 1.0 / AREA_COUNT)
    area_mask = torch.ones((1, 3), dtype=torch.bool)
    tables = compute_balance_price_tables(preds, area_probability, area_mask)
    # Each door has two real partners with equal target probability.
    torch.testing.assert_close(
        tables.left, torch.tensor([[[0.0, 45.0, -45.0], [45.0, 0.0, -45.0], [0.0, 0.0, 0.0]]])
    )
    assert torch.count_nonzero(tables.up) == 0
    assert torch.count_nonzero(tables.down) == 0
    door_matches = DoorMatches(
        left=torch.tensor([[1, 0, 0]]),
        right=torch.full((1, 3), -1),
        up=torch.full((1, 1), -1),
        down=torch.full((1, 1), -1),
    )
    loss = compute_balance_loss(
        area_order=torch.full((preds.room_area.shape[0], 6), -1, dtype=torch.int64),
        order_beta=1.0,
        preds=preds,
        door_matches=door_matches,
        toilet_crossed_room_idx=torch.tensor([-1]),
        room_area=torch.full((1, 3), -1),
        area_probability=area_probability,
        area_dual_mask=area_mask,
        record_weight=torch.ones(1),
        door_beta=0.0,
        toilet_beta=0.0,
        area_beta=0.0,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(preds.left.grad).all()
    torch.testing.assert_close(
        preds.left.grad, torch.tensor([[[-1.0 / 8.0, 1.0 / 8.0], [0.0, 0.0]]])
    )

    door_matches.left[0, 0] = 0
    try:
        compute_balance_loss(
            area_order=torch.full((preds.room_area.shape[0], 6), -1, dtype=torch.int64),
            order_beta=1.0,
            preds=preds,
            door_matches=door_matches,
            toilet_crossed_room_idx=torch.tensor([-1]),
            room_area=torch.full((1, 3), -1),
            area_probability=area_probability,
            area_dual_mask=area_mask,
            record_weight=torch.ones(1),
            door_beta=0.0,
            toilet_beta=0.0,
            area_beta=0.0,
        )
    except ValueError as error:
        assert str(error) == "observed door pairing is incompatible"
    else:
        raise AssertionError("same-room observation was accepted")


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
    torch.testing.assert_close(tables.toilet_crossed_room, torch.tensor([[-1.0, 0.0, 1.0, 0.0]]))
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


def test_prices_are_centered_masked_and_have_no_fixed_prior() -> None:
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
        [[True, True, False], [True, True, False], [True, True, True]]
    )
    assert torch.count_nonzero(tables.left[:, ~left_compatibility]) == 0
    torch.testing.assert_close((tables.left * preds.left_probability).sum(-1), torch.zeros((1, 3)))
    assert tables.toilet_crossed_room[0, 1] == 0.0
    torch.testing.assert_close(tables.room_area, torch.zeros_like(tables.room_area))
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
    area_probability[0, 0] = torch.tensor([0.5, 0.1, 0.1, 0.1, 0.1, 0.1])

    loss = compute_balance_loss(
        area_order=torch.full((preds.room_area.shape[0], 6), -1, dtype=torch.int64),
        order_beta=1.0,
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

    # This row's only partner in a complete matching has target probability 1.
    torch.testing.assert_close(preds.left.grad[0, 1], torch.tensor([-1.0 / 8.0, 1.0 / 8.0]))
    torch.testing.assert_close(preds.toilet_crossed_room.grad[0], torch.tensor([-0.5, 0.5]))
    torch.testing.assert_close(
        preds.room_area.grad[0, 0],
        torch.tensor([-0.5, 0.1, 0.1, 0.1, 0.1, 0.1]),
    )


def test_zero_area_prices_have_zero_regularization_gradient() -> None:
    preds = example_predictions(requires_grad=True)
    area_probability, area_mask = uniform_area_targets()
    area_probability[0, 0] = torch.tensor([0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
    area_mask[0, 1] = False
    with torch.no_grad():
        preds.room_area.zero_()
    loss = compute_balance_loss(
        area_order=torch.full((preds.room_area.shape[0], 6), -1, dtype=torch.int64),
        order_beta=1.0,
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
        area_order=torch.full((preds.room_area.shape[0], 6), -1, dtype=torch.int64),
        order_beta=1.0,
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
            area_order=torch.full((preds.room_area.shape[0], 6), -1, dtype=torch.int64),
            order_beta=1.0,
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


def test_room_area_targets_use_zero_for_terminal_absence() -> None:
    preds = example_predictions()
    probability, mask = uniform_area_targets()
    tables = compute_balance_price_tables(preds, probability, mask)
    tables.room_area = torch.tensor(
        [[[7.0, 2.0, -4.0, 1.0, 3.0, 5.0], [-8.0, 3.0, 2.0, 4.0, 6.0, 9.0]]],
        requires_grad=True,
    )
    targets = compute_room_area_balance_score_targets(tables, torch.tensor([[-1, 2]]))
    torch.testing.assert_close(targets, torch.tensor([[0.0, 2.0]]))
    assert not targets.requires_grad


def test_proposal_prices_use_compatible_instances_in_both_directions() -> None:
    model = BalanceModel(
        left_count=3,
        right_count=3,
        up_count=3,
        down_count=3,
        door_output_variant_idx=torch.tensor([0, 0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7]),
        door_room_idx=torch.tensor([0, 1, 2] * 4),
        door_variant_compatibility=torch.ones((8, 8), dtype=torch.bool),
        room_connection_variant_idx=torch.tensor([0, 0, 1]),
        num_room_connection_variants=2,
        toilet_compatibility=torch.zeros(3, dtype=torch.bool),
        hidden_width=4,
        num_layers=1,
    )
    preds = model(torch.zeros((2, len(GENERATION_VARIABLE_FLOAT_FIELDS))))
    for direction in ("left", "right", "up", "down"):
        setattr(
            preds, direction, torch.tensor([[[1.0, 7.0], [3.0, 9.0]], [[-2.0, 4.0], [8.0, 1.0]]])
        )
    tables = compute_balance_price_tables(
        preds,
        torch.full((2, 3, AREA_COUNT), 1.0 / AREA_COUNT),
        torch.ones((2, 3), dtype=torch.bool),
    )
    proposal = compute_proposal_balance_score_table(preds, tables, 8)
    for direction, reverse in (("left", "right"), ("up", "down")):
        source_idx, target_idx = getattr(preds, f"{direction}_compatibility").nonzero(
            as_tuple=True
        )
        source_variant = getattr(preds, f"{direction}_global_door_variant_idx")[
            getattr(preds, f"{direction}_door_variant_idx")[source_idx]
        ]
        target_variant = getattr(preds, f"{reverse}_global_door_variant_idx")[
            getattr(preds, f"{reverse}_door_variant_idx")[target_idx]
        ]
        exact = (
            getattr(tables, direction)[:, source_idx, target_idx]
            + getattr(tables, reverse)[:, target_idx, source_idx]
        )
        torch.testing.assert_close(proposal[:, source_variant, target_variant], exact)
        torch.testing.assert_close(proposal[:, target_variant, source_variant], exact)
    assert torch.all(proposal[:, 0, 2] != 0)
    assert torch.all(proposal[:, 4, 6] != 0)
    # These variants only occur in room 2, so they have no compatible realization.
    assert torch.count_nonzero(proposal[:, [1, 3, 5, 7], [3, 1, 7, 5]]) == 0
    assert "horizontal_proposal_door_pairs" not in model.state_dict()
    assert "vertical_proposal_door_pairs" not in model.state_dict()


def test_compatible_proposal_door_pairs_with_empty_directions() -> None:
    for source_count, target_count in ((0, 0), (0, 2), (2, 0), (2, 2)):
        pairs = compatible_proposal_door_pairs(
            torch.arange(source_count),
            torch.arange(target_count),
            torch.zeros((source_count, target_count), dtype=torch.bool),
            source_count,
            target_count,
        )
        assert pairs.shape == (2, 0)
        assert pairs.dtype == torch.int64


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


def test_toilet_failure_drives_dual_and_has_unconditional_target() -> None:
    preds = example_predictions(requires_grad=True)
    preds.toilet_compatibility[:] = True
    area_probability, area_mask = uniform_area_targets()
    loss = compute_balance_loss(
        area_order=torch.full((preds.room_area.shape[0], 6), -1, dtype=torch.int64),
        order_beta=1.0,
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
    # Gradient descent raises failure relative to the mean successful price.
    torch.testing.assert_close(preds.toilet_failure.grad, torch.tensor([-1.0]))
    torch.testing.assert_close(preds.toilet_crossed_room.grad, torch.zeros((1, 2)))
    preds.toilet_crossed_room = torch.tensor([[2.0, 4.0]])
    preds.toilet_failure = torch.tensor([8.0], requires_grad=True)
    tables = compute_balance_price_tables(preds, area_probability, area_mask)
    torch.testing.assert_close(tables.toilet_crossed_room, torch.tensor([[-1.0, 1.0]]))
    torch.testing.assert_close(tables.toilet_failure, torch.tensor([8.0]))
    failed_target = compute_toilet_balance_score_targets(tables, torch.tensor([-1]))
    assert not failed_target.requires_grad
    torch.testing.assert_close(failed_target, torch.tensor([8.0]))
    torch.testing.assert_close(
        compute_toilet_balance_score_targets(tables, torch.tensor([1])), torch.tensor([1.0])
    )
    # Failure's zero target does not dilute the successful-outcome centering.
    preds.toilet_crossed_room += 10.0
    shifted = compute_balance_price_tables(preds, area_probability, area_mask)
    torch.testing.assert_close(shifted.toilet_failure, tables.toilet_failure)
    torch.testing.assert_close(shifted.toilet_crossed_room, tables.toilet_crossed_room)
    preds.toilet_compatibility[:] = False
    disabled = compute_balance_price_tables(preds, area_probability, area_mask)
    assert torch.count_nonzero(disabled.toilet_failure) == 0
    assert torch.count_nonzero(disabled.toilet_crossed_room) == 0


def test_toilet_failure_price_has_regularized_equilibrium() -> None:
    preds = example_predictions(requires_grad=True)
    preds.toilet_compatibility[:] = True
    with torch.no_grad():
        preds.toilet_failure.fill_(0.7 / 2.0)
    area_probability, area_mask = uniform_area_targets()
    # With 70% failure and equally frequent successful rooms, beta=2 gives f=0.35.
    for outcome, probability in ((-1, 0.7), (0, 0.15), (1, 0.15)):
        loss = compute_balance_loss(
            area_order=torch.full((preds.room_area.shape[0], 6), -1, dtype=torch.int64),
            order_beta=1.0,
            preds=preds,
            door_matches=empty_door_matches(),
            toilet_crossed_room_idx=torch.tensor([outcome]),
            room_area=torch.full((1, 2), -1),
            area_probability=area_probability,
            area_dual_mask=area_mask,
            record_weight=torch.ones(1),
            door_beta=1.0,
            toilet_beta=2.0,
            area_beta=1.0,
        )
        (probability * loss).backward()
    torch.testing.assert_close(preds.toilet_failure.grad, torch.zeros(1), atol=1e-7, rtol=0)
    torch.testing.assert_close(
        preds.toilet_crossed_room.grad, torch.zeros((1, 2)), atol=1e-7, rtol=0
    )


def main() -> None:
    test_toilet_failure_drives_dual_and_has_unconditional_target()
    test_toilet_failure_price_has_regularized_equilibrium()
    test_balance_model_outputs_direction_local_variant_pairs()
    test_concrete_door_masks_exclude_same_room_and_preserve_other_instances()
    test_toilet_compatibility_uses_crossing_columns()
    test_area_targets_apply_preferences_and_mask_forced_rooms()
    test_prices_are_centered_masked_and_have_no_fixed_prior()
    test_forced_one_hot_area_target_has_finite_zero_price()
    test_dual_gradient_uses_probability_error_scale()
    test_zero_area_prices_have_zero_regularization_gradient()
    test_prices_are_unbounded_and_beta_pulls_corrections_toward_zero()
    test_infeasible_toilet_observation_is_rejected()
    test_room_area_targets_use_zero_for_terminal_absence()
    test_proposal_prices_use_compatible_instances_in_both_directions()
    test_compatible_proposal_door_pairs_with_empty_directions()
    test_proposal_price_residual_is_negative_price_without_gain()


if __name__ == "__main__":
    main()
