import math
from dataclasses import replace

import torch

from env import AREA_COUNT, DoorMatches
from loss import (
    categorical_balance_loss,
    compute_balance_door_match_ss,
    compute_balance_loss,
    compute_balance_score_tables,
    compute_balance_score_target_logits,
    compute_room_area_balance_score_target_logits,
    compute_proposal_area_balance_score_residual,
    compute_proposal_area_balance_score_table,
    compute_proposal_balance_score_residual,
    compute_proposal_balance_score_table,
    compute_step_balance_score_target_logits,
    expand_balance_door_match_probabilities,
    expand_direction_balance_probabilities,
    materialize_direction_balance_compatibility,
    materialize_direction_balance_logits,
    masked_bernoulli_kl_loss,
    masked_offset_bernoulli_kl_loss,
)
from model import BalanceModel, BalancePredictions
from train import compute_balance_metric_values
from train_config import GENERATION_VARIABLE_FLOAT_FIELDS


def example_door_variant_compatibility() -> torch.Tensor:
    compatibility = torch.zeros((6, 6), dtype=torch.bool)
    compatible_pairs = (
        (0, 2),
        (1, 2),
        (1, 3),
        (2, 0),
        (2, 1),
        (3, 1),
        (4, 5),
        (5, 4),
    )
    for source_variant_idx, target_variant_idx in compatible_pairs:
        compatibility[source_variant_idx, target_variant_idx] = True
    return compatibility


def example_predictions() -> BalancePredictions:
    return BalancePredictions(
        left=torch.tensor([[[0.0, 1.0], [2.0, -1.0]]]),
        right=torch.tensor([[[0.5, -0.5], [1.0, 0.0]]]),
        up=torch.zeros((1, 1, 1)),
        down=torch.zeros((1, 1, 1)),
        toilet_crossed_room=torch.zeros((1, 2)),
        room_area=torch.zeros((1, 2, AREA_COUNT)),
        left_door_variant_idx=torch.tensor([0, 0, 1]),
        right_door_variant_idx=torch.tensor([0, 1, 1]),
        up_door_variant_idx=torch.tensor([0]),
        down_door_variant_idx=torch.tensor([0]),
        left_global_door_variant_idx=torch.tensor([0, 1]),
        right_global_door_variant_idx=torch.tensor([2, 3]),
        up_global_door_variant_idx=torch.tensor([4]),
        down_global_door_variant_idx=torch.tensor([5]),
        door_variant_compatibility=example_door_variant_compatibility(),
    )


class ConfigurationBalanceModel:
    def __call__(self, inputs: torch.Tensor) -> BalancePredictions:
        preds = example_predictions()
        batch_size = inputs.shape[0]
        left = preds.left.expand(batch_size, -1, -1).clone()
        left[:, 0, 0] += inputs[:, 0]
        toilet = preds.toilet_crossed_room.expand(batch_size, -1).clone()
        toilet[:, 0] += inputs[:, 0]
        return replace(
            preds,
            left=left,
            right=preds.right.expand(batch_size, -1, -1),
            up=preds.up.expand(batch_size, -1, -1),
            down=preds.down.expand(batch_size, -1, -1),
            toilet_crossed_room=toilet,
            room_area=preds.room_area.expand(batch_size, -1, -1),
        )


def example_door_matches() -> DoorMatches:
    return DoorMatches(
        left=torch.tensor([[0, 0, 2]]),
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
        hidden_width=4,
        num_layers=1,
    )
    with torch.no_grad():
        model.net[-1].bias[-2 * AREA_COUNT :] = torch.arange(2 * AREA_COUNT)
    preds = model(torch.zeros((1, len(GENERATION_VARIABLE_FLOAT_FIELDS))))

    assert preds.left.shape == (1, 2, 2)
    assert preds.right.shape == (1, 2, 2)
    assert preds.up.shape == (1, 1, 1)
    assert preds.down.shape == (1, 1, 1)
    assert preds.room_area.shape == (1, 3, AREA_COUNT)
    assert torch.equal(preds.room_area[0, 0], preds.room_area[0, 1])
    assert not torch.equal(preds.room_area[0, 0], preds.room_area[0, 2])
    assert preds.left_door_variant_idx.tolist() == [0, 0, 1]
    assert preds.right_door_variant_idx.tolist() == [0, 1, 1]
    assert preds.left_global_door_variant_idx.tolist() == [10, 11]
    assert preds.right_global_door_variant_idx.tolist() == [20, 21]


def test_room_area_balance_scores_are_centered_and_gathered() -> None:
    preds = example_predictions()
    preds.room_area[0, 0] = torch.tensor([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    tables = compute_balance_score_tables(preds)
    assert tables.room_area[0, 0, 0] > 0
    assert tables.room_area[0, 0, 1] < 0
    assert torch.allclose(tables.room_area[0, 1], torch.zeros(AREA_COUNT))

    target, uniform_log_odds, mask = compute_room_area_balance_score_target_logits(
        tables,
        torch.tensor([[0, -1]]),
    )
    assert target[0, 0] == tables.room_area[0, 0, 0]
    assert torch.all(uniform_log_odds == tables.room_area_uniform_log_odds)
    assert mask.tolist() == [[True, False]]


def test_balance_loss_maps_concrete_matches_to_variant_pairs() -> None:
    preds = example_predictions()
    room_area = torch.full((1, 2), -1)
    loss = compute_balance_loss(
        preds,
        example_door_matches(),
        toilet_crossed_room_idx=torch.tensor([-1]),
        room_area=room_area,
        room_area_mask=torch.ones_like(room_area, dtype=torch.bool),
        record_weight=torch.ones(1),
    )
    concrete_logits = materialize_direction_balance_logits(
        preds.left,
        preds.left_door_variant_idx,
        preds.right_door_variant_idx,
    )
    compatibility = materialize_direction_balance_compatibility(
        preds.door_variant_compatibility,
        preds.left_global_door_variant_idx,
        preds.right_global_door_variant_idx,
        preds.left_door_variant_idx,
        preds.right_door_variant_idx,
    )
    expected = torch.nn.functional.cross_entropy(
        concrete_logits.masked_fill(~compatibility.unsqueeze(0), -torch.inf).flatten(0, 1),
        example_door_matches().left.flatten(),
    )

    torch.testing.assert_close(loss, expected)


def test_balance_loss_masks_room_area_targets() -> None:
    preds = example_predictions()
    preds.room_area[0, 1] = torch.arange(AREA_COUNT, dtype=torch.float32)
    door_matches = example_door_matches()
    door_matches.left.fill_(-1)
    room_area = torch.tensor([[0, 1]])
    loss = compute_balance_loss(
        preds,
        door_matches,
        toilet_crossed_room_idx=torch.tensor([-1]),
        room_area=room_area,
        room_area_mask=torch.tensor([[False, True]]),
        record_weight=torch.ones(1),
    )
    expected = torch.nn.functional.cross_entropy(
        preds.room_area[:, 1],
        room_area[:, 1],
    )

    torch.testing.assert_close(loss, expected)


def test_categorical_balance_loss_weights_records_and_denominator() -> None:
    logits = torch.tensor([[0.0, 0.0], [0.0, 2.0]])
    targets = torch.tensor([0, 0])
    record_weight = torch.tensor([1.0, 0.25])

    loss, weight = categorical_balance_loss(logits, targets, record_weight)
    per_record_loss = torch.nn.functional.cross_entropy(logits, targets, reduction="none")

    torch.testing.assert_close(loss, per_record_loss[0] + 0.25 * per_record_loss[1])
    torch.testing.assert_close(weight, torch.tensor(1.25))


def test_balance_target_scores_expand_variant_probabilities_to_concrete_matches() -> None:
    preds = example_predictions()
    tables = compute_balance_score_tables(preds)
    left_compatibility = materialize_direction_balance_compatibility(
        preds.door_variant_compatibility,
        preds.left_global_door_variant_idx,
        preds.right_global_door_variant_idx,
        preds.left_door_variant_idx,
        preds.right_door_variant_idx,
    )
    assert torch.count_nonzero(tables.left[:, ~left_compatibility]) == 0
    assert torch.count_nonzero(tables.left[:, :2]) == 0
    assert torch.count_nonzero(tables.left[:, 2]) > 0
    torch.testing.assert_close(tables.left[0, 2, 0], torch.tensor(3.0))
    torch.testing.assert_close(
        tables.left_uniform_log_odds,
        torch.tensor([20.0, 20.0, -math.log(2.0)]),
    )
    zero_preds = example_predictions()
    zero_preds.left.zero_()
    zero_preds.right.zero_()
    zero_preds.up.zero_()
    zero_preds.down.zero_()
    zero_tables = compute_balance_score_tables(zero_preds)
    for table in (zero_tables.left, zero_tables.right, zero_tables.up, zero_tables.down):
        assert torch.count_nonzero(table) == 0
    zero_proposal_table = compute_proposal_balance_score_table(
        zero_preds,
        zero_tables,
        num_door_variants=6,
    )
    assert torch.count_nonzero(zero_proposal_table) == 0

    values, uniform_log_odds, mask = compute_balance_score_target_logits(
        tables,
        example_door_matches(),
    )
    left_table = tables.left
    expected_left = torch.stack(
        [
            left_table[0, 0, 0],
            left_table[0, 1, 1],
            left_table[0, 2, 2],
        ]
    ).unsqueeze(0)

    torch.testing.assert_close(values[:, :3], expected_left)
    torch.testing.assert_close(
        uniform_log_odds[:, :3],
        tables.left_uniform_log_odds.unsqueeze(0),
    )
    assert mask[:, :3].all()
    assert not mask[:, 3:].any()

    concrete_door_match = torch.cat(
        [
            example_door_matches().left,
            example_door_matches().right,
            example_door_matches().up,
            example_door_matches().down,
        ],
        dim=1,
    ).unsqueeze(1)
    step_values, step_mask = compute_step_balance_score_target_logits(
        tables,
        concrete_door_match,
    )
    torch.testing.assert_close(step_values[:, 0, :3], expected_left)
    assert step_mask[:, 0, :3].all()
    assert not step_mask[:, 0, 3:].any()


def test_main_balance_kl_restores_uniform_log_odds_offset() -> None:
    prediction = torch.tensor([0.25])
    target = torch.tensor([1.0])
    uniform_log_odds = torch.tensor([-3.0])
    mask = torch.tensor([True])
    offset_loss, offset_weight = masked_offset_bernoulli_kl_loss(
        prediction,
        target,
        uniform_log_odds,
        mask,
        1.0,
    )
    expected_loss, expected_weight = masked_bernoulli_kl_loss(
        prediction + uniform_log_odds,
        target + uniform_log_odds,
        mask,
        1.0,
    )
    relative_loss, _ = masked_bernoulli_kl_loss(
        prediction,
        target,
        mask,
        1.0,
    )

    torch.testing.assert_close(offset_loss, expected_loss)
    torch.testing.assert_close(offset_weight, expected_weight)
    assert not torch.isclose(offset_loss, relative_loss)
    neutral_loss, _ = masked_offset_bernoulli_kl_loss(
        torch.zeros(1),
        torch.zeros(1),
        uniform_log_odds,
        mask,
        1.0,
    )
    torch.testing.assert_close(neutral_loss, torch.tensor(0.0))


def test_proposal_balance_residual_adds_both_door_directions() -> None:
    preds = example_predictions()
    tables = compute_balance_score_tables(preds)
    reward_balance = torch.tensor([0.25])
    proposal_score_table = compute_proposal_balance_score_table(
        preds,
        tables,
        num_door_variants=6,
    )
    residual = compute_proposal_balance_score_residual(
        proposal_score_table,
        frontier_door_variant=torch.tensor([0, 3, 4]),
        row_snapshot_idx=torch.tensor([0, 0, 0]),
        reward_balance=reward_balance,
    ).reshape(3, 6, AREA_COUNT)

    expected_left_to_right = -reward_balance[0] * torch.stack(
        [
            tables.left[0, 0, 0] + tables.right[0, 0, 0],
            tables.left[0, 0, 1] + tables.right[0, 1, 0],
        ]
    )
    torch.testing.assert_close(
        residual[0, 2:4],
        expected_left_to_right.unsqueeze(1).expand(-1, AREA_COUNT),
    )
    expected_right_to_left = -reward_balance[0] * (tables.right[0, 1, 2] + tables.left[0, 2, 1])
    torch.testing.assert_close(
        residual[1, 1],
        expected_right_to_left.expand(AREA_COUNT),
    )
    assert torch.count_nonzero(residual[0, [0, 1, 4, 5]]) == 0
    assert torch.count_nonzero(residual[1, [2, 3, 4, 5]]) == 0


def test_proposal_area_balance_residual_uses_tied_room_score() -> None:
    room_area_score_table = torch.arange(3 * AREA_COUNT, dtype=torch.float32).reshape(
        1, 3, AREA_COUNT
    )
    room_area_score_table[:, 1] = room_area_score_table[:, 0]
    proposal_score_table = compute_proposal_area_balance_score_table(
        room_area_score_table,
        exempt_room=torch.tensor([[True, True, False]]),
        door_room_idx=torch.tensor([0, 1, 2]),
        door_output_variant_idx=torch.tensor([0, 0, 1]),
        num_door_variants=2,
    )
    residual = compute_proposal_area_balance_score_residual(
        proposal_score_table,
        row_snapshot_idx=torch.tensor([0, 0]),
        reward_area_balance=torch.tensor([0.25]),
    ).reshape(2, 2, AREA_COUNT)

    assert torch.count_nonzero(residual[:, 0]) == 0
    torch.testing.assert_close(
        residual[:, 1],
        -0.25 * room_area_score_table[:, 2].expand(2, -1),
    )


def test_balance_ss_materializes_concrete_pair_probabilities() -> None:
    preds = example_predictions()
    expanded_left = expand_direction_balance_probabilities(
        preds.left,
        preds.left_door_variant_idx,
        preds.right_door_variant_idx,
        preds.left_global_door_variant_idx,
        preds.right_global_door_variant_idx,
        preds.door_variant_compatibility,
    )
    left_compatibility = materialize_direction_balance_compatibility(
        preds.door_variant_compatibility,
        preds.left_global_door_variant_idx,
        preds.right_global_door_variant_idx,
        preds.left_door_variant_idx,
        preds.right_door_variant_idx,
    )
    expected_left = torch.softmax(
        materialize_direction_balance_logits(
            preds.left,
            preds.left_door_variant_idx,
            preds.right_door_variant_idx,
        ).masked_fill(~left_compatibility.unsqueeze(0), -torch.inf),
        dim=-1,
    )

    torch.testing.assert_close(expanded_left, expected_left)
    torch.testing.assert_close(
        expanded_left.sum(dim=-1),
        torch.ones((1, 3)),
    )
    expected_ss = sum(
        torch.sum(probability.square())
        for logits, source_idx, target_idx, source_global_idx, target_global_idx in (
            (
                preds.left,
                preds.left_door_variant_idx,
                preds.right_door_variant_idx,
                preds.left_global_door_variant_idx,
                preds.right_global_door_variant_idx,
            ),
            (
                preds.right,
                preds.right_door_variant_idx,
                preds.left_door_variant_idx,
                preds.right_global_door_variant_idx,
                preds.left_global_door_variant_idx,
            ),
            (
                preds.up,
                preds.up_door_variant_idx,
                preds.down_door_variant_idx,
                preds.up_global_door_variant_idx,
                preds.down_global_door_variant_idx,
            ),
            (
                preds.down,
                preds.down_door_variant_idx,
                preds.up_door_variant_idx,
                preds.down_global_door_variant_idx,
                preds.up_global_door_variant_idx,
            ),
        )
        for probability in (
            expand_direction_balance_probabilities(
                logits,
                source_idx,
                target_idx,
                source_global_idx,
                target_global_idx,
                preds.door_variant_compatibility,
            ),
        )
    )
    torch.testing.assert_close(compute_balance_door_match_ss(preds), expected_ss)


def test_balance_metrics_average_only_configurations_with_valid_targets() -> None:
    inputs = torch.zeros((2, len(GENERATION_VARIABLE_FLOAT_FIELDS)))
    inputs[1, 0] = 2.0
    model = ConfigurationBalanceModel()
    preds = model(inputs)
    door_matches = DoorMatches(
        left=torch.tensor([[0, -1, 0], [-1, 0, 0]]),
        right=torch.zeros((2, 3), dtype=torch.int64),
        up=torch.tensor([[0], [-1]]),
        down=torch.tensor([[-1], [0]]),
    )
    match_targets = (
        door_matches.left,
        door_matches.right,
        door_matches.up,
        door_matches.down,
    )
    mean_probabilities = [
        torch.sum(probability * (targets >= 0).unsqueeze(-1), dim=0)
        / torch.sum(targets >= 0, dim=0).unsqueeze(-1)
        for probability, targets in zip(
            expand_balance_door_match_probabilities(preds),
            match_targets,
            strict=True,
        )
    ]
    mean_toilet_probability = torch.softmax(preds.toilet_crossed_room[0], dim=-1)

    metrics = compute_balance_metric_values(
        balance_model=model,
        generation_variable_floats=inputs,
        door_matches=door_matches,
        toilet_crossed_room_idx=torch.tensor([0, -1]),
        batch_size=1,
    )

    torch.testing.assert_close(
        metrics.door_match_ss,
        sum(torch.sum(probability.square()) for probability in mean_probabilities).to(
            torch.float64
        ),
    )
    torch.testing.assert_close(
        metrics.toilet_crossed_room_ss,
        torch.sum(mean_toilet_probability.square()).to(torch.float64),
    )


def main() -> None:
    test_balance_model_outputs_direction_local_variant_pairs()
    test_room_area_balance_scores_are_centered_and_gathered()
    test_balance_loss_maps_concrete_matches_to_variant_pairs()
    test_balance_loss_masks_room_area_targets()
    test_categorical_balance_loss_weights_records_and_denominator()
    test_balance_target_scores_expand_variant_probabilities_to_concrete_matches()
    test_main_balance_kl_restores_uniform_log_odds_offset()
    test_proposal_balance_residual_adds_both_door_directions()
    test_proposal_area_balance_residual_uses_tied_room_score()
    test_balance_ss_materializes_concrete_pair_probabilities()
    test_balance_metrics_average_only_configurations_with_valid_targets()


if __name__ == "__main__":
    main()
