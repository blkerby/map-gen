import torch

from env import AREA_COUNT, DoorMatches
from loss import (
    compute_balance_door_match_ss,
    compute_balance_loss,
    compute_balance_score_tables,
    compute_balance_score_target_logits,
    compute_proposal_balance_score_residual,
    compute_proposal_balance_score_table,
    compute_step_balance_score_target_logits,
    expand_direction_balance_probabilities,
    materialize_direction_balance_logits,
)
from model import BalanceModel, BalancePredictions
from train_config import GENERATION_VARIABLE_FLOAT_FIELDS


def example_predictions() -> BalancePredictions:
    return BalancePredictions(
        left=torch.tensor([[[0.0, 1.0], [2.0, -1.0]]]),
        right=torch.tensor([[[0.5, -0.5], [1.0, 0.0]]]),
        up=torch.zeros((1, 1, 1)),
        down=torch.zeros((1, 1, 1)),
        toilet_crossed_room=torch.zeros((1, 2)),
        left_door_variant_idx=torch.tensor([0, 0, 1]),
        right_door_variant_idx=torch.tensor([0, 1, 1]),
        up_door_variant_idx=torch.tensor([0]),
        down_door_variant_idx=torch.tensor([0]),
        left_global_door_variant_idx=torch.tensor([0, 1]),
        right_global_door_variant_idx=torch.tensor([2, 3]),
        up_global_door_variant_idx=torch.tensor([4]),
        down_global_door_variant_idx=torch.tensor([5]),
    )


def example_door_matches() -> DoorMatches:
    return DoorMatches(
        left=torch.tensor([[0, 1, 2]]),
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
        num_rooms=2,
        hidden_width=4,
        num_layers=1,
    )
    preds = model(torch.zeros((1, len(GENERATION_VARIABLE_FLOAT_FIELDS))))

    assert preds.left.shape == (1, 2, 2)
    assert preds.right.shape == (1, 2, 2)
    assert preds.up.shape == (1, 1, 1)
    assert preds.down.shape == (1, 1, 1)
    assert preds.left_door_variant_idx.tolist() == [0, 0, 1]
    assert preds.right_door_variant_idx.tolist() == [0, 1, 1]
    assert preds.left_global_door_variant_idx.tolist() == [10, 11]
    assert preds.right_global_door_variant_idx.tolist() == [20, 21]


def test_balance_loss_maps_concrete_matches_to_variant_pairs() -> None:
    preds = example_predictions()
    loss = compute_balance_loss(
        preds,
        example_door_matches(),
        toilet_crossed_room_idx=torch.tensor([-1]),
    )
    concrete_logits = materialize_direction_balance_logits(
        preds.left,
        preds.left_door_variant_idx,
        preds.right_door_variant_idx,
    )
    expected = torch.nn.functional.cross_entropy(
        concrete_logits.flatten(0, 1),
        example_door_matches().left.flatten(),
    )

    torch.testing.assert_close(loss, expected)


def test_balance_target_scores_expand_variant_probabilities_to_concrete_matches() -> None:
    preds = example_predictions()
    tables = compute_balance_score_tables(preds)
    left_concrete_probabilities = torch.softmax(
        materialize_direction_balance_logits(
            preds.left,
            preds.left_door_variant_idx,
            preds.right_door_variant_idx,
        ),
        dim=-1,
    )
    torch.testing.assert_close(
        torch.sigmoid(tables.left),
        left_concrete_probabilities,
    )
    zero_preds = example_predictions()
    zero_preds.left.zero_()
    zero_tables = compute_balance_score_tables(zero_preds)
    torch.testing.assert_close(
        torch.sigmoid(zero_tables.left),
        torch.full((1, 3, 3), 1.0 / 3.0),
    )

    values, mask = compute_balance_score_target_logits(
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


def test_balance_ss_materializes_concrete_pair_probabilities() -> None:
    preds = example_predictions()
    expanded_left = expand_direction_balance_probabilities(
        preds.left,
        preds.left_door_variant_idx,
        preds.right_door_variant_idx,
    )
    expected_left = torch.softmax(
        materialize_direction_balance_logits(
            preds.left,
            preds.left_door_variant_idx,
            preds.right_door_variant_idx,
        ),
        dim=-1,
    )

    torch.testing.assert_close(expanded_left, expected_left)
    torch.testing.assert_close(
        expanded_left.sum(dim=-1),
        torch.ones((1, 3)),
    )
    expected_ss = sum(
        torch.sum(expand_direction_balance_probabilities(logits, source_idx, target_idx).square())
        for logits, source_idx, target_idx in (
            (preds.left, preds.left_door_variant_idx, preds.right_door_variant_idx),
            (preds.right, preds.right_door_variant_idx, preds.left_door_variant_idx),
            (preds.up, preds.up_door_variant_idx, preds.down_door_variant_idx),
            (preds.down, preds.down_door_variant_idx, preds.up_door_variant_idx),
        )
    )
    torch.testing.assert_close(compute_balance_door_match_ss(preds), expected_ss)


def main() -> None:
    test_balance_model_outputs_direction_local_variant_pairs()
    test_balance_loss_maps_concrete_matches_to_variant_pairs()
    test_balance_target_scores_expand_variant_probabilities_to_concrete_matches()
    test_proposal_balance_residual_adds_both_door_directions()
    test_balance_ss_materializes_concrete_pair_probabilities()


if __name__ == "__main__":
    main()
