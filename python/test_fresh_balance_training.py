from types import SimpleNamespace
from unittest.mock import patch

import torch

from learn import empty_main_loss_breakdown, train_batch_backward


def test_replay_batch_skips_balance_model_training() -> None:
    context = SimpleNamespace()
    prepared_batch = SimpleNamespace(kind="replay")
    with (
        patch(
            "learn.train_feature_batch_backward",
            return_value=empty_main_loss_breakdown(),
        ),
        patch("learn.train_balance_batch_backward") as train_balance,
    ):
        _, balance_loss = train_batch_backward(
            context,
            prepared_batch,
            main_loss_scale=1.0,
            balance_loss_scale=1.0,
        )

    train_balance.assert_not_called()
    assert balance_loss == 0.0


def test_fresh_batch_trains_balance_model() -> None:
    context = SimpleNamespace(
        balance_model=object(),
        train_batch_envs=[SimpleNamespace(engine=SimpleNamespace(rooms=[]))],
    )
    prepared_batch = SimpleNamespace(kind="fresh")
    with (
        patch(
            "learn.train_feature_batch_backward",
            return_value=empty_main_loss_breakdown(),
        ),
        patch(
            "learn.train_balance_batch_backward",
            return_value=torch.tensor(2.0),
        ) as train_balance,
    ):
        _, balance_loss = train_batch_backward(
            context,
            prepared_batch,
            main_loss_scale=1.0,
            balance_loss_scale=0.25,
        )

    train_balance.assert_called_once_with(
        context.balance_model,
        prepared_batch,
        [],
        0.25,
    )
    assert balance_loss == 2.0


def main() -> None:
    test_replay_batch_skips_balance_model_training()
    test_fresh_batch_trains_balance_model()


if __name__ == "__main__":
    main()
