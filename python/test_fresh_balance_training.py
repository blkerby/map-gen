from types import SimpleNamespace
from unittest.mock import patch

import torch

from env import Actions, DoorMatches, EpisodeData
from learn import train_balance_batch, train_balance_fresh
from train_config import GENERATION_VARIABLE_FLOAT_FIELDS


class FakeBalanceEngine:
    def __init__(self) -> None:
        self.rooms = [{"water": 1}]
        self.batch_sizes = []

    def compute_balance_targets(
        self,
        actions: Actions,
        device: torch.device,
    ) -> tuple[DoorMatches, torch.Tensor]:
        batch_size = actions.room_idx.shape[0]
        self.batch_sizes.append(batch_size)
        empty = torch.empty((batch_size, 0), dtype=torch.int64, device=device)
        return (
            DoorMatches(left=empty, right=empty, up=empty, down=empty),
            torch.full((batch_size,), -1, dtype=torch.int64, device=device),
        )


def example_episode_data(value: int) -> EpisodeData:
    generation_variable_floats = torch.zeros(
        (2, len(GENERATION_VARIABLE_FLOAT_FIELDS)),
    )
    if value == 2:
        reward_idx = GENERATION_VARIABLE_FLOAT_FIELDS.index("reward_maridia_water_1")
        generation_variable_floats[:, reward_idx] = 1.0
    actions = Actions(
        room_idx=torch.zeros((2, 1), dtype=torch.int64),
        room_x=torch.full((2, 1), value, dtype=torch.int64),
        room_y=torch.zeros((2, 1), dtype=torch.int64),
        room_area=torch.zeros((2, 1), dtype=torch.int64),
    )
    return EpisodeData(
        actions=actions,
        temperature=torch.zeros(2),
        recommended_candidates=torch.zeros(2, dtype=torch.int64),
        generation_variable_floats=generation_variable_floats,
    )


def test_balance_training_uses_only_fresh_data() -> None:
    fresh = example_episode_data(2)
    engine = FakeBalanceEngine()
    balance_ema_model = object()
    context = SimpleNamespace(
        config=SimpleNamespace(
            balance_train=SimpleNamespace(
                batch_size=2,
            ),
            generation=SimpleNamespace(num_iterations=1, num_environments=2),
            train=SimpleNamespace(fresh_pass_factor=0.0, batch_size=1),
        ),
        step_config=SimpleNamespace(
            balance_optimizer=object(),
            balance_train=SimpleNamespace(batch_size=2, ema_half_life_episodes=4.0),
            generation=SimpleNamespace(num_iterations=1, num_environments=2),
            train=SimpleNamespace(fresh_pass_factor=0.0, batch_size=1),
        ),
        train_batch_envs=[SimpleNamespace(engine=engine)],
        device=torch.device("cpu"),
        num_rooms=1,
        balance_model=object(),
        balance_ema_model=balance_ema_model,
        balance_optimizer=object(),
    )

    with (
        patch("learn.set_optimizer_lrs"),
        patch(
            "learn.torch.randperm",
            return_value=torch.tensor([1, 0]),
        ),
        patch("learn.train_balance_batch", return_value=1.0) as train_batch,
    ):
        loss = train_balance_fresh(context, fresh)

    assert loss == 1.0
    assert engine.batch_sizes == [2]
    assert train_batch.call_count == 1
    assert train_batch.call_args.kwargs["room_area_mask"].tolist() == [[False], [False]]
    torch.testing.assert_close(
        train_batch.call_args.kwargs["record_weight"],
        torch.ones(2),
    )
    assert train_batch.call_args.kwargs["balance_ema_model"] is balance_ema_model
    assert train_batch.call_args.kwargs["ema_half_life_episodes"] == 4.0


def test_balance_batch_updates_ema_after_optimizer_step() -> None:
    balance_model = torch.nn.Linear(1, 1, bias=False)
    balance_ema_model = torch.nn.Linear(1, 1, bias=False)
    balance_model.weight.data.fill_(1.0)
    balance_ema_model.weight.data.zero_()
    balance_ema_model.requires_grad_(False)
    optimizer = torch.optim.SGD(balance_model.parameters(), lr=0.1)

    with patch(
        "learn.compute_balance_loss",
        side_effect=lambda prediction, *_args: prediction.square().sum(),
    ):
        train_balance_batch(
            generation_variable_floats=torch.ones((1, 1)),
            door_matches=object(),
            toilet_crossed_room_idx=torch.zeros(1, dtype=torch.int64),
            room_area=torch.zeros((1, 1), dtype=torch.int64),
            room_area_mask=torch.ones((1, 1), dtype=torch.bool),
            record_weight=torch.ones(1),
            balance_model=balance_model,
            balance_ema_model=balance_ema_model,
            balance_optimizer=optimizer,
            ema_half_life_episodes=1.0,
        )

    torch.testing.assert_close(balance_model.weight, torch.tensor([[0.9]]))
    torch.testing.assert_close(balance_ema_model.weight, torch.tensor([[0.45]]))


if __name__ == "__main__":
    test_balance_training_uses_only_fresh_data()
    test_balance_batch_updates_ema_after_optimizer_step()
