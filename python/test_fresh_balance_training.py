from types import SimpleNamespace
from unittest.mock import patch

import torch

from env import Actions, DoorMatches, EpisodeData
from learn import train_balance_replay
from train_config import GENERATION_VARIABLE_FLOAT_FIELDS


class FakeBalanceEngine:
    def __init__(self) -> None:
        self.rooms = [{}]
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


class FakeBalanceExperience:
    def __init__(self, episode_data: EpisodeData) -> None:
        self.num_files = 1
        self.episode_data = episode_data
        self.requested_files = []

    def read_balance_files(self, file_num_list: list[int]) -> tuple[Actions, torch.Tensor]:
        self.requested_files.append(file_num_list)
        return self.episode_data.actions, self.episode_data.generation_variable_floats


def example_episode_data(value: int) -> EpisodeData:
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
        generation_variable_floats=torch.zeros(
            (2, len(GENERATION_VARIABLE_FLOAT_FIELDS)),
        ),
    )


def test_balance_replay_includes_history_and_applies_linear_age_weight() -> None:
    history = example_episode_data(1)
    fresh = example_episode_data(2)
    experience = FakeBalanceExperience(history)
    engine = FakeBalanceEngine()
    context = SimpleNamespace(
        config=SimpleNamespace(
            balance_train=SimpleNamespace(
                replay_window_rounds=3,
                window_weight="linear",
                batch_size=2,
            ),
            generation=SimpleNamespace(num_iterations=1, num_environments=2),
            train=SimpleNamespace(fresh_pass_factor=0.0, batch_size=1),
        ),
        step_config=SimpleNamespace(balance_optimizer=object()),
        experience=experience,
        train_batch_envs=[SimpleNamespace(engine=engine)],
        device=torch.device("cpu"),
        num_rooms=1,
        balance_model=object(),
        balance_optimizer=object(),
    )

    with (
        patch("learn.set_optimizer_lrs"),
        patch("learn.train_balance_batch", return_value=1.0) as train_batch,
    ):
        loss = train_balance_replay(context, fresh)

    assert loss == 1.0
    assert experience.requested_files == [[0]]
    assert engine.batch_sizes == [2, 2]
    weights = torch.cat([call.kwargs["record_weight"] for call in train_batch.call_args_list])
    torch.testing.assert_close(
        weights.sort().values,
        torch.tensor([2.0 / 3.0, 2.0 / 3.0, 1.0, 1.0]),
    )


if __name__ == "__main__":
    test_balance_replay_includes_history_and_applies_linear_age_weight()
