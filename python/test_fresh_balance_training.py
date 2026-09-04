from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from env import AREA_COUNT, Actions, DoorMatches, EpisodeData
from learn import train_balance_fresh
from train_config import (
    GENERATION_VARIABLE_FLOAT_FIELDS,
    HEAT_WATER_FAMILIES,
    VANILLA_AREA_CONDITION_FIELDS,
)


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


def example_episode_data() -> EpisodeData:
    variables = torch.zeros((4, len(GENERATION_VARIABLE_FLOAT_FIELDS)))
    for area in range(AREA_COUNT):
        variables[:, GENERATION_VARIABLE_FLOAT_FIELDS.index(f"target_area_rooms_{area}")] = (
            1.0 / AREA_COUNT
        )
    for family in HEAT_WATER_FAMILIES:
        for tier in range(1, 4):
            variables[
                :, GENERATION_VARIABLE_FLOAT_FIELDS.index(f"{family}_preferred_probability_{tier}")
            ] = 1.0 / AREA_COUNT
    for name in VANILLA_AREA_CONDITION_FIELDS:
        variables[:, GENERATION_VARIABLE_FLOAT_FIELDS.index(name)] = 0.0
    actions = Actions(
        room_idx=torch.zeros((4, 1), dtype=torch.int64),
        room_x=torch.zeros((4, 1), dtype=torch.int64),
        room_y=torch.zeros((4, 1), dtype=torch.int64),
        room_area=torch.zeros((4, 1), dtype=torch.int64),
    )
    return EpisodeData(
        actions=actions,
        temperature=torch.ones(4),
        recommended_candidates=torch.zeros(4, dtype=torch.int64),
        generation_variable_floats=variables,
    )


def test_balance_training_chunks_one_round_into_one_optimizer_step() -> None:
    engine = FakeBalanceEngine()
    optimizer = Mock()
    balance_model = torch.nn.Linear(1, 1, bias=False)
    context = SimpleNamespace(
        step_config=SimpleNamespace(
            balance_train=SimpleNamespace(
                batch_size=2,
                door_eta=0.02,
                toilet_eta=0.03,
                area_eta=0.04,
                price_limit=20.0,
            ),
            generation=SimpleNamespace(num_iterations=1, num_environments=4),
            train=SimpleNamespace(fresh_pass_factor=0.0, batch_size=1),
        ),
        train_batch_envs=[SimpleNamespace(engine=engine)],
        device=torch.device("cpu"),
        num_rooms=1,
        balance_model=balance_model,
        balance_optimizer=optimizer,
    )

    with (
        patch("learn.torch.randperm", return_value=torch.arange(4)),
        patch("learn.train_balance_batch", return_value=1.0) as train_batch,
    ):
        loss = train_balance_fresh(context, example_episode_data())

    assert loss == 1.0
    assert engine.batch_sizes == [2, 2]
    assert train_batch.call_count == 2
    assert train_batch.call_args.kwargs["loss_scale"] == 0.5
    assert train_batch.call_args.kwargs["area_dual_mask"].all()
    optimizer.zero_grad.assert_called_once_with(set_to_none=True)
    optimizer.step.assert_called_once_with()


if __name__ == "__main__":
    test_balance_training_chunks_one_round_into_one_optimizer_step()
