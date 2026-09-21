import copy
import inspect
import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from env import Engine
from generate import run_generation_groups
from learn import (
    distance_proximity_utility,
    generation_area_balance_targets,
    prepare_feature_batch,
    train_feature_batch_backward,
)
from loss import LossConfig, compute_loss_breakdown
from model import FrontierModel
from model_loading import create_balance_model, frontier_model_kwargs
from train import create_generate_config
from train_config import Config, instantiate_scheduleable_config


UTILITY_FAMILIES = ("save_to_room", "save_from_room", "refill_to_room", "refill_from_room")
LOSS_SIGNATURE = inspect.signature(compute_loss_breakdown)


class CheckUtilitySupervision:
    def __init__(
        self,
        active: torch.Tensor,
        expected: dict[str, torch.Tensor],
        area_masks: list[torch.Tensor],
        absent_rooms: torch.Tensor,
        order_targets: list[torch.Tensor],
        order_masks: list[torch.Tensor],
    ):
        self.active = active.unsqueeze(1)
        self.expected = expected
        self.checked = 0
        self.area_masks = area_masks
        self.absent_rooms = absent_rooms.unsqueeze(1)
        self.absent_area_gradients_checked = 0
        self.order_targets = order_targets
        self.order_masks = order_masks

    def __call__(self, *args, **kwargs):
        bound = LOSS_SIGNATURE.bind(*args, **kwargs)
        values = bound.arguments
        torch.testing.assert_close(
            values["order_balance_score_target"], self.order_targets[self.checked]
        )
        torch.testing.assert_close(
            values["order_balance_score_mask"], self.order_masks[self.checked]
        )
        assert values["save_utility_mask"].all()
        assert values["refill_utility_mask"].all()
        predictions = {}
        for family in UTILITY_FAMILIES:
            target = values[f"{family}_utility_target"]
            torch.testing.assert_close(target, self.expected[family])
            assert torch.count_nonzero(target[~self.active]) == 0
            predictions[f"{family}_utility"] = torch.full_like(target, 0.5, requires_grad=True)
        area_mask = values["area_balance_score_mask"]
        torch.testing.assert_close(area_mask, self.area_masks[self.checked])
        area_target = values["area_balance_score_target"]
        assert torch.count_nonzero(area_target[self.absent_rooms]) == 0
        area_prediction = torch.full_like(area_target, -4.0, requires_grad=True)
        predictions["area_balance_score"] = area_prediction
        # Use known positive predictions to verify the actual combined loss's
        # gradient direction, independent of random model initialization.
        values["preds"] = replace(values["preds"], **predictions)
        loss = compute_loss_breakdown(**values)
        gradients = torch.autograd.grad(loss.total, tuple(predictions.values()), retain_graph=True)
        for gradient in gradients[:-1]:
            assert torch.isfinite(gradient).all()
            assert (gradient[~self.active] > 0).all()
        area_gradient = gradients[-1]
        assert torch.count_nonzero(area_gradient[~area_mask]) == 0
        absent_supervised = self.absent_rooms & area_mask
        assert (area_gradient[absent_supervised] < 0).all()
        self.absent_area_gradients_checked += int(absent_supervised.sum())
        self.checked += 1
        return loss


def test_terminally_absent_rooms_receive_zero_utility_supervision() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs/debug.json"
    torch.manual_seed(1984)
    config = instantiate_scheduleable_config(
        Config.model_validate_json(Path(config_path).read_text()), 0
    )
    config.model.compile = False
    config.model.embedding_width = 32
    config.model.global_embedding_width = 32
    config.model.hidden_width = 32
    config.model.proposal_hidden_widths = [32]
    config.model.missing_connect_query_hidden_width = 32
    config.model.missing_connect_query_frontier_width = 8
    config.model.missing_connect_query_distance_width = 4
    config.model.utility_query_hidden_width = 16
    config.model.utility_query_frontier_width = 8
    config.model.num_layers = 2
    config.balance_model.hidden_width = 32
    config.balance_model.num_layers = 1
    for name in (
        "global_room_position",
        "lookahead_outcomes",
        "frontier_position",
        "frontier_door_variant",
        "frontier_neighbor_position_embedding",
    ):
        if getattr(config.features, name):
            setattr(config.features, name, 16)
    config.generation.num_environments = 8
    config.generation.num_iterations = 1
    config.generation.num_devices = 1
    config.generation.pipeline_groups = 1
    config.generation.recommended_candidates = 4
    config.generation.shortlist_candidates = 64
    config.generation.num_scored_invalid_candidates = 4
    config.generation.num_threads = 2
    config.generation.gpu_prefetch_batches = 0
    config.train.save_distance_weight = 1.0
    config.train.refill_distance_weight = 1.0
    config.train.area_balance_weight = 1.0
    config.train.batch_size = 8
    config.balance_train.batch_size = 8
    rooms = json.loads(config.room_set.read_text())
    device = torch.device("cpu")
    engine = Engine(
        rooms, config.features, config.generation.min_area_size, config.generation.max_area_size
    )
    environments = [
        engine.create_environment_group(
            map_size=config.map_size,
            num_envs=8,
            candidate_spatial_cell_size=config.generation.candidate_spatial_cell_size,
            area_bounding_box_width=config.generation.area_bounding_box_width,
            area_bounding_box_height=config.generation.area_bounding_box_height,
            seed=1234,
            frontier_neighbor_count=config.generation.frontier_neighbor_count,
            frontier_window_size=config.generation.frontier_window_size,
            num_threads=2,
            frontier_neighbor_algorithm=config.generation.frontier_neighbor_algorithm,
        )
        for _ in range(2)
    ]
    main = FrontierModel(**frontier_model_kwargs(config, rooms, engine))
    ema = copy.deepcopy(main).requires_grad_(False)
    balance = create_balance_model(config, rooms, engine, device)
    gen_balance = copy.deepcopy(balance).requires_grad_(False)
    gen_config = create_generate_config(config, rooms, len(rooms), 8, device, False)
    episode_data, outcomes, _, proposals, captured, _, _ = run_generation_groups(
        environments[:1],
        ema,
        gen_balance,
        [gen_config],
        device,
        verify_outcome_consistency=True,
        capture_generated_features=True,
    )
    # Distinct nonzero prices expose incorrect rank masks and omitted failures.
    with torch.no_grad():
        balance.order_net[-1].bias.copy_(torch.arange(42.0))
    loss_config = LossConfig(
        **{
            name: getattr(config.train, name)
            for name in LossConfig.__dataclass_fields__
            if name.endswith("_weight")
        },
        map_width=config.map_size[0],
        map_height=config.map_size[1],
        distance_proximity_scale=config.distance_proximity_scale,
    )
    context = SimpleNamespace(
        config=config,
        step_config=config,
        device=device,
        num_rooms=len(rooms),
        train_batch_envs=environments[1:],
        balance_model=balance,
        main_model=main,
        loss_config=loss_config,
    )
    prepared = prepare_feature_batch(
        config,
        device,
        "fresh",
        episode_data,
        outcomes,
        proposals,
        environments[1],
        len(rooms),
        captured.feature_batches,
    )
    active = outcomes.end_outcomes.active_room_part_mask.bool()
    assert active.any() and (~active).any()
    expected = {
        family: torch.where(
            active,
            distance_proximity_utility(
                getattr(outcomes.end_outcomes, f"{family}_distance"),
                getattr(outcomes.end_outcomes, f"{family}_distance_mask"),
                config.distance_proximity_scale,
            ),
            0.0,
        ).unsqueeze(1)
        for family in UTILITY_FAMILIES
    }
    area_dual_mask = generation_area_balance_targets(
        rooms, episode_data.generation_variable_floats
    ).dual_mask
    area_masks = [
        (~batch.features.global_features.room_placed.bool() & area_dual_mask).unsqueeze(1)
        for batch in prepared.feature_batches
    ]
    terminal_order_costs = []
    for room_indices, areas in zip(
        episode_data.actions.room_idx.tolist(), episode_data.actions.room_area.tolist(), strict=True
    ):
        order = []
        for room, area in zip(room_indices, areas, strict=True):
            if room < len(rooms) and area < 6 and area not in order:
                order.append(area)
        terminal_order_costs.append(
            [order[rank] - 2.5 if rank < len(order) else 36.0 + rank for rank in range(6)]
        )
    order_targets = []
    order_masks = []
    for batch in prepared.feature_batches:
        started = batch.features.global_features.area_used.long().sum(-1).tolist()
        order_targets.append(torch.tensor(
            [[sum(costs[count:])] for costs, count in zip(terminal_order_costs, started, strict=True)],
            dtype=torch.float32,
        ))
        order_masks.append(torch.tensor([[count < 6] for count in started]))
    checker = CheckUtilitySupervision(
        active=active,
        expected=expected,
        area_masks=area_masks + [torch.zeros_like(mask) for mask in area_masks],
        absent_rooms=prepared.room_area < 0,
        order_targets=order_targets + [torch.zeros_like(target) for target in order_targets],
        order_masks=order_masks + [torch.zeros_like(mask) for mask in order_masks],
    )
    # Deliberately give absent parts reachable distances: terminal absence must
    # override these values, independently of the engine's distance convention.
    for family in UTILITY_FAMILIES:
        getattr(outcomes.end_outcomes, f"{family}_distance")[~active] = 0
        getattr(outcomes.end_outcomes, f"{family}_distance_mask")[~active] = True
    for kind in ("fresh", "replay"):
        prepared.kind = kind
        with patch("learn.compute_loss_breakdown", side_effect=checker):
            result = train_feature_batch_backward(context, prepared, 1.0)
        assert math.isfinite(result.total)
    assert checker.checked > 0
    assert checker.absent_area_gradients_checked > 0


if __name__ == "__main__":
    torch.set_num_threads(2)
    test_terminally_absent_rooms_receive_zero_utility_supervision()
