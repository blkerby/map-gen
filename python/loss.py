from dataclasses import dataclass

import torch

from env import AREA_COUNT, DoorMatches, StepOutcomes
from model import BalancePredictions, Predictions


@dataclass
class LossConfig:
    door_weight: float
    connection_weight: float
    toilet_weight: float
    phantoon_pair_weight: float
    phantoon_area_weight: float
    vanilla_area_weight: float
    balance_weight: float
    area_balance_weight: float
    toilet_balance_weight: float
    avg_frontiers_weight: float
    graph_diameter_weight: float
    save_distance_weight: float
    refill_distance_weight: float
    missing_connect_utility_weight: float
    area_crossing_weight: float
    area_size_weight: float
    area_map_station_weight: float
    area_x_weight: float
    area_y_weight: float
    map_width: int
    map_height: int
    distance_proximity_scale: float


@dataclass
class LossBreakdown:
    total: torch.Tensor
    door: torch.Tensor
    connection: torch.Tensor
    toilet: torch.Tensor
    phantoon_pair: torch.Tensor
    phantoon_area: torch.Tensor
    vanilla_area: torch.Tensor
    balance: torch.Tensor
    area_balance: torch.Tensor
    toilet_balance: torch.Tensor
    avg_frontiers: torch.Tensor
    graph_diameter: torch.Tensor
    save_distance: torch.Tensor
    refill_distance: torch.Tensor
    missing_connect_utility: torch.Tensor
    area_crossings: torch.Tensor
    area_size: torch.Tensor
    area_map_station: torch.Tensor
    area_x: torch.Tensor
    area_y: torch.Tensor
    door_contribution: torch.Tensor
    connection_contribution: torch.Tensor
    toilet_contribution: torch.Tensor
    phantoon_pair_contribution: torch.Tensor
    phantoon_area_contribution: torch.Tensor
    vanilla_area_contribution: torch.Tensor
    balance_contribution: torch.Tensor
    area_balance_contribution: torch.Tensor
    toilet_balance_contribution: torch.Tensor
    avg_frontiers_contribution: torch.Tensor
    graph_diameter_contribution: torch.Tensor
    save_distance_contribution: torch.Tensor
    refill_distance_contribution: torch.Tensor
    missing_connect_utility_contribution: torch.Tensor
    area_crossings_contribution: torch.Tensor
    area_size_contribution: torch.Tensor
    area_map_station_contribution: torch.Tensor
    area_x_contribution: torch.Tensor
    area_y_contribution: torch.Tensor


@dataclass
class BalancePriceTables:
    left: torch.Tensor
    right: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor
    toilet_crossed_room: torch.Tensor
    room_area: torch.Tensor


def masked_binary_cross_entropy_loss(
    preds: torch.Tensor, outcomes: torch.Tensor, mask: torch.Tensor, weight: float
) -> torch.Tensor:
    mask = (mask & (outcomes >= 0)).to(preds.dtype)
    binary_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        preds, outcomes.to(preds.dtype), reduction="none"
    )
    return weight * torch.sum(binary_loss * mask), weight * torch.sum(mask)


def masked_bernoulli_kl_loss(
    logits: torch.Tensor,
    target_logits: torch.Tensor,
    mask: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    logits = logits.to(torch.float32)
    mask = mask.to(logits.dtype)
    target_logits = target_logits.detach().to(logits.dtype)
    target_prob = torch.sigmoid(target_logits)
    prediction_cross_entropy = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        target_prob,
        reduction="none",
    )
    target_entropy = -(
        target_prob * torch.nn.functional.logsigmoid(target_logits)
        + (1.0 - target_prob) * torch.nn.functional.logsigmoid(-target_logits)
    )
    return (
        weight * torch.sum((prediction_cross_entropy - target_entropy) * mask),
        weight * torch.sum(mask),
    )


def masked_offset_bernoulli_kl_loss(
    logits: torch.Tensor,
    target_logits: torch.Tensor,
    logit_offset: torch.Tensor,
    mask: torch.Tensor,
    weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return masked_bernoulli_kl_loss(
        logits + logit_offset,
        target_logits + logit_offset,
        mask,
        weight,
    )


def masked_mse_loss(
    preds: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = mask.to(torch.float32)
    error = preds.to(torch.float32) - target.to(torch.float32)
    return weight * torch.sum(error.square() * mask), weight * torch.sum(mask)


def masked_cross_entropy_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = mask.to(torch.bool)
    if not torch.any(mask):
        return torch.sum(logits) * 0.0, logits.new_tensor(0.0)
    loss = torch.nn.functional.cross_entropy(
        logits[mask].to(torch.float32),
        target[mask].to(torch.int64),
        reduction="sum",
    )
    return weight * loss, weight * torch.sum(mask).to(logits.dtype)


def compute_loss_breakdown(
    preds: Predictions,
    outcomes: StepOutcomes,
    mask: torch.Tensor,
    vanilla_area_constraint_mask: torch.Tensor,
    balance_score_target: torch.Tensor,
    balance_score_mask: torch.Tensor,
    area_balance_score_target: torch.Tensor,
    area_balance_score_mask: torch.Tensor,
    toilet_balance_score_target: torch.Tensor,
    toilet_balance_score_mask: torch.Tensor,
    avg_frontiers_target: torch.Tensor,
    avg_frontiers_mask: torch.Tensor,
    graph_diameter_target: torch.Tensor,
    graph_diameter_mask: torch.Tensor,
    save_to_room_utility_target: torch.Tensor,
    save_from_room_utility_target: torch.Tensor,
    save_utility_mask: torch.Tensor,
    refill_to_room_utility_target: torch.Tensor,
    refill_from_room_utility_target: torch.Tensor,
    refill_utility_mask: torch.Tensor,
    missing_connect_utility_target: torch.Tensor,
    missing_connect_utility_mask: torch.Tensor,
    area_crossings_target: torch.Tensor,
    area_size_target: torch.Tensor,
    area_map_station_target: torch.Tensor,
    area_x_target: torch.Tensor,
    area_y_target: torch.Tensor,
    area_mask: torch.Tensor,
    area_coordinate_mask: torch.Tensor,
    area_crossings_mask: torch.Tensor,
    config: LossConfig,
) -> LossBreakdown:
    door_loss, door_wt = masked_binary_cross_entropy_loss(
        preds.door_invalid, outcomes.door_invalid, mask, config.door_weight
    )
    conn_loss, conn_wt = masked_binary_cross_entropy_loss(
        preds.connection_invalid, outcomes.connection_invalid, mask, config.connection_weight
    )
    toilet_loss, toilet_wt = masked_binary_cross_entropy_loss(
        preds.toilet_invalid, outcomes.toilet_invalid, mask.squeeze(-1), config.toilet_weight
    )
    phantoon_pair_loss, phantoon_pair_wt = masked_binary_cross_entropy_loss(
        preds.phantoon_pair_invalid,
        outcomes.phantoon_pair_invalid,
        mask.squeeze(-1),
        config.phantoon_pair_weight,
    )
    phantoon_area_loss, phantoon_area_wt = masked_binary_cross_entropy_loss(
        preds.phantoon_area_invalid,
        outcomes.phantoon_area_invalid,
        mask.squeeze(-1),
        config.phantoon_area_weight,
    )
    vanilla_area_loss, vanilla_area_wt = masked_binary_cross_entropy_loss(
        preds.vanilla_area_invalid,
        outcomes.vanilla_area_invalid,
        mask & vanilla_area_constraint_mask.unsqueeze(1),
        config.vanilla_area_weight,
    )
    balance_loss, balance_wt = masked_mse_loss(
        preds.balance_score,
        balance_score_target,
        mask & balance_score_mask,
        config.balance_weight,
    )
    area_balance_loss, area_balance_wt = masked_mse_loss(
        preds.area_balance_score,
        area_balance_score_target,
        mask & area_balance_score_mask,
        config.area_balance_weight,
    )
    toilet_balance_loss, toilet_balance_wt = masked_mse_loss(
        preds.toilet_balance_score,
        toilet_balance_score_target,
        mask.squeeze(-1) & toilet_balance_score_mask,
        config.toilet_balance_weight,
    )
    avg_frontiers_mask = avg_frontiers_mask.to(torch.float32)
    avg_frontiers_error = preds.avg_frontiers.to(torch.float32) - avg_frontiers_target.to(
        torch.float32
    )
    avg_frontiers_loss = config.avg_frontiers_weight * torch.sum(
        avg_frontiers_error.square() * avg_frontiers_mask
    )
    avg_frontiers_wt = config.avg_frontiers_weight * torch.sum(avg_frontiers_mask)
    graph_diameter_loss, graph_diameter_wt = masked_mse_loss(
        preds.graph_diameter,
        graph_diameter_target,
        graph_diameter_mask,
        config.graph_diameter_weight,
    )
    save_to_room_loss, save_to_room_wt = masked_mse_loss(
        preds.save_to_room_utility,
        save_to_room_utility_target,
        save_utility_mask,
        config.save_distance_weight,
    )
    save_from_room_loss, save_from_room_wt = masked_mse_loss(
        preds.save_from_room_utility,
        save_from_room_utility_target,
        save_utility_mask,
        config.save_distance_weight,
    )
    save_distance_loss = save_to_room_loss + save_from_room_loss
    save_distance_wt = save_to_room_wt + save_from_room_wt
    refill_to_room_loss, refill_to_room_wt = masked_mse_loss(
        preds.refill_to_room_utility,
        refill_to_room_utility_target,
        refill_utility_mask,
        config.refill_distance_weight,
    )
    refill_from_room_loss, refill_from_room_wt = masked_mse_loss(
        preds.refill_from_room_utility,
        refill_from_room_utility_target,
        refill_utility_mask,
        config.refill_distance_weight,
    )
    refill_distance_loss = refill_to_room_loss + refill_from_room_loss
    refill_distance_wt = refill_to_room_wt + refill_from_room_wt
    missing_connect_utility_loss, missing_connect_utility_wt = masked_mse_loss(
        preds.missing_connect_utility,
        missing_connect_utility_target,
        missing_connect_utility_mask,
        config.missing_connect_utility_weight,
    )
    area_crossings_loss, area_crossings_wt = masked_mse_loss(
        preds.area_crossings,
        area_crossings_target,
        area_crossings_mask,
        config.area_crossing_weight,
    )
    area_size_loss, area_size_wt = masked_cross_entropy_loss(
        preds.area_size,
        area_size_target,
        area_mask,
        config.area_size_weight,
    )
    area_map_station_loss, area_map_station_wt = masked_cross_entropy_loss(
        preds.area_map_station_count,
        area_map_station_target,
        area_mask,
        config.area_map_station_weight,
    )
    area_x_loss, area_x_wt = masked_mse_loss(
        preds.area_x,
        area_x_target,
        area_coordinate_mask,
        config.area_x_weight,
    )
    area_y_loss, area_y_wt = masked_mse_loss(
        preds.area_y,
        area_y_target,
        area_coordinate_mask,
        config.area_y_weight,
    )
    total_weight = (
        door_wt
        + conn_wt
        + toilet_wt
        + phantoon_pair_wt
        + phantoon_area_wt
        + vanilla_area_wt
        + balance_wt
        + area_balance_wt
        + toilet_balance_wt
        + avg_frontiers_wt
        + graph_diameter_wt
        + save_distance_wt
        + refill_distance_wt
        + missing_connect_utility_wt
        + area_crossings_wt
        + area_size_wt
        + area_map_station_wt
        + area_x_wt
        + area_y_wt
        + 1e-15
    )
    door_contribution = door_loss / total_weight
    connection_contribution = conn_loss / total_weight
    toilet_contribution = toilet_loss / total_weight
    phantoon_pair_contribution = phantoon_pair_loss / total_weight
    phantoon_area_contribution = phantoon_area_loss / total_weight
    vanilla_area_contribution = vanilla_area_loss / total_weight
    balance_contribution = balance_loss / total_weight
    area_balance_contribution = area_balance_loss / total_weight
    toilet_balance_contribution = toilet_balance_loss / total_weight
    avg_frontiers_contribution = avg_frontiers_loss / total_weight
    graph_diameter_contribution = graph_diameter_loss / total_weight
    save_distance_contribution = save_distance_loss / total_weight
    refill_distance_contribution = refill_distance_loss / total_weight
    missing_connect_utility_contribution = missing_connect_utility_loss / total_weight
    area_crossings_contribution = area_crossings_loss / total_weight
    area_size_contribution = area_size_loss / total_weight
    area_map_station_contribution = area_map_station_loss / total_weight
    area_x_contribution = area_x_loss / total_weight
    area_y_contribution = area_y_loss / total_weight
    mean_loss = (
        door_contribution
        + connection_contribution
        + toilet_contribution
        + phantoon_pair_contribution
        + phantoon_area_contribution
        + vanilla_area_contribution
        + balance_contribution
        + area_balance_contribution
        + toilet_balance_contribution
        + avg_frontiers_contribution
        + graph_diameter_contribution
        + save_distance_contribution
        + refill_distance_contribution
        + missing_connect_utility_contribution
        + area_crossings_contribution
        + area_size_contribution
        + area_map_station_contribution
        + area_x_contribution
        + area_y_contribution
    )
    return LossBreakdown(
        total=mean_loss,
        door=door_loss / (door_wt + 1e-15),
        connection=conn_loss / (conn_wt + 1e-15),
        toilet=toilet_loss / (toilet_wt + 1e-15),
        phantoon_pair=phantoon_pair_loss / (phantoon_pair_wt + 1e-15),
        phantoon_area=phantoon_area_loss / (phantoon_area_wt + 1e-15),
        vanilla_area=vanilla_area_loss / (vanilla_area_wt + 1e-15),
        balance=balance_loss / (balance_wt + 1e-15),
        area_balance=area_balance_loss / (area_balance_wt + 1e-15),
        toilet_balance=toilet_balance_loss / (toilet_balance_wt + 1e-15),
        avg_frontiers=avg_frontiers_loss / (avg_frontiers_wt + 1e-15),
        graph_diameter=graph_diameter_loss / (graph_diameter_wt + 1e-15),
        save_distance=save_distance_loss / (save_distance_wt + 1e-15),
        refill_distance=refill_distance_loss / (refill_distance_wt + 1e-15),
        missing_connect_utility=(
            missing_connect_utility_loss / (missing_connect_utility_wt + 1e-15)
        ),
        area_crossings=area_crossings_loss / (area_crossings_wt + 1e-15),
        area_size=area_size_loss / (area_size_wt + 1e-15),
        area_map_station=area_map_station_loss / (area_map_station_wt + 1e-15),
        area_x=area_x_loss / (area_x_wt + 1e-15),
        area_y=area_y_loss / (area_y_wt + 1e-15),
        door_contribution=door_contribution,
        connection_contribution=connection_contribution,
        toilet_contribution=toilet_contribution,
        phantoon_pair_contribution=phantoon_pair_contribution,
        phantoon_area_contribution=phantoon_area_contribution,
        vanilla_area_contribution=vanilla_area_contribution,
        balance_contribution=balance_contribution,
        area_balance_contribution=area_balance_contribution,
        toilet_balance_contribution=toilet_balance_contribution,
        avg_frontiers_contribution=avg_frontiers_contribution,
        graph_diameter_contribution=graph_diameter_contribution,
        save_distance_contribution=save_distance_contribution,
        refill_distance_contribution=refill_distance_contribution,
        missing_connect_utility_contribution=missing_connect_utility_contribution,
        area_crossings_contribution=area_crossings_contribution,
        area_size_contribution=area_size_contribution,
        area_map_station_contribution=area_map_station_contribution,
        area_x_contribution=area_x_contribution,
        area_y_contribution=area_y_contribution,
    )


def materialize_direction_balance_logits(
    logits: torch.Tensor,
    source_door_variant_idx: torch.Tensor,
    target_door_variant_idx: torch.Tensor,
) -> torch.Tensor:
    return logits[:, source_door_variant_idx, :][:, :, target_door_variant_idx]


def materialize_direction_balance_compatibility(
    door_variant_compatibility: torch.Tensor,
    source_global_door_variant_idx: torch.Tensor,
    target_global_door_variant_idx: torch.Tensor,
    source_door_variant_idx: torch.Tensor,
    target_door_variant_idx: torch.Tensor,
) -> torch.Tensor:
    source_variant_idx = source_global_door_variant_idx[source_door_variant_idx]
    target_variant_idx = target_global_door_variant_idx[target_door_variant_idx]
    return door_variant_compatibility[
        source_variant_idx.unsqueeze(1),
        target_variant_idx.unsqueeze(0),
    ]


def compute_balance_loss(
    preds: BalancePredictions,
    door_matches: DoorMatches,
    toilet_crossed_room_idx: torch.Tensor,
    room_area: torch.Tensor,
    area_probability: torch.Tensor,
    area_dual_mask: torch.Tensor,
    record_weight: torch.Tensor,
    door_eta: float,
    toilet_eta: float,
    area_eta: float,
    price_limit: float,
) -> torch.Tensor:
    tables = compute_balance_price_tables(
        preds,
        area_probability,
        area_dual_mask,
        price_limit,
    )
    door_residual_per_record = tables.left.new_zeros(record_weight.shape)
    door_count_per_record = tables.left.new_zeros(record_weight.shape)
    for prices, targets, compatibility in (
        (
            tables.left,
            door_matches.left,
            materialize_direction_balance_compatibility(
                preds.door_variant_compatibility,
                preds.left_global_door_variant_idx,
                preds.right_global_door_variant_idx,
                preds.left_door_variant_idx,
                preds.right_door_variant_idx,
            ),
        ),
        (
            tables.right,
            door_matches.right,
            materialize_direction_balance_compatibility(
                preds.door_variant_compatibility,
                preds.right_global_door_variant_idx,
                preds.left_global_door_variant_idx,
                preds.right_door_variant_idx,
                preds.left_door_variant_idx,
            ),
        ),
        (
            tables.up,
            door_matches.up,
            materialize_direction_balance_compatibility(
                preds.door_variant_compatibility,
                preds.up_global_door_variant_idx,
                preds.down_global_door_variant_idx,
                preds.up_door_variant_idx,
                preds.down_door_variant_idx,
            ),
        ),
        (
            tables.down,
            door_matches.down,
            materialize_direction_balance_compatibility(
                preds.door_variant_compatibility,
                preds.down_global_door_variant_idx,
                preds.up_global_door_variant_idx,
                preds.down_door_variant_idx,
                preds.up_door_variant_idx,
            ),
        ),
    ):
        mask = targets >= 0
        if not torch.any(mask):
            continue
        if torch.any(targets[mask] >= prices.shape[-1]):
            raise ValueError("door balance target is out of range")
        safe_targets = targets.clamp(0, prices.shape[-1] - 1).to(torch.int64)
        observed_compatible = torch.gather(
            compatibility.unsqueeze(0).expand(targets.shape[0], -1, -1),
            -1,
            safe_targets.unsqueeze(-1),
        ).squeeze(-1)
        if torch.any(mask & ~observed_compatible):
            raise ValueError("observed door pairing is incompatible")
        selected = torch.gather(prices, -1, safe_targets.unsqueeze(-1)).squeeze(-1)
        feasible_count = compatibility.sum(dim=-1).clamp_min(1)
        residual_value = feasible_count * selected - prices.sum(dim=-1)
        door_residual_per_record += torch.sum(residual_value * mask, dim=1)
        door_count_per_record += torch.sum(mask, dim=1)

    toilet_mask = toilet_crossed_room_idx >= 0
    safe_toilet = toilet_crossed_room_idx.clamp(0, tables.toilet_crossed_room.shape[-1] - 1)
    if torch.any(toilet_mask & ~preds.toilet_compatibility[safe_toilet]):
        raise ValueError("observed Toilet crossing room is infeasible")
    toilet_selected = torch.gather(
        tables.toilet_crossed_room,
        -1,
        safe_toilet.unsqueeze(-1),
    ).squeeze(-1)
    toilet_residual = (
        preds.toilet_compatibility.sum() * toilet_selected - tables.toilet_crossed_room.sum(dim=-1)
    )
    area_mask = (room_area >= 0) & area_dual_mask
    safe_area = room_area.clamp(0, AREA_COUNT - 1).to(torch.int64)
    selected_area_price = torch.gather(
        tables.room_area,
        -1,
        safe_area.unsqueeze(-1),
    ).squeeze(-1)
    selected_area_probability = torch.gather(
        area_probability,
        -1,
        safe_area.unsqueeze(-1),
    ).squeeze(-1)
    if torch.any(area_mask & (selected_area_probability <= 0.0)):
        raise ValueError("observed room-area assignment has zero target probability")
    area_residual = selected_area_price / selected_area_probability.clamp_min(
        torch.finfo(torch.float32).tiny
    ) - tables.room_area.sum(dim=-1)
    total_record_weight = record_weight.sum().clamp_min(1.0)
    door_objective = (
        torch.sum(door_residual_per_record / door_count_per_record.clamp_min(1.0) * record_weight)
        / total_record_weight
    )
    toilet_objective = (
        torch.sum(toilet_residual * toilet_mask * record_weight) / total_record_weight
    )
    area_objective = (
        torch.sum(
            torch.sum(area_residual * area_mask, dim=1)
            / torch.sum(area_mask, dim=1).clamp_min(1.0)
            * record_weight
        )
        / total_record_weight
    )
    return -(door_eta * door_objective + toilet_eta * toilet_objective + area_eta * area_objective)


def direction_valid_match_balance_score_target_logits(
    logit_table: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = (targets >= 0) & (targets < logit_table.shape[-1])
    if logit_table.shape[-1] == 0:
        return logit_table.new_empty(targets.shape, dtype=torch.float32), mask
    safe_targets = targets.clamp(0, logit_table.shape[-1] - 1).to(torch.int64)
    concrete_logit_table = logit_table
    while concrete_logit_table.ndim < safe_targets.ndim + 1:
        concrete_logit_table = concrete_logit_table.unsqueeze(1)
    concrete_logit_table = concrete_logit_table.expand(
        *safe_targets.shape,
        concrete_logit_table.shape[-1],
    )
    target_logits = torch.gather(
        concrete_logit_table,
        -1,
        safe_targets.unsqueeze(-1),
    ).squeeze(-1)
    return target_logits.detach(), mask


def direction_balance_price_table(
    prices: torch.Tensor,
    source_door_variant_idx: torch.Tensor,
    target_door_variant_idx: torch.Tensor,
    source_global_door_variant_idx: torch.Tensor,
    target_global_door_variant_idx: torch.Tensor,
    door_variant_compatibility: torch.Tensor,
    price_limit: float,
) -> torch.Tensor:
    concrete_prices = materialize_direction_balance_logits(
        prices,
        source_door_variant_idx,
        target_door_variant_idx,
    ).to(torch.float32)
    compatibility = materialize_direction_balance_compatibility(
        door_variant_compatibility,
        source_global_door_variant_idx,
        target_global_door_variant_idx,
        source_door_variant_idx,
        target_door_variant_idx,
    )
    counts = compatibility.sum(dim=-1).clamp_min(1)
    means = torch.sum(
        concrete_prices * compatibility.unsqueeze(0),
        dim=-1,
    ) / counts.unsqueeze(0)
    centered = concrete_prices - means.unsqueeze(-1)
    return torch.where(
        compatibility.unsqueeze(0),
        centered.clamp(-price_limit, price_limit),
        0.0,
    )


def compute_balance_price_tables(
    preds: BalancePredictions,
    area_probability: torch.Tensor,
    area_dual_mask: torch.Tensor,
    price_limit: float,
) -> BalancePriceTables:
    direction_inputs = (
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
    left, right, up, down = (
        direction_balance_price_table(
            prices,
            source_idx,
            target_idx,
            source_global_idx,
            target_global_idx,
            preds.door_variant_compatibility,
            price_limit,
        )
        for prices, source_idx, target_idx, source_global_idx, target_global_idx in direction_inputs
    )
    toilet_mask = preds.toilet_compatibility.unsqueeze(0)
    toilet_count = toilet_mask.sum(dim=-1).clamp_min(1)
    toilet_mean = torch.sum(preds.toilet_crossed_room * toilet_mask, dim=-1) / toilet_count
    toilet = torch.where(
        toilet_mask,
        (preds.toilet_crossed_room - toilet_mean.unsqueeze(-1)).clamp(-price_limit, price_limit),
        0.0,
    )
    if area_probability.shape != preds.room_area.shape:
        raise ValueError("area_probability shape must match balance room-area prices")
    if area_dual_mask.shape != preds.room_area.shape[:2]:
        raise ValueError("area_dual_mask shape must match balance room rows")
    safe_probability = area_probability.clamp_min(torch.finfo(torch.float32).tiny)
    area_raw = preds.room_area.to(torch.float32) - safe_probability.log()
    area_mean = torch.sum(area_raw * area_probability, dim=-1, keepdim=True)
    room_area = torch.where(
        area_dual_mask.unsqueeze(-1),
        (area_raw - area_mean).clamp(-price_limit, price_limit),
        0.0,
    )
    return BalancePriceTables(
        left=left,
        right=right,
        up=up,
        down=down,
        toilet_crossed_room=toilet,
        room_area=room_area,
    )


def compute_room_area_balance_score_target_logits(
    tables: BalancePriceTables,
    room_area: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = room_area >= 0
    safe_room_area = room_area.clamp_min(0).to(torch.int64)
    target_logits = torch.gather(
        tables.room_area,
        -1,
        safe_room_area.unsqueeze(-1),
    ).squeeze(-1)
    return target_logits.detach(), mask


def compute_balance_score_target_logits(
    tables: BalancePriceTables,
    door_matches: DoorMatches,
) -> tuple[torch.Tensor, torch.Tensor]:
    values_and_masks = tuple(
        direction_valid_match_balance_score_target_logits(table, targets)
        for table, targets in (
            (tables.left, door_matches.left),
            (tables.right, door_matches.right),
            (tables.up, door_matches.up),
            (tables.down, door_matches.down),
        )
    )
    return (
        torch.cat([values for values, _ in values_and_masks], dim=-1),
        torch.cat([mask for _, mask in values_and_masks], dim=-1),
    )


def compute_step_balance_score_target_logits(
    tables: BalancePriceTables,
    door_match: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    left, right, up, down = torch.split(
        door_match,
        [
            tables.left.shape[-2],
            tables.right.shape[-2],
            tables.up.shape[-2],
            tables.down.shape[-2],
        ],
        dim=-1,
    )
    left_values, left_mask = direction_valid_match_balance_score_target_logits(
        tables.left,
        left,
    )
    right_values, right_mask = direction_valid_match_balance_score_target_logits(
        tables.right,
        right,
    )
    up_values, up_mask = direction_valid_match_balance_score_target_logits(
        tables.up,
        up,
    )
    down_values, down_mask = direction_valid_match_balance_score_target_logits(
        tables.down,
        down,
    )
    return (
        torch.cat([left_values, right_values, up_values, down_values], dim=-1),
        torch.cat([left_mask, right_mask, up_mask, down_mask], dim=-1),
    )


def add_direction_proposal_balance_score_table(
    proposal_score_table: torch.Tensor,
    forward_score_table: torch.Tensor,
    reverse_score_table: torch.Tensor,
    source_representative_door_idx: torch.Tensor,
    target_representative_door_idx: torch.Tensor,
    source_global_door_variant_idx: torch.Tensor,
    target_global_door_variant_idx: torch.Tensor,
) -> None:
    if source_global_door_variant_idx.numel() == 0 or target_global_door_variant_idx.numel() == 0:
        return
    # A placement fixes both directed sides of its door match.
    proposal_score_table[
        :,
        source_global_door_variant_idx.unsqueeze(1),
        target_global_door_variant_idx.unsqueeze(0),
    ] = forward_score_table[:, source_representative_door_idx, :][
        :, :, target_representative_door_idx
    ] + reverse_score_table[:, target_representative_door_idx, :][
        :, :, source_representative_door_idx
    ].transpose(1, 2)


def first_concrete_door_idx_by_variant(
    door_variant_idx: torch.Tensor,
    variant_count: int,
) -> torch.Tensor:
    concrete_door_idx = torch.arange(
        door_variant_idx.numel(),
        dtype=torch.int64,
        device=door_variant_idx.device,
    )
    first_idx = torch.full(
        [variant_count],
        door_variant_idx.numel(),
        dtype=torch.int64,
        device=door_variant_idx.device,
    )
    return first_idx.scatter_reduce(
        0,
        door_variant_idx,
        concrete_door_idx,
        reduce="amin",
        include_self=True,
    )


def compute_proposal_balance_score_table(
    preds: BalancePredictions,
    tables: BalancePriceTables,
    num_door_variants: int,
) -> torch.Tensor:
    proposal_score_table = torch.zeros(
        [tables.left.shape[0], num_door_variants, num_door_variants],
        dtype=torch.float32,
        device=tables.left.device,
    )
    left_representative_door_idx = first_concrete_door_idx_by_variant(
        preds.left_door_variant_idx,
        preds.left_global_door_variant_idx.numel(),
    )
    right_representative_door_idx = first_concrete_door_idx_by_variant(
        preds.right_door_variant_idx,
        preds.right_global_door_variant_idx.numel(),
    )
    up_representative_door_idx = first_concrete_door_idx_by_variant(
        preds.up_door_variant_idx,
        preds.up_global_door_variant_idx.numel(),
    )
    down_representative_door_idx = first_concrete_door_idx_by_variant(
        preds.down_door_variant_idx,
        preds.down_global_door_variant_idx.numel(),
    )
    direction_inputs = (
        (
            tables.left,
            tables.right,
            left_representative_door_idx,
            right_representative_door_idx,
            preds.left_global_door_variant_idx,
            preds.right_global_door_variant_idx,
        ),
        (
            tables.right,
            tables.left,
            right_representative_door_idx,
            left_representative_door_idx,
            preds.right_global_door_variant_idx,
            preds.left_global_door_variant_idx,
        ),
        (
            tables.up,
            tables.down,
            up_representative_door_idx,
            down_representative_door_idx,
            preds.up_global_door_variant_idx,
            preds.down_global_door_variant_idx,
        ),
        (
            tables.down,
            tables.up,
            down_representative_door_idx,
            up_representative_door_idx,
            preds.down_global_door_variant_idx,
            preds.up_global_door_variant_idx,
        ),
    )
    for (
        forward_score_table,
        reverse_score_table,
        source_representative_door_idx,
        target_representative_door_idx,
        source_global_door_variant_idx,
        target_global_door_variant_idx,
    ) in direction_inputs:
        add_direction_proposal_balance_score_table(
            proposal_score_table,
            forward_score_table,
            reverse_score_table,
            source_representative_door_idx,
            target_representative_door_idx,
            source_global_door_variant_idx,
            target_global_door_variant_idx,
        )
    return proposal_score_table


def compute_proposal_balance_score_residual(
    proposal_score_table: torch.Tensor,
    frontier_door_variant: torch.Tensor,
    row_snapshot_idx: torch.Tensor,
) -> torch.Tensor:
    device = proposal_score_table.device
    frontier_door_variant = frontier_door_variant.to(device=device, dtype=torch.int64)
    row_snapshot_idx = row_snapshot_idx.to(device=device, dtype=torch.int64)
    variant_residual = -proposal_score_table[row_snapshot_idx, frontier_door_variant]
    return (
        variant_residual.unsqueeze(-1)
        .expand(-1, -1, AREA_COUNT)
        .reshape(
            frontier_door_variant.numel(),
            proposal_score_table.shape[-1] * AREA_COUNT,
        )
    )


def compute_proposal_area_balance_score_table(
    room_area_score_table: torch.Tensor,
    exempt_room: torch.Tensor,
    door_room_idx: torch.Tensor,
    door_output_variant_idx: torch.Tensor,
    num_door_variants: int,
) -> torch.Tensor:
    representative_door_idx = first_concrete_door_idx_by_variant(
        door_output_variant_idx,
        num_door_variants,
    )
    proposal_room_idx = door_room_idx[representative_door_idx]
    scores = room_area_score_table[:, proposal_room_idx]
    return torch.where(
        exempt_room[:, proposal_room_idx].unsqueeze(-1),
        0.0,
        scores,
    ).flatten(1)


def compute_proposal_area_balance_score_residual(
    proposal_score_table: torch.Tensor,
    row_snapshot_idx: torch.Tensor,
) -> torch.Tensor:
    device = proposal_score_table.device
    row_snapshot_idx = row_snapshot_idx.to(device=device, dtype=torch.int64)
    return -proposal_score_table[row_snapshot_idx]


def compute_toilet_balance_score_target_logits(
    tables: BalancePriceTables,
    toilet_crossed_room_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = toilet_crossed_room_idx >= 0
    safe_target = toilet_crossed_room_idx.clamp(0, tables.toilet_crossed_room.shape[-1] - 1).to(
        torch.int64
    )
    target = torch.gather(
        tables.toilet_crossed_room,
        -1,
        safe_target.unsqueeze(-1),
    ).squeeze(-1)
    return target.detach(), mask
