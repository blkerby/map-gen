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
    order_balance_weight: float
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
    order_balance: torch.Tensor
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
    order_balance_contribution: torch.Tensor
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
    door_failure: torch.Tensor
    toilet_crossed_room: torch.Tensor
    toilet_failure: torch.Tensor
    room_area: torch.Tensor
    room_area_failure: torch.Tensor
    area_order: torch.Tensor
    area_order_failure: torch.Tensor


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
    order_balance_score_target: torch.Tensor,
    order_balance_score_mask: torch.Tensor,
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
    order_balance_loss, order_balance_wt = masked_mse_loss(
        preds.order_balance_score, order_balance_score_target,
        order_balance_score_mask, config.order_balance_weight,
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
        + order_balance_wt
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
    order_balance_contribution = order_balance_loss / total_weight
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
        + order_balance_contribution
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
        order_balance=order_balance_loss / (order_balance_wt + 1e-15),
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
        order_balance_contribution=order_balance_contribution,
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


@dataclass
class BalanceObjectiveTerms:
    observed_price: torch.Tensor
    squared_prices: torch.Tensor
    fourth_power_prices: torch.Tensor
    group_count: torch.Tensor


def terminal_balance_cost(
    success_prices: torch.Tensor,
    failure_price: torch.Tensor,
    outcome: torch.Tensor,
) -> torch.Tensor:
    """Select a terminal price; -1 denotes failure, including absent rooms/doors."""
    if torch.any((outcome < -1) | (outcome >= success_prices.shape[-1])):
        raise ValueError("terminal balance outcome is out of range")
    prices = torch.cat((success_prices, failure_price.unsqueeze(-1)), dim=-1)
    while prices.ndim < outcome.ndim + 1:
        prices = prices.unsqueeze(1)
    prices = prices.expand(*outcome.shape, prices.shape[-1])
    index = torch.where(outcome < 0, success_prices.shape[-1], outcome).to(torch.int64)
    return torch.gather(prices, -1, index.unsqueeze(-1)).squeeze(-1)


def balance_objective_terms(
    success_prices: torch.Tensor,
    failure_price: torch.Tensor,
    outcome: torch.Tensor,
    enabled: torch.Tensor,
) -> BalanceObjectiveTerms:
    enabled = enabled.expand_as(failure_price)
    selected = terminal_balance_cost(success_prices, failure_price, outcome)
    success_squared = success_prices.square()
    failure_squared = failure_price.square()
    return BalanceObjectiveTerms(
        observed_price=(selected * enabled).sum(-1),
        squared_prices=((success_squared.sum(-1) + failure_squared) * enabled).sum(-1),
        fourth_power_prices=(
            (success_squared.square().sum(-1) + failure_squared.square()) * enabled
        ).sum(-1),
        group_count=enabled.sum(-1),
    )


def balance_family_loss(
    terms: list[BalanceObjectiveTerms],
    beta: float,
    price_scale: float,
    record_weight: torch.Tensor,
) -> torch.Tensor:
    # The target's expected price is zero: successful prices are target-centered,
    # and failure has target probability zero. Every enabled group contributes,
    # so failures cannot change the denominator of the observed-price term.
    observed = torch.stack([term.observed_price for term in terms]).sum(0)
    squared_prices = torch.stack([term.squared_prices for term in terms]).sum(0)
    fourth_power_prices = torch.stack([term.fourth_power_prices for term in terms]).sum(0)
    # beta * (price^2 / 2 + price^4 / (4 * c^2)), for every priced outcome.
    regularizer = beta * (0.5 * squared_prices + fourth_power_prices / (4 * price_scale**2))
    count = torch.stack([term.group_count for term in terms]).sum(0).clamp_min(1)
    per_record = (regularizer - observed) / count
    return (per_record * record_weight).sum() / record_weight.sum().clamp_min(1.0)


def compute_balance_loss(
    preds: BalancePredictions,
    door_matches: DoorMatches,
    toilet_crossed_room_idx: torch.Tensor,
    room_area: torch.Tensor,
    area_order: torch.Tensor,
    area_probability: torch.Tensor,
    area_dual_mask: torch.Tensor,
    record_weight: torch.Tensor,
    door_beta: float,
    toilet_beta: float,
    area_beta: float,
    order_beta: float,
    door_price_scale: float,
    toilet_price_scale: float,
    area_price_scale: float,
    order_price_scale: float,
) -> torch.Tensor:
    tables = compute_balance_price_tables(preds, area_probability, area_dual_mask)
    failures = tables.door_failure.split(
        [tables.left.shape[1], tables.right.shape[1], tables.up.shape[1], tables.down.shape[1]],
        dim=-1,
    )
    door_terms = []
    for prices, failure, targets, compatibility in zip(
        (tables.left, tables.right, tables.up, tables.down),
        failures,
        (door_matches.left, door_matches.right, door_matches.up, door_matches.down),
        (preds.left_compatibility, preds.right_compatibility,
         preds.up_compatibility, preds.down_compatibility),
        strict=True,
    ):
        mask = targets >= 0
        if mask.any():
            if torch.any(targets[mask] >= prices.shape[-1]):
                raise ValueError("door balance target is out of range")
            source = torch.arange(targets.shape[-1], device=targets.device).expand_as(targets)
            if not compatibility[source[mask], targets[mask]].all():
                raise ValueError("observed door pairing is incompatible")
        door_terms.append(balance_objective_terms(
            prices, failure, targets, compatibility.any(-1),
        ))
    toilet_mask = toilet_crossed_room_idx >= 0
    if torch.any(toilet_crossed_room_idx[toilet_mask] >= preds.toilet_compatibility.numel()):
        raise ValueError("observed Toilet crossing room is out of range")
    if not preds.toilet_compatibility[toilet_crossed_room_idx[toilet_mask]].all():
        raise ValueError("observed Toilet crossing room is infeasible")
    toilet_terms = balance_objective_terms(
        tables.toilet_crossed_room.unsqueeze(1), tables.toilet_failure.unsqueeze(1),
        toilet_crossed_room_idx.unsqueeze(1), preds.toilet_compatibility.any().reshape(1, 1),
    )
    placed = (room_area >= 0) & area_dual_mask
    if torch.any(room_area[placed] >= AREA_COUNT):
        raise ValueError("observed room-area assignment is out of range")
    # Zero-probability areas can still occur in rejected-candidate fallbacks or
    # older replay data. Keep their observed prices and regularization so the
    # controller can discourage these outcomes without assigning them a target.
    area_terms = balance_objective_terms(
        tables.room_area, tables.room_area_failure, room_area, area_dual_mask,
    )
    return (
        balance_family_loss(door_terms, door_beta, door_price_scale, record_weight)
        + balance_family_loss([toilet_terms], toilet_beta, toilet_price_scale, record_weight)
        + balance_family_loss([area_terms], area_beta, area_price_scale, record_weight)
        + balance_family_loss([balance_objective_terms(
            tables.area_order, tables.area_order_failure, area_order,
            torch.ones_like(area_order, dtype=torch.bool),
        )], order_beta, order_price_scale, record_weight)
    )


def direction_balance_price_table(
    prices: torch.Tensor,
    source_door_variant_idx: torch.Tensor,
    target_door_variant_idx: torch.Tensor,
    compatibility: torch.Tensor,
    probability: torch.Tensor,
) -> torch.Tensor:
    concrete_prices = materialize_direction_balance_logits(
        prices,
        source_door_variant_idx,
        target_door_variant_idx,
    ).to(torch.float32)
    means = torch.sum(
        concrete_prices * probability.unsqueeze(0),
        dim=-1,
    )
    centered = concrete_prices - means.unsqueeze(-1)
    return torch.where(
        compatibility.unsqueeze(0),
        centered,
        0.0,
    )


def center_area_balance_prices(
    prices: torch.Tensor,
    area_probability: torch.Tensor,
    area_dual_mask: torch.Tensor,
) -> torch.Tensor:
    mean = torch.sum(prices * area_probability, dim=-1, keepdim=True)
    return torch.where(
        area_dual_mask.unsqueeze(-1),
        prices - mean,
        0.0,
    )


def compute_balance_price_tables(
    preds: BalancePredictions,
    area_probability: torch.Tensor,
    area_dual_mask: torch.Tensor,
) -> BalancePriceTables:
    direction_inputs = (
        (
            preds.left,
            preds.left_door_variant_idx,
            preds.right_door_variant_idx,
            preds.left_compatibility,
            preds.left_probability,
        ),
        (
            preds.right,
            preds.right_door_variant_idx,
            preds.left_door_variant_idx,
            preds.right_compatibility,
            preds.right_probability,
        ),
        (
            preds.up,
            preds.up_door_variant_idx,
            preds.down_door_variant_idx,
            preds.up_compatibility,
            preds.up_probability,
        ),
        (
            preds.down,
            preds.down_door_variant_idx,
            preds.up_door_variant_idx,
            preds.down_compatibility,
            preds.down_probability,
        ),
    )
    left, right, up, down = (
        direction_balance_price_table(prices, source_idx, target_idx, compatibility, probability)
        for prices, source_idx, target_idx, compatibility, probability in direction_inputs
    )
    toilet_mask = preds.toilet_compatibility.unsqueeze(0)
    toilet_count = toilet_mask.sum(dim=-1).clamp_min(1)
    toilet_mean = torch.sum(preds.toilet_crossed_room * toilet_mask, dim=-1) / toilet_count
    toilet = torch.where(
        toilet_mask,
        preds.toilet_crossed_room - toilet_mean.unsqueeze(-1),
        0.0,
    )
    toilet_failure = torch.where(
        toilet_mask.any(dim=-1),
        preds.toilet_failure,
        0.0,
    )
    if area_probability.shape != preds.room_area.shape:
        raise ValueError("area_probability shape must match balance room-area prices")
    if area_dual_mask.shape != preds.room_area.shape[:2]:
        raise ValueError("area_dual_mask shape must match balance room rows")
    room_area = center_area_balance_prices(
        preds.room_area.to(torch.float32),
        area_probability,
        area_dual_mask,
    )
    return BalancePriceTables(
        left=left,
        right=right,
        up=up,
        down=down,
        door_failure=torch.where(
            torch.cat([getattr(preds, name + "_compatibility").any(-1)
                       for name in ("left", "right", "up", "down")]),
            preds.door_failure, 0.0,
        ),
        toilet_crossed_room=toilet,
        toilet_failure=toilet_failure,
        room_area=room_area,
        room_area_failure=torch.where(area_dual_mask, preds.room_area_failure, 0.0),
        # Uniform target over the six areas at every start rank.
        area_order=preds.area_order - preds.area_order.mean(-1, keepdim=True),
        area_order_failure=preds.area_order_failure,
    )


def compute_room_area_balance_score_targets(
    tables: BalancePriceTables,
    room_area: torch.Tensor,
) -> torch.Tensor:
    return terminal_balance_cost(tables.room_area, tables.room_area_failure, room_area).detach()


def compute_balance_score_targets(
    tables: BalancePriceTables,
    door_matches: DoorMatches,
) -> torch.Tensor:
    prices = (tables.left, tables.right, tables.up, tables.down)
    failures = tables.door_failure.split([table.shape[1] for table in prices], dim=-1)
    return torch.cat([
        terminal_balance_cost(table, failure, targets)
        for table, failure, targets in zip(
            prices, failures,
            (door_matches.left, door_matches.right, door_matches.up, door_matches.down),
            strict=True,
        )
    ], dim=-1).detach()


def compute_step_balance_score_targets(
    tables: BalancePriceTables,
    door_match: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Known matches and known failures override the unconditional prediction."""
    prices = (tables.left, tables.right, tables.up, tables.down)
    sizes = [table.shape[1] for table in prices]
    failures = tables.door_failure.split(sizes, dim=-1)
    matches = door_match.split(sizes, dim=-1)
    values = []
    for table, failure, match in zip(prices, failures, matches, strict=True):
        # StepOutcomes uses the opposite-door count as the known-failure sentinel.
        terminal = torch.where(match == table.shape[-1], -1, match)
        values.append(terminal_balance_cost(table, failure, terminal))
    return torch.cat(values, dim=-1).detach(), door_match >= 0


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
    for (
        forward,
        reverse,
        pairs,
        source_variants,
        target_variants,
        source_global,
        target_global,
    ) in (
        (
            tables.left,
            tables.right,
            preds.horizontal_proposal_door_pairs,
            preds.left_door_variant_idx,
            preds.right_door_variant_idx,
            preds.left_global_door_variant_idx,
            preds.right_global_door_variant_idx,
        ),
        (
            tables.up,
            tables.down,
            preds.vertical_proposal_door_pairs,
            preds.up_door_variant_idx,
            preds.down_door_variant_idx,
            preds.up_global_door_variant_idx,
            preds.down_global_door_variant_idx,
        ),
    ):
        source_idx, target_idx = pairs.unbind(0)
        source_variant = source_global[source_variants[source_idx]]
        target_variant = target_global[target_variants[target_idx]]
        # A placement fixes both directed sides of the same compatible door pair.
        prices = forward[:, source_idx, target_idx] + reverse[:, target_idx, source_idx]
        proposal_score_table[:, source_variant, target_variant] = prices
        proposal_score_table[:, target_variant, source_variant] = prices
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


def compute_toilet_balance_score_targets(
    tables: BalancePriceTables,
    toilet_crossed_room_idx: torch.Tensor,
) -> torch.Tensor:
    return terminal_balance_cost(
        tables.toilet_crossed_room, tables.toilet_failure, toilet_crossed_room_idx,
    ).detach()
