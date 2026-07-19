from dataclasses import dataclass

import torch

from env import DoorMatches, StepOutcomes
from model import BalancePredictions, Predictions

BALANCE_TARGET_LOG_ODDS_LIMIT = 20.0


@dataclass
class LossConfig:
    door_weight: float
    connection_weight: float
    toilet_weight: float
    phantoon_weight: float
    balance_weight: float
    toilet_balance_weight: float
    avg_frontiers_weight: float
    graph_diameter_weight: float
    save_distance_weight: float
    refill_distance_weight: float
    missing_connect_utility_weight: float
    area_crossing_weight: float
    area_size_weight: float
    area_map_station_weight: float
    distance_proximity_scale: float


@dataclass
class LossBreakdown:
    total: torch.Tensor
    door: torch.Tensor
    connection: torch.Tensor
    toilet: torch.Tensor
    phantoon: torch.Tensor
    balance: torch.Tensor
    toilet_balance: torch.Tensor
    avg_frontiers: torch.Tensor
    graph_diameter: torch.Tensor
    save_distance: torch.Tensor
    refill_distance: torch.Tensor
    missing_connect_utility: torch.Tensor
    area_crossings: torch.Tensor
    area_size: torch.Tensor
    area_map_station: torch.Tensor
    door_contribution: torch.Tensor
    connection_contribution: torch.Tensor
    toilet_contribution: torch.Tensor
    phantoon_contribution: torch.Tensor
    balance_contribution: torch.Tensor
    toilet_balance_contribution: torch.Tensor
    avg_frontiers_contribution: torch.Tensor
    graph_diameter_contribution: torch.Tensor
    save_distance_contribution: torch.Tensor
    refill_distance_contribution: torch.Tensor
    missing_connect_utility_contribution: torch.Tensor
    area_crossings_contribution: torch.Tensor
    area_size_contribution: torch.Tensor
    area_map_station_contribution: torch.Tensor


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
    balance_score_target_logits: torch.Tensor,
    balance_score_mask: torch.Tensor,
    toilet_balance_score_target_logits: torch.Tensor,
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
    area_mask: torch.Tensor,
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
    phantoon_loss, phantoon_wt = masked_binary_cross_entropy_loss(
        preds.phantoon_invalid,
        outcomes.phantoon_invalid,
        mask.squeeze(-1),
        config.phantoon_weight,
    )
    balance_loss, balance_wt = masked_bernoulli_kl_loss(
        preds.balance_score,
        balance_score_target_logits,
        mask & balance_score_mask,
        config.balance_weight,
    )
    toilet_balance_loss, toilet_balance_wt = masked_bernoulli_kl_loss(
        preds.toilet_balance_score,
        toilet_balance_score_target_logits,
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
    total_weight = (
        door_wt
        + conn_wt
        + toilet_wt
        + phantoon_wt
        + balance_wt
        + toilet_balance_wt
        + avg_frontiers_wt
        + graph_diameter_wt
        + save_distance_wt
        + refill_distance_wt
        + missing_connect_utility_wt
        + area_crossings_wt
        + area_size_wt
        + area_map_station_wt
        + 1e-15
    )
    door_contribution = door_loss / total_weight
    connection_contribution = conn_loss / total_weight
    toilet_contribution = toilet_loss / total_weight
    phantoon_contribution = phantoon_loss / total_weight
    balance_contribution = balance_loss / total_weight
    toilet_balance_contribution = toilet_balance_loss / total_weight
    avg_frontiers_contribution = avg_frontiers_loss / total_weight
    graph_diameter_contribution = graph_diameter_loss / total_weight
    save_distance_contribution = save_distance_loss / total_weight
    refill_distance_contribution = refill_distance_loss / total_weight
    missing_connect_utility_contribution = missing_connect_utility_loss / total_weight
    area_crossings_contribution = area_crossings_loss / total_weight
    area_size_contribution = area_size_loss / total_weight
    area_map_station_contribution = area_map_station_loss / total_weight
    mean_loss = (
        door_contribution
        + connection_contribution
        + toilet_contribution
        + phantoon_contribution
        + balance_contribution
        + toilet_balance_contribution
        + avg_frontiers_contribution
        + graph_diameter_contribution
        + save_distance_contribution
        + refill_distance_contribution
        + missing_connect_utility_contribution
        + area_crossings_contribution
        + area_size_contribution
        + area_map_station_contribution
    )
    return LossBreakdown(
        total=mean_loss,
        door=door_loss / (door_wt + 1e-15),
        connection=conn_loss / (conn_wt + 1e-15),
        toilet=toilet_loss / (toilet_wt + 1e-15),
        phantoon=phantoon_loss / (phantoon_wt + 1e-15),
        balance=balance_loss / (balance_wt + 1e-15),
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
        door_contribution=door_contribution,
        connection_contribution=connection_contribution,
        toilet_contribution=toilet_contribution,
        phantoon_contribution=phantoon_contribution,
        balance_contribution=balance_contribution,
        toilet_balance_contribution=toilet_balance_contribution,
        avg_frontiers_contribution=avg_frontiers_contribution,
        graph_diameter_contribution=graph_diameter_contribution,
        save_distance_contribution=save_distance_contribution,
        refill_distance_contribution=refill_distance_contribution,
        missing_connect_utility_contribution=missing_connect_utility_contribution,
        area_crossings_contribution=area_crossings_contribution,
        area_size_contribution=area_size_contribution,
        area_map_station_contribution=area_map_station_contribution,
    )


def categorical_balance_loss(
    logits: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = targets >= 0
    if not torch.any(mask):
        return torch.sum(logits) * 0.0, logits.new_tensor(0.0)
    return (
        torch.nn.functional.cross_entropy(logits[mask], targets[mask], reduction="sum"),
        torch.sum(mask).to(logits.dtype),
    )


def direction_variant_balance_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    source_door_variant_idx: torch.Tensor,
    target_door_variant_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = targets >= 0
    if not torch.any(mask):
        return torch.sum(logits) * 0.0, logits.new_tensor(0.0)
    safe_targets = targets.clamp(0, target_door_variant_idx.numel() - 1).to(torch.int64)
    variant_targets = target_door_variant_idx[safe_targets]
    concrete_source_logits = logits[:, source_door_variant_idx, :]
    return (
        torch.nn.functional.cross_entropy(
            concrete_source_logits[mask],
            variant_targets[mask],
            reduction="sum",
        ),
        torch.sum(mask).to(logits.dtype),
    )


def compute_balance_loss(
    preds: BalancePredictions,
    door_matches: DoorMatches,
    toilet_crossed_room_idx: torch.Tensor,
) -> torch.Tensor:
    left_loss, left_weight = direction_variant_balance_loss(
        preds.left,
        door_matches.left,
        preds.left_door_variant_idx,
        preds.right_door_variant_idx,
    )
    right_loss, right_weight = direction_variant_balance_loss(
        preds.right,
        door_matches.right,
        preds.right_door_variant_idx,
        preds.left_door_variant_idx,
    )
    up_loss, up_weight = direction_variant_balance_loss(
        preds.up,
        door_matches.up,
        preds.up_door_variant_idx,
        preds.down_door_variant_idx,
    )
    down_loss, down_weight = direction_variant_balance_loss(
        preds.down,
        door_matches.down,
        preds.down_door_variant_idx,
        preds.up_door_variant_idx,
    )
    toilet_loss, toilet_weight = categorical_balance_loss(
        preds.toilet_crossed_room,
        toilet_crossed_room_idx,
    )
    total_loss = left_loss + right_loss + up_loss + down_loss + toilet_loss
    total_weight = left_weight + right_weight + up_weight + down_weight + toilet_weight
    return total_loss / (total_weight + 1e-15)


def expand_direction_balance_probabilities(
    logits: torch.Tensor,
    source_door_variant_idx: torch.Tensor,
    target_door_variant_idx: torch.Tensor,
) -> torch.Tensor:
    variant_probabilities = torch.softmax(logits, dim=-1)
    concrete_probabilities = variant_probabilities[:, source_door_variant_idx, :][
        :, :, target_door_variant_idx
    ]
    target_variant_count = torch.bincount(
        target_door_variant_idx,
        minlength=logits.shape[-1],
    ).to(concrete_probabilities.dtype)
    concrete_target_count = target_variant_count[target_door_variant_idx]
    return concrete_probabilities / concrete_target_count.view(1, 1, -1)


def compute_balance_door_match_ss(preds: BalancePredictions) -> torch.Tensor:
    return (
        torch.sum(
            expand_direction_balance_probabilities(
                preds.left,
                preds.left_door_variant_idx,
                preds.right_door_variant_idx,
            ).square()
        )
        + torch.sum(
            expand_direction_balance_probabilities(
                preds.right,
                preds.right_door_variant_idx,
                preds.left_door_variant_idx,
            ).square()
        )
        + torch.sum(
            expand_direction_balance_probabilities(
                preds.up,
                preds.up_door_variant_idx,
                preds.down_door_variant_idx,
            ).square()
        )
        + torch.sum(
            expand_direction_balance_probabilities(
                preds.down,
                preds.down_door_variant_idx,
                preds.up_door_variant_idx,
            ).square()
        )
    )


def compute_balance_toilet_crossed_room_ss(preds: BalancePredictions) -> torch.Tensor:
    return torch.sum(torch.softmax(preds.toilet_crossed_room, dim=-1).square())


def categorical_balance_score_target_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = targets >= 0
    if logits.shape[-1] == 0:
        return logits.new_empty(targets.shape, dtype=torch.float32), mask
    safe_targets = torch.clamp(targets, min=0).to(torch.int64)
    target_logits = torch.gather(
        direction_balance_score_logit_table(logits),
        -1,
        safe_targets.unsqueeze(-1),
    ).squeeze(-1)
    return target_logits.detach(), mask


def direction_variant_balance_score_target_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    source_door_variant_idx: torch.Tensor,
    target_door_variant_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = targets >= 0
    if logits.shape[-1] == 0:
        return logits.new_empty(targets.shape, dtype=torch.float32), mask
    safe_targets = targets.clamp(0, target_door_variant_idx.numel() - 1).to(torch.int64)
    variant_targets = target_door_variant_idx[safe_targets]
    concrete_source_logit_table = direction_balance_score_logit_table(logits)[
        :, source_door_variant_idx, :
    ]
    target_logits = torch.gather(
        concrete_source_logit_table,
        -1,
        variant_targets.unsqueeze(-1),
    ).squeeze(-1)
    return target_logits.detach(), mask


def direction_valid_match_variant_balance_score_target_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    source_door_variant_idx: torch.Tensor,
    target_door_variant_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = (targets >= 0) & (targets < target_door_variant_idx.numel())
    if logits.shape[-1] == 0:
        return logits.new_empty(targets.shape, dtype=torch.float32), mask
    safe_targets = targets.clamp(0, target_door_variant_idx.numel() - 1).to(torch.int64)
    variant_targets = target_door_variant_idx[safe_targets]
    logit_table = direction_balance_score_logit_table(logits)[:, source_door_variant_idx, :]
    while logit_table.ndim < safe_targets.ndim + 1:
        logit_table = logit_table.unsqueeze(1)
    logit_table = logit_table.expand(*safe_targets.shape, logit_table.shape[-1])
    target_logits = torch.gather(
        logit_table,
        -1,
        variant_targets.unsqueeze(-1),
    ).squeeze(-1)
    return target_logits.detach(), mask


def direction_balance_score_logit_table(logits: torch.Tensor) -> torch.Tensor:
    logits = logits.to(torch.float32)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    non_target_log_probs = torch.log(-torch.expm1(log_probs))
    return torch.clamp(
        log_probs - non_target_log_probs,
        min=-BALANCE_TARGET_LOG_ODDS_LIMIT,
        max=BALANCE_TARGET_LOG_ODDS_LIMIT,
    )


def compute_balance_score_target_logits(
    preds: BalancePredictions,
    door_matches: DoorMatches,
) -> tuple[torch.Tensor, torch.Tensor]:
    left_values, left_mask = direction_variant_balance_score_target_logits(
        preds.left,
        door_matches.left,
        preds.left_door_variant_idx,
        preds.right_door_variant_idx,
    )
    right_values, right_mask = direction_variant_balance_score_target_logits(
        preds.right,
        door_matches.right,
        preds.right_door_variant_idx,
        preds.left_door_variant_idx,
    )
    up_values, up_mask = direction_variant_balance_score_target_logits(
        preds.up,
        door_matches.up,
        preds.up_door_variant_idx,
        preds.down_door_variant_idx,
    )
    down_values, down_mask = direction_variant_balance_score_target_logits(
        preds.down,
        door_matches.down,
        preds.down_door_variant_idx,
        preds.up_door_variant_idx,
    )
    return (
        torch.cat([left_values, right_values, up_values, down_values], dim=-1),
        torch.cat([left_mask, right_mask, up_mask, down_mask], dim=-1),
    )


def compute_step_balance_score_target_logits(
    preds: BalancePredictions,
    door_match: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    left, right, up, down = torch.split(
        door_match,
        [
            preds.left_door_variant_idx.numel(),
            preds.right_door_variant_idx.numel(),
            preds.up_door_variant_idx.numel(),
            preds.down_door_variant_idx.numel(),
        ],
        dim=-1,
    )
    left_values, left_mask = direction_valid_match_variant_balance_score_target_logits(
        preds.left,
        left,
        preds.left_door_variant_idx,
        preds.right_door_variant_idx,
    )
    right_values, right_mask = direction_valid_match_variant_balance_score_target_logits(
        preds.right,
        right,
        preds.right_door_variant_idx,
        preds.left_door_variant_idx,
    )
    up_values, up_mask = direction_valid_match_variant_balance_score_target_logits(
        preds.up,
        up,
        preds.up_door_variant_idx,
        preds.down_door_variant_idx,
    )
    down_values, down_mask = direction_valid_match_variant_balance_score_target_logits(
        preds.down,
        down,
        preds.down_door_variant_idx,
        preds.up_door_variant_idx,
    )
    return (
        torch.cat([left_values, right_values, up_values, down_values], dim=-1),
        torch.cat([left_mask, right_mask, up_mask, down_mask], dim=-1),
    )


def compute_toilet_balance_score_target_logits(
    preds: BalancePredictions,
    toilet_crossed_room_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return categorical_balance_score_target_logits(
        preds.toilet_crossed_room,
        toilet_crossed_room_idx,
    )
