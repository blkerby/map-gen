"""Concrete-room placement-step outcomes and balance costs.

Step indices are zero-based internally, including the random initial placement.
The uniform target includes that placement; its room identity is sampled
separately from candidate scoring. A room never placed has outcome -1,
independent of episode length.
"""

import torch


def episode_room_steps(room_idx: torch.Tensor, num_rooms: int) -> torch.Tensor:
    valid = (room_idx >= 0) & (room_idx < num_rooms)
    length = room_idx.shape[1]
    steps = torch.arange(length, device=room_idx.device).expand_as(room_idx)
    first = torch.full((room_idx.shape[0], num_rooms + 1), length, device=room_idx.device)
    first.scatter_reduce_(
        1, torch.where(valid, room_idx, num_rooms).long(),
        torch.where(valid, steps, length), reduce="amin", include_self=True,
    )
    first = first[:, :num_rooms]
    return torch.where(first < length, first, -1)


def remaining_step_price(terminal_prices: torch.Tensor, room_placed: torch.Tensor) -> torch.Tensor:
    """Cost of still-unplaced rooms, including their eventual failure costs."""
    return torch.where(room_placed.bool(), 0.0, terminal_prices).sum(-1)


def candidate_step_balance_score(
    predicted_future: torch.Tensor,
    prices: torch.Tensor,
    failure_prices: torch.Tensor,
    post_room_placed: torch.Tensor,
    candidate_room_idx: torch.Tensor,
    step: int,
    episode_length: int,
) -> torch.Tensor:
    """Exact current cost plus predicted remaining cost; exact failures at the horizon."""
    num_rooms = prices.shape[1]
    valid = (candidate_room_idx >= 0) & (candidate_room_idx < num_rooms)
    immediate = prices[:, :, step].gather(-1, candidate_room_idx.long().clamp(0, num_rooms - 1))
    immediate = torch.where(valid, immediate, 0.0)
    if step + 1 == episode_length:
        future = remaining_step_price(failure_prices.unsqueeze(1), post_room_placed)
    else:
        future = torch.where(post_room_placed.bool().all(-1), 0.0, predicted_future)
    return immediate + future


def proposal_step_prices(
    prices: torch.Tensor,
    room_placed: torch.Tensor,
    room_connection_variant_idx: torch.Tensor,
    num_room_connection_variants: int,
    door_variant_connection_variant_idx: torch.Tensor,
    step: int,
) -> torch.Tensor:
    """Mean immediate cost of unused concrete rooms for each proposed door variant.

The proposal does not choose a concrete room. Resolution samples an available
representative; candidate evaluation then uses that room's exact price.
"""
    available = (~room_placed.bool()).to(prices.dtype)
    indices = room_connection_variant_idx.unsqueeze(0).expand_as(available)
    shape = (prices.shape[0], num_room_connection_variants)
    sums = prices.new_zeros(shape).scatter_add_(1, indices, prices[:, :, step] * available)
    counts = prices.new_zeros(shape).scatter_add_(1, indices, available)
    means = sums / counts.clamp_min(1)
    return means[:, door_variant_connection_variant_idx]
