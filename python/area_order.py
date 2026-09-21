"""Area-start order outcomes and exact prices for starting a new area."""

import torch

from env import AREA_COUNT, Actions


def episode_area_order(actions: Actions, num_rooms: int) -> torch.Tensor:
    """Return the area at each start rank; -1 means that rank was never reached."""
    areas = actions.room_area.to(torch.int64)
    length = areas.shape[1]
    valid = (actions.room_idx < num_rooms) & (areas < AREA_COUNT)
    steps = torch.arange(length, device=areas.device).expand_as(areas)
    first = torch.full((areas.shape[0], AREA_COUNT), length, device=areas.device)
    first.scatter_reduce_(
        1,
        areas.clamp(0, AREA_COUNT - 1),
        torch.where(valid, steps, length),
        reduce="amin",
        include_self=True,
    )
    first_steps, order = first.sort(dim=1, stable=True)
    return torch.where(first_steps < length, order, -1)


def remaining_order_price(rank_prices: torch.Tensor, area_used: torch.Tensor) -> torch.Tensor:
    """Sum terminal rank costs for ranks still unfilled in this state."""
    count = area_used.to(torch.int64).sum(-1)
    ranks = torch.arange(AREA_COUNT, device=count.device)
    return torch.where(ranks >= count.unsqueeze(-1), rank_prices, 0.0).sum(-1)


def candidate_order_price(
    prices: torch.Tensor,
    post_area_size: torch.Tensor,
    candidates: Actions,
    room_tile_count: torch.Tensor,
) -> torch.Tensor:
    """Exact immediate cost, using the area sizes after each candidate placement."""
    post_area_size = post_area_size.to(torch.int64)
    valid = (candidates.room_idx < room_tile_count.numel()) & (candidates.room_area < AREA_COUNT)
    room = candidates.room_idx.long().clamp_max(room_tile_count.numel() - 1)
    area = candidates.room_area.long().clamp_max(AREA_COUNT - 1)
    selected_size = post_area_size.gather(-1, area.unsqueeze(-1)).squeeze(-1)
    starts_area = valid & (selected_size == room_tile_count[room])
    rank = ((post_area_size > 0).sum(-1) - 1).clamp(0, AREA_COUNT - 1)
    batch = torch.arange(prices.shape[0], device=prices.device).unsqueeze(1)
    return torch.where(starts_area, prices[batch, rank, area], 0.0)


def proposal_order_price(prices: torch.Tensor, area_used: torch.Tensor) -> torch.Tensor:
    """Immediate costs of each proposed area in the current state."""
    used = area_used.to(torch.bool)
    rank = used.sum(-1).clamp_max(AREA_COUNT - 1)
    batch = torch.arange(prices.shape[0], device=prices.device)
    return torch.where(used, 0.0, prices[batch, rank])


def candidate_order_balance_score(
    predicted_future: torch.Tensor,
    prices: torch.Tensor,
    failure_prices: torch.Tensor,
    post_area_size: torch.Tensor,
    post_room_placed: torch.Tensor,
    candidates: Actions,
    room_tile_count: torch.Tensor,
) -> torch.Tensor:
    post_area_size = post_area_size.to(torch.int64)
    immediate = candidate_order_price(prices, post_area_size, candidates, room_tile_count)
    used = post_area_size > 0
    future = torch.where(used.sum(-1) < AREA_COUNT, predicted_future, 0.0)
    future = torch.where(
        post_room_placed.to(torch.bool).all(-1),
        remaining_order_price(failure_prices.unsqueeze(1), used),
        future,
    )
    return immediate + future
