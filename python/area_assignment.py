import torch


def assign_room_areas(room_idx: torch.Tensor) -> torch.Tensor:
    return torch.randint(
        0,
        6,
        room_idx.shape,
        device=room_idx.device,
        dtype=torch.int64,
    )
