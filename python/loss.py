import torch

from model import Predictions
from env import Outcomes
from dataclasses import dataclass


@dataclass
class LossConfig:
    door_weight: float
    connection_weight: float


def masked_binary_cross_entropy_loss(preds: torch.Tensor, outcomes: torch.Tensor, mask: torch.Tensor, weight: float) -> torch.Tensor:
    mask = (mask & (outcomes >= 0)).to(preds.dtype)
    binary_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        preds, outcomes.to(preds.dtype), reduction='none')
    return weight * torch.sum(binary_loss * mask), weight * torch.sum(mask)


def compute_loss(preds: Predictions, outcomes: Outcomes, mask: torch.Tensor, config: LossConfig) -> torch.Tensor:
    door_loss, door_wt = masked_binary_cross_entropy_loss(
        preds.door_invalid, outcomes.door_invalid, mask, config.door_weight)
    conn_loss, conn_wt = masked_binary_cross_entropy_loss(
        preds.connection_invalid, outcomes.connection_invalid, mask, config.connection_weight)
    mean_loss = (door_loss + conn_loss) / (door_wt + conn_wt + 1e-15)
    return mean_loss