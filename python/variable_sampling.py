"""Sampling instantiated config variables for generation and replay."""

import math

import torch

from train_config import (
    VariableFloat,
    VariableMixture,
    VariableRange,
    VariableSchedule,
    variable_schedule_kind_and_value,
)


def variable_mixture_weights(value: VariableMixture, path: str) -> list[float]:
    weights = [float(component.weight) for component in value.mixture]
    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
        raise ValueError(f"{path}.mixture weights must be finite and nonnegative")
    if sum(weights) <= 0.0:
        raise ValueError(f"{path}.mixture must have a positive total weight")
    return weights


def sample_variable_float(
    value: VariableFloat, path: str, num_envs: int, device: torch.device,
) -> torch.Tensor:
    if isinstance(value, VariableMixture):
        weights = torch.tensor(
            variable_mixture_weights(value, path),
            dtype=torch.float32,
            device=device,
        )
        component_idx = torch.multinomial(weights, num_envs, replacement=True)
        component_values = torch.stack(
            [
                sample_variable_float(
                    component.value,
                    f"{path}.mixture[{index}].value",
                    num_envs,
                    device,
                )
                for index, component in enumerate(value.mixture)
            ],
            dim=1,
        )
        return torch.gather(component_values, 1, component_idx.unsqueeze(1)).squeeze(1)
    if isinstance(value, VariableSchedule):
        kind, range_value = variable_schedule_kind_and_value(value, path)
        if not isinstance(range_value, VariableRange):
            raise ValueError(f"{path} must be instantiated before sampling")
        min_value = float(range_value.min)
        max_value = float(range_value.max)
        if min_value > max_value:
            raise ValueError(f"{path}.min must be less than or equal to {path}.max")
        sample = torch.rand([num_envs], dtype=torch.float32, device=device)
        if kind != "log":
            return min_value + sample * (max_value - min_value)
        if min_value <= 0.0:
            raise ValueError(f"{path}.min must be greater than zero for a log range")
        log_min = math.log(min_value)
        log_max = math.log(max_value)
        return torch.exp(log_min + sample * (log_max - log_min))
    return torch.full([num_envs], float(value), dtype=torch.float32, device=device)
