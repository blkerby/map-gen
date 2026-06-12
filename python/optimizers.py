import math
from typing import Iterable

import torch
from torch.optim import Optimizer


def linear_warmup_value(step: int, target: float, warmup_steps: int) -> float:
    if warmup_steps == 0:
        return target
    return target * min(step / warmup_steps, 1.0)


def beta3_warmup_value(step: int, start: float, target: float, warmup_steps: int) -> float:
    if warmup_steps == 0:
        return target
    progress = min(step / warmup_steps, 1.0)
    start_half_life = math.log(0.5) / math.log(start)
    target_half_life = math.log(0.5) / math.log(target)
    half_life = start_half_life + progress * (target_half_life - start_half_life)
    return math.exp(math.log(0.5) / half_life)


class AdEMAMix(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float,
        beta1: float,
        beta2: float,
        beta3: float,
        alpha: float,
        beta3_warmup_steps: int,
        alpha_warmup_steps: int,
        eps: float,
        weight_decay: float,
    ):
        defaults = {
            "lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "beta3": beta3,
            "alpha": alpha,
            "beta3_warmup_steps": beta3_warmup_steps,
            "alpha_warmup_steps": alpha_warmup_steps,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            beta3_target = group["beta3"]
            alpha_target = group["alpha"]
            beta3_warmup_steps = group["beta3_warmup_steps"]
            alpha_warmup_steps = group["alpha_warmup_steps"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("AdEMAMix does not support sparse gradients")

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["m1"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    state["m2"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    state["nu"] = torch.zeros_like(param, memory_format=torch.preserve_format)

                state["step"] += 1
                step = state["step"]
                m1 = state["m1"]
                m2 = state["m2"]
                nu = state["nu"]
                grad = param.grad

                alpha = linear_warmup_value(step, alpha_target, alpha_warmup_steps)
                beta3 = beta3_warmup_value(step, beta1, beta3_target, beta3_warmup_steps)

                m1.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                m2.mul_(beta3).add_(grad, alpha=1.0 - beta3)
                nu.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                denom = nu.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
                update = m1.div(bias_correction1).add(m2, alpha=alpha).div_(denom)
                if weight_decay != 0.0:
                    update.add_(param, alpha=weight_decay)
                param.add_(update, alpha=-lr)
