import numpy as np
import torch


def sinkhorn_door_target(compatibility: torch.Tensor) -> torch.Tensor:
    """Maximum-entropy complete-matching marginals on concrete compatible doors."""
    if compatibility.ndim != 2 or compatibility.shape[0] != compatibility.shape[1]:
        raise ValueError("complete door matching requires a square compatibility matrix")
    if compatibility.dtype != torch.bool:
        raise ValueError("door compatibility must be boolean")
    if compatibility.numel() == 0:
        return compatibility.to(torch.float32)
    probability = compatibility.detach().cpu().numpy().astype(np.float64)
    if np.any(probability.sum(axis=1) == 0) or np.any(probability.sum(axis=0) == 0):
        raise ValueError("door compatibility contains an empty row or column")
    for _ in range(10000):
        probability /= probability.sum(axis=1, keepdims=True)
        probability /= probability.sum(axis=0, keepdims=True)
        if np.max(np.abs(probability.sum(axis=1) - 1.0)) <= 1e-10:
            return torch.tensor(probability, dtype=torch.float32, device=compatibility.device)
    raise RuntimeError("door target Sinkhorn iteration did not converge within 10000 iterations")
