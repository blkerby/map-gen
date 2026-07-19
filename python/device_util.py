from types import ModuleType

import torch

GPU_DEVICE_TYPES = ("cuda", "xpu")


def is_gpu(device: torch.device) -> bool:
    return device.type in GPU_DEVICE_TYPES


def backend_for_type(device_type: str) -> ModuleType:
    if device_type == "cuda":
        return torch.cuda
    if device_type == "xpu":
        return torch.xpu
    raise ValueError(f"{device_type} is not a supported GPU device type")


def gpu_backend(device: torch.device) -> ModuleType:
    return backend_for_type(device.type)


def available_gpu_type() -> str | None:
    for device_type in GPU_DEVICE_TYPES:
        if backend_for_type(device_type).is_available():
            return device_type
    return None
