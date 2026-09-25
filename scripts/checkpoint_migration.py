"""Strict helpers for adding model parameters without changing saved optimizer state."""

import json
import torch

from model_loading import without_prefix
from train import load_named_optimizer_checkpoint_state, named_checkpoint_optimizers


def add_model_parameters(tensors, model, prefix: str, added_prefix: str) -> set[str]:
    expected = model.state_dict()
    old = without_prefix(tensors, prefix)
    added = {name for name in expected if name.startswith(added_prefix)}
    if set(old) != set(expected) - added:
        raise ValueError(
            f"unexpected {prefix} source keys: missing={set(expected) - added - set(old)}, extra={set(old) - set(expected) | (set(old) & added)}"
        )
    for name in added:
        old[name] = expected[name].detach().cpu().contiguous().clone()
        tensors[f"{prefix}.{name}"] = old[name]
    model.load_state_dict(old, strict=True)
    return added


def extend_optimizer_groups(
    tensors, metadata, model, optimizer, prefix: str, added: set[str]
) -> None:
    """Keep every old state ID; insert fresh IDs in the new parameter ordering."""
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    parts = named_checkpoint_optimizers(optimizer)
    if set(json.loads(metadata[f"{prefix}_names"])) != set(parts):
        raise ValueError(f"unexpected optimizer parts for {prefix}")
    for part_name, part in parts.items():
        key = f"{prefix}_{part_name}_param_groups"
        saved_groups = json.loads(metadata[key])
        next_id = max((i for group in saved_groups for i in group["params"]), default=-1) + 1
        for saved, current in zip(saved_groups, part.param_groups, strict=True):
            old_parameters = [p for p in current["params"] if names[id(p)] not in added]
            if len(old_parameters) != len(saved["params"]):
                raise ValueError(f"unexpected parameter count for {prefix}.{part_name}")
            old_ids = iter(saved["params"])
            ids = []
            for parameter in current["params"]:
                if names[id(parameter)] in added:
                    ids.append(next_id)
                    next_id += 1
                else:
                    ids.append(next(old_ids))
            saved["params"] = ids
        metadata[key] = json.dumps(saved_groups)
    load_named_optimizer_checkpoint_state(optimizer, tensors, metadata, prefix)
    # Loading accepts differently shaped moments; check them explicitly.
    for part in parts.values():
        for parameter, state in part.state.items():
            for name, value in state.items():
                if torch.is_tensor(value) and value.ndim and value.shape != parameter.shape:
                    raise ValueError(
                        f"optimizer moment {name} has shape {value.shape}, expected {parameter.shape}"
                    )


