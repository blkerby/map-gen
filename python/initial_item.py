"""Completed-map initial-item reachability and sampled start/skill conditions."""

import json
from pathlib import Path

import map_gen
import torch

DATA_TEXT = (Path(__file__).resolve().parents[1] / "data/initial-item-graph.json").read_text()
DATA = json.loads(DATA_TEXT)
NODES = DATA["nodes"]
GRAPHS = DATA["graphs"]
DIFFICULTIES = list(dict.fromkeys(graph["difficulty"] for graph in GRAPHS))
NODE_COUNT = len(NODES)
GRAPH_COUNT = len(GRAPHS)
SHIP_START = next(
    index + 1 for index, node in enumerate(NODES) if (node["room_id"], node["node_id"]) == (8, 5)
)
# Stored starts are zero when unconstrained, otherwise their data-node index plus one.


def encode_generation_variables(values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """The last two stored columns are categorical start and graph indices."""
    starts = values[:, -2].long()
    graphs = values[:, -1].long()
    start_features = torch.nn.functional.one_hot(starts, NODE_COUNT + 1)
    graph_features = torch.nn.functional.one_hot(graphs, GRAPH_COUNT) * (starts > 0).unsqueeze(1)
    return torch.cat(
        (values[:, :-2].to(dtype), start_features.to(dtype), graph_features.to(dtype)), dim=-1
    )


def create_initial_item_graph(rooms: list[dict]):
    return map_gen.InitialItemGraph(json.dumps(rooms), DATA_TEXT)


def sample_initial_item_conditions(
    rooms: list[dict],
    probabilities: list[float],
    wall_jump_probability: float,
    count: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    room_ids = {room["room_id"] for room in rooms if "room_id" in room}
    other_starts = [
        index + 1
        for index, node in enumerate(NODES)
        if node["start"] and node["room_id"] in room_ids and index + 1 != SHIP_START
    ]
    if probabilities[1] > 0 and 8 not in room_ids:
        raise ValueError("Ship start sampling requires Landing Site in the room set")
    if probabilities[2] > 0 and not other_starts:
        raise ValueError("other-start sampling requires extracted starts in the room set")
    choices = torch.multinomial(
        torch.tensor(probabilities, device=device), count, replacement=True
    )
    starts = torch.zeros(count, dtype=torch.int64, device=device)
    starts[choices == 1] = SHIP_START
    other_count = int((choices == 2).sum())
    if other_count:
        available = torch.tensor(other_starts, device=device)
        starts[choices == 2] = available[
            torch.randint(len(other_starts), (other_count,), device=device)
        ]
    difficulty = torch.randint(len(DIFFICULTIES), (count,), device=device)
    wall_jump = torch.rand(count, device=device) < wall_jump_probability
    graph_indices = torch.tensor(
        [
            [
                next(
                    i
                    for i, graph in enumerate(GRAPHS)
                    if graph["difficulty"] == name and graph["wall_jump"] == jump
                )
                for jump in (False, True)
            ]
            for name in DIFFICULTIES
        ],
        device=device,
    )
    return {
        "initial_item_start": starts.float(),
        "initial_item_graph": graph_indices[difficulty, wall_jump.long()].float(),
    }


def initial_item_metrics(
    invalid: torch.Tensor,
    variables: torch.Tensor,
    valid_maps: torch.Tensor,
    fields: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    starts = variables[:, fields.index("initial_item_start")].long()
    graphs = variables[:, fields.index("initial_item_graph")].long()
    constrained = starts > 0
    groups = {
        "all": constrained,
        "ship": starts == SHIP_START,
        "other": constrained & (starts != SHIP_START),
    }
    groups.update(
        {
            f"{graph['difficulty'].lower().replace(' ', '_')}_{'wall_jump' if graph['wall_jump'] else 'no_wall_jump'}": constrained
            & (graphs == index)
            for index, graph in enumerate(GRAPHS)
        }
    )
    metrics = {}
    for name, mask in groups.items():
        prefix = f"initial_item/{name}"
        metrics[f"{prefix}/count"] = mask.sum()
        if torch.any(mask):
            metrics[f"{prefix}/pass_fraction"] = (invalid[mask] == 0).float().mean()
        valid = mask & valid_maps
        metrics[f"{prefix}/valid_map_count"] = valid.sum()
        if torch.any(valid):
            metrics[f"{prefix}/valid_map_pass_fraction"] = (invalid[valid] == 0).float().mean()
    return metrics
