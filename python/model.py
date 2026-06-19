from __future__ import annotations

import torch
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from env import OutputMetadata, Features

if TYPE_CHECKING:
    from train_config import FeatureConfig

NUM_COORD_VALUES = 256
COORD_OFFSET = 128
DETERMINISTIC_INVALID_LOGIT = 20.0


# These tensors are all f32 with shape
#    [batch, time, output]  during training,
#    [batch, candidate, output]  during generation
@dataclass
class Predictions:
    # log-odds of invalid door (unconnected):
    door_invalid: torch.Tensor
    # log-odds of invalid connection (lack of return path):
    connection_invalid: torch.Tensor
    # log-odds of invalid Toilet crossing count:
    toilet_invalid: torch.Tensor
    # Predicted balance-model log-odds for the matched target door:
    balance_score: torch.Tensor
    # Predicted balance-model log-odds for the room crossed by the Toilet:
    toilet_balance_score: torch.Tensor
    # Predicted average live frontier count across the full episode:
    avg_frontiers: torch.Tensor
    # Predicted graph diameter across placed room parts:
    graph_diameter: torch.Tensor
    # Predicted save-to-room proximity utility for each global room part:
    save_to_room_utility: torch.Tensor
    # Predicted room-to-save proximity utility for each global room part:
    save_from_room_utility: torch.Tensor
    # Predicted refill-to-room proximity utility for each global room part:
    refill_to_room_utility: torch.Tensor
    # Predicted room-to-refill proximity utility for each global room part:
    refill_from_room_utility: torch.Tensor
    # Predicted distance for each required missing connection:
    missing_connect_distance: torch.Tensor
    # Frontier-local proposal logits for door variants:
    proposal_score: torch.Tensor
    # Optional frontier-local state before global pooling:
    proposal_state: torch.Tensor
    proposal_row_snapshot_idx: torch.Tensor
    proposal_row_frontier_idx: torch.Tensor


@dataclass
class BalancePredictions:
    left: torch.Tensor
    right: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor
    toilet_crossed_room: torch.Tensor


def get_predictions(raw_preds, output_sizes):
    preds = []
    col = 0
    for size in output_sizes:
        preds.append(raw_preds[:, :, col : (col + size)])
        col += size

    return Predictions(
        door_invalid=preds[0],
        connection_invalid=preds[1],
        toilet_invalid=preds[2].squeeze(-1),
        balance_score=preds[3],
        toilet_balance_score=preds[4].squeeze(-1),
        avg_frontiers=raw_preds.new_empty([raw_preds.shape[0], raw_preds.shape[1]]),
        graph_diameter=raw_preds.new_empty([raw_preds.shape[0], raw_preds.shape[1]]),
        save_to_room_utility=raw_preds.new_empty([raw_preds.shape[0], raw_preds.shape[1], 0]),
        save_from_room_utility=raw_preds.new_empty([raw_preds.shape[0], raw_preds.shape[1], 0]),
        refill_to_room_utility=raw_preds.new_empty([raw_preds.shape[0], raw_preds.shape[1], 0]),
        refill_from_room_utility=raw_preds.new_empty([raw_preds.shape[0], raw_preds.shape[1], 0]),
        missing_connect_distance=raw_preds.new_empty([raw_preds.shape[0], raw_preds.shape[1], 0]),
        proposal_score=raw_preds.new_empty([raw_preds.shape[0], raw_preds.shape[1], 0]),
        proposal_state=raw_preds.new_empty([raw_preds.shape[0], raw_preds.shape[1], 0]),
        proposal_row_snapshot_idx=raw_preds.new_empty([0], dtype=torch.int64),
        proposal_row_frontier_idx=raw_preds.new_empty([0], dtype=torch.int16),
    )


def apply_known_invalid_logits(
    invalid_logits: torch.Tensor,
    known_invalid: torch.Tensor,
    outcome_name: str,
) -> torch.Tensor:
    if known_invalid.shape[-1] == 0:
        return invalid_logits
    torch._assert(
        known_invalid.shape[-1] == invalid_logits.shape[-1],
        f"known {outcome_name} outcomes must match {outcome_name} prediction width",
    )
    while known_invalid.ndim < invalid_logits.ndim:
        known_invalid = known_invalid.unsqueeze(1)
    deterministic_logits = torch.where(
        known_invalid == 0,
        -DETERMINISTIC_INVALID_LOGIT,
        DETERMINISTIC_INVALID_LOGIT,
    ).to(invalid_logits.dtype)
    return torch.where(known_invalid >= 0, deterministic_logits, invalid_logits)


def apply_frontier_door_invalid_logits(
    door_invalid: torch.Tensor,
    frontier_door_invalid: torch.Tensor,
    row_snapshot_idx: torch.Tensor,
    row_door_output_idx: torch.Tensor,
) -> torch.Tensor:
    if door_invalid.shape[-1] == 0 or frontier_door_invalid.shape[0] == 0:
        return door_invalid
    snapshot_count = door_invalid.shape[0]
    door_output_count = door_invalid.shape[-1]
    frontier_door_invalid = frontier_door_invalid.squeeze(-1).to(door_invalid.dtype)
    row_snapshot_idx = row_snapshot_idx.to(device=door_invalid.device, dtype=torch.int64)
    row_door_output_idx = row_door_output_idx.to(device=door_invalid.device, dtype=torch.int64)
    valid_rows = (
        (row_snapshot_idx >= 0)
        & (row_snapshot_idx < snapshot_count)
        & (row_door_output_idx >= 0)
        & (row_door_output_idx < door_output_count)
    )
    safe_row_snapshot_idx = row_snapshot_idx.clamp(0, snapshot_count - 1)
    safe_row_door_output_idx = row_door_output_idx.clamp(0, door_output_count - 1)
    row_lookup_idx = safe_row_snapshot_idx * door_output_count + safe_row_door_output_idx
    door_invalid_flat = door_invalid.flatten().clone()
    scatter_values = torch.where(
        valid_rows,
        frontier_door_invalid,
        door_invalid_flat.detach().gather(0, row_lookup_idx),
    )
    door_invalid_flat.scatter_(0, row_lookup_idx, scatter_values)
    return door_invalid_flat.view_as(door_invalid)


def normalize(x: torch.Tensor):
    return torch.nn.functional.rms_norm(x, (x.size(-1),))


def activation_dtype(device: torch.device, parameter_dtype: torch.dtype) -> torch.dtype:
    if device.type == "cuda" and torch.is_autocast_enabled("cuda"):
        return torch.get_autocast_dtype("cuda")
    return parameter_dtype


class FactorizedOutcomeHead(torch.nn.Module):
    def __init__(self, output_metadata, num_geometry_outcomes, embedding_width):
        super().__init__()
        self.embedding_width = embedding_width
        self.num_outputs = len(output_metadata)
        metadata = torch.tensor(output_metadata, dtype=torch.int64).reshape(self.num_outputs, 2)
        self.register_buffer("room_idx", metadata[:, 0])
        self.register_buffer("geometry_outcome_idx", metadata[:, 1])
        self.geometry_outcome_embedding = torch.nn.Parameter(
            torch.randn([num_geometry_outcomes, embedding_width]) / math.sqrt(embedding_width)
        )
        self.state = torch.nn.Linear(embedding_width, embedding_width, bias=False)
        self.logit_scale = torch.nn.Parameter(
            torch.tensor(math.log(math.sqrt(embedding_width) / 2))
        )

    def forward(self, X, room_x, room_y, room_placed, pos_embedding_x, pos_embedding_y):
        if self.num_outputs == 0:
            return X.new_empty([X.shape[0], X.shape[1], 0], dtype=torch.float32)
        state = self.state(X)
        # Keep normalization, base logits, and final logits out of reduced
        # precision. These scores directly drive both the loss and candidate
        # selection.
        with torch.amp.autocast(X.device.type, enabled=False):
            state = torch.nn.functional.normalize(state.to(torch.float32), dim=-1)
            geometry_outcome_embedding = torch.nn.functional.normalize(
                self.geometry_outcome_embedding.to(torch.float32), dim=-1
            )
            pos_embedding_x = torch.nn.functional.normalize(
                pos_embedding_x.to(torch.float32), dim=-1
            )
            pos_embedding_y = torch.nn.functional.normalize(
                pos_embedding_y.to(torch.float32), dim=-1
            )
            base_query = geometry_outcome_embedding[self.geometry_outcome_idx]
            base_logits = torch.matmul(state, base_query.transpose(0, 1))
            x_logits = torch.matmul(state, pos_embedding_x.transpose(0, 1))
            y_logits = torch.matmul(state, pos_embedding_y.transpose(0, 1))
            room_logits = torch.gather(x_logits, -1, room_x) + torch.gather(y_logits, -1, room_y)
            room_logits = torch.where(room_placed, room_logits, 0.0)
            position_logits = room_logits[..., self.room_idx]
            return (base_logits + position_logits) * torch.exp(
                torch.clamp(self.logit_scale.to(torch.float32), max=math.log(100.0))
            )


class ProposalOutput(torch.nn.Module):
    def __init__(
        self,
        input_width: int,
        hidden_width: int,
        output_width: int,
    ):
        super().__init__()
        if hidden_width <= 0:
            raise ValueError("proposal_hidden_width must be greater than zero")
        self.out_features = output_width
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_width, hidden_width, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_width, output_width, bias=False),
        )
        self.layers[-1].weight.data.zero_()

    @property
    def output_dtype(self) -> torch.dtype:
        return self.layers[-1].weight.dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class BoundedMessagePassingLayer(torch.nn.Module):
    def __init__(
        self,
        source_width: int,
        target_width: int,
        global_width: int,
        message_hidden_width: int,
        update_hidden_width: int,
        edge_width: int,
    ):
        super().__init__()
        if source_width <= 0:
            raise ValueError("source_width must be greater than zero")
        if target_width <= 0:
            raise ValueError("target_width must be greater than zero")
        if global_width <= 0:
            raise ValueError("global_width must be greater than zero")
        if message_hidden_width <= 0:
            raise ValueError("message_hidden_width must be greater than zero")
        if update_hidden_width <= 0:
            raise ValueError("update_hidden_width must be greater than zero")
        if edge_width < 0:
            raise ValueError("edge_width must be greater than or equal to zero")
        self.source_layer = torch.nn.Linear(source_width, message_hidden_width, bias=False)
        self.edge_layer = (
            torch.nn.Linear(edge_width, message_hidden_width, bias=False)
            if edge_width > 0
            else None
        )
        self.message_output_layer = torch.nn.Sequential(
            torch.nn.GELU(),
            torch.nn.Linear(message_hidden_width, target_width, bias=False),
        )
        self.update_layer = torch.nn.Sequential(
            torch.nn.Linear(
                target_width * 2 + global_width,
                update_hidden_width,
                bias=False,
            ),
            torch.nn.GELU(),
            torch.nn.Linear(update_hidden_width, target_width, bias=False),
        )

    def forward(
        self,
        target_state: torch.Tensor,
        source_state: torch.Tensor,
        neighbor: torch.Tensor,
        neighbor_mask: torch.Tensor,
        target_global_state: torch.Tensor,
        edge_features: torch.Tensor | None,
        extra_message_features: torch.Tensor | None,
    ) -> torch.Tensor:
        source = self.source_layer(source_state)
        source = source[neighbor]
        messages = source
        if self.edge_layer is not None:
            if edge_features is None:
                raise ValueError("edge_features are required when edge_width is greater than zero")
            messages = messages + self.edge_layer(edge_features)
        if extra_message_features is not None:
            messages = messages + extra_message_features
        messages = self.message_output_layer(messages) * neighbor_mask
        if neighbor.ndim != 1:
            neighbor_count = neighbor_mask.sum(1).clamp_min(1)
            messages = messages.sum(1) / neighbor_count
        return target_state + self.update_layer(
            torch.cat([target_state, messages, target_global_state], dim=-1)
        )


class FrontierModel(torch.nn.Module):
    def __init__(
        self,
        num_rooms,
        output_metadata: OutputMetadata,
        map_x,
        map_y,
        embedding_width,
        frontier_embedding_width,
        room_part_embedding_width,
        global_embedding_width,
        global_room_position_embedding_width,
        pooling_hidden_width,
        frontier_message_hidden_width,
        part_from_frontier_message_hidden_width,
        frontier_from_part_message_hidden_width,
        proposal_hidden_width,
        door_match_embedding_width,
        toilet_crossed_room_embedding_width,
        num_layers,
        door_counts,
        frontier_window_size,
        features: FeatureConfig,
    ):
        super().__init__()
        self.features = features
        self.num_rooms = num_rooms
        self.map_x = map_x
        self.map_y = map_y
        self.embedding_width = embedding_width
        self.frontier_embedding_width = frontier_embedding_width
        self.room_part_embedding_width = room_part_embedding_width
        self.global_embedding_width = global_embedding_width
        self.global_room_position_embedding_width = global_room_position_embedding_width
        self.num_room_parts = output_metadata.num_room_parts
        self.room_part_node_flag_width = 5
        width_checks = {
            "embedding_width": embedding_width,
            "frontier_embedding_width": frontier_embedding_width,
            "room_part_embedding_width": room_part_embedding_width,
            "global_embedding_width": global_embedding_width,
            "global_room_position_embedding_width": global_room_position_embedding_width,
            "pooling_hidden_width": pooling_hidden_width,
            "frontier_message_hidden_width": frontier_message_hidden_width,
            "part_from_frontier_message_hidden_width": part_from_frontier_message_hidden_width,
            "frontier_from_part_message_hidden_width": frontier_from_part_message_hidden_width,
            "proposal_hidden_width": proposal_hidden_width,
        }
        for name, width in width_checks.items():
            if width <= 0:
                raise ValueError(f"{name} must be greater than zero")
        self.left_count, self.right_count, self.up_count, self.down_count = door_counts
        if self.features.lookahead_outcomes and door_match_embedding_width <= 0:
            raise ValueError("door_match_embedding_width must be greater than zero")
        if self.features.toilet_crossed_room and toilet_crossed_room_embedding_width <= 0:
            raise ValueError("toilet_crossed_room_embedding_width must be greater than zero")
        door_output_size, connection_output_size = output_metadata.get_output_sizes()
        self.output_sizes = (
            door_output_size,
            connection_output_size,
            1,
            door_output_size,
            1,
        )
        if sum(door_counts) != door_output_size:
            raise ValueError("door_counts must sum to the door output size")
        self.num_connection_outputs = len(output_metadata.connection)
        self.include_inventory = self.features.inventory
        if self.features.global_room_position and not self.features.room_position:
            raise ValueError("features.global_room_position requires features.room_position")
        self.register_buffer(
            "room_connection_variant_idx",
            torch.tensor(output_metadata.room_connection_variant_idx, dtype=torch.int64),
        )
        # self.inventory_embedding = torch.nn.Parameter(
        #     torch.randn([output_metadata.num_room_connection_variants, embedding_width]) / math.sqrt(embedding_width)
        # ) if self.features.inventory else None
        self.orientation_embedding = (
            torch.nn.Embedding(2, frontier_embedding_width)
            if self.features.frontier_orientation
            else None
        )
        self.kind_embedding = (
            torch.nn.Embedding(256, frontier_embedding_width)
            if self.features.frontier_kind
            else None
        )
        node_numeric_width = (
            frontier_window_size**2 * self.features.frontier_occupancy
            + 2 * self.num_connection_outputs * self.features.frontier_connection_reachability
        )
        self.node_numeric = (
            torch.nn.Linear(node_numeric_width, frontier_embedding_width, bias=False)
            if node_numeric_width > 0
            else None
        )
        self.frontier_window_area = frontier_window_size**2
        self.register_buffer(
            "frontier_occupancy_bits",
            1 << torch.arange(8, dtype=torch.uint8),
            persistent=False,
        )
        pair_width = 3 * self.features.frontier_neighbor_flags
        use_neighbors = self.features.frontier_neighbor
        self.frontier_message_layers = torch.nn.ModuleList(
            [
                    BoundedMessagePassingLayer(
                    source_width=frontier_embedding_width,
                    target_width=frontier_embedding_width,
                    global_width=global_embedding_width,
                    message_hidden_width=frontier_message_hidden_width,
                    update_hidden_width=frontier_message_hidden_width,
                    edge_width=pair_width,
                )
                for _ in range(num_layers if use_neighbors else 0)
            ]
        )
        use_part_frontier = self.features.room_part_nodes and self.features.frontier_mask
        self.part_from_frontier_message_layers = torch.nn.ModuleList(
            [
                BoundedMessagePassingLayer(
                    source_width=frontier_embedding_width,
                    target_width=room_part_embedding_width,
                    global_width=global_embedding_width,
                    message_hidden_width=part_from_frontier_message_hidden_width,
                    update_hidden_width=part_from_frontier_message_hidden_width,
                    edge_width=10,
                )
                for _ in range(num_layers if use_part_frontier else 0)
            ]
        )
        self.frontier_from_part_message_layers = torch.nn.ModuleList(
            [
                BoundedMessagePassingLayer(
                    source_width=room_part_embedding_width,
                    target_width=frontier_embedding_width,
                    global_width=global_embedding_width,
                    message_hidden_width=frontier_from_part_message_hidden_width,
                    update_hidden_width=frontier_from_part_message_hidden_width,
                    edge_width=10,
                )
                for _ in range(num_layers if use_part_frontier else 0)
            ]
        )
        global_width = (
            output_metadata.num_room_connection_variants * self.features.inventory
            + embedding_width
            * (self.features.connection_reachability and self.num_connection_outputs > 0)
            + int(self.features.temperature)
            + int(self.features.recommended_candidates)
            + global_room_position_embedding_width * int(self.features.global_room_position)
            + 2 * self.num_room_parts * int(self.features.room_part_furthest_distance)
            + 2 * self.num_room_parts * int(self.features.room_part_save_distance)
            + 2 * self.num_room_parts * int(self.features.room_part_refill_distance)
            + 2 * self.num_room_parts * int(self.features.room_part_frontier_distance)
            + 4 * self.num_room_parts
            + (door_match_embedding_width + 2 * connection_output_size + 2)
            * int(self.features.lookahead_outcomes)
            + (toilet_crossed_room_embedding_width * int(self.features.toilet_crossed_room))
        )
        self.global_mlp = (
            torch.nn.Linear(global_width, global_embedding_width, bias=False)
            if global_width > 0
            else None
        )
        pooled_width = (
            output_metadata.num_room_connection_variants * self.features.inventory
            + 2 * frontier_embedding_width * self.features.frontier_mask
            + embedding_width
            * (self.features.connection_reachability and self.num_connection_outputs > 0)
            + int(self.features.temperature)
            + int(self.features.recommended_candidates)
            + global_room_position_embedding_width * int(self.features.global_room_position)
            + (door_match_embedding_width + 2 * connection_output_size + 2)
            * int(self.features.lookahead_outcomes)
            + (toilet_crossed_room_embedding_width * int(self.features.toilet_crossed_room))
            + 4 * self.num_room_parts
        )
        self.pooled_mlp = (
            torch.nn.Sequential(
                torch.nn.Linear(pooled_width, pooling_hidden_width, bias=False),
                torch.nn.GELU(),
                torch.nn.Linear(pooling_hidden_width, embedding_width, bias=False),
            )
            if pooled_width > 0
            else None
        )
        self.door_match_embedding_width = door_match_embedding_width
        self.left_door_match_embedding = self._door_match_embedding(
            self.left_count,
            self.right_count,
            door_match_embedding_width,
        )
        self.right_door_match_embedding = self._door_match_embedding(
            self.right_count,
            self.left_count,
            door_match_embedding_width,
        )
        self.up_door_match_embedding = self._door_match_embedding(
            self.up_count,
            self.down_count,
            door_match_embedding_width,
        )
        self.down_door_match_embedding = self._door_match_embedding(
            self.down_count,
            self.up_count,
            door_match_embedding_width,
        )
        self.connection_reachability_embedding = (
            torch.nn.Linear(self.num_connection_outputs, embedding_width, bias=False)
            if self.features.connection_reachability and self.num_connection_outputs > 0
            else None
        )
        self.toilet_crossed_room_embedding = (
            torch.nn.Embedding(num_rooms + 1, toilet_crossed_room_embedding_width)
            if self.features.toilet_crossed_room
            else None
        )
        self.room_part_furthest_distance_embedding = (
            torch.nn.Embedding(256, 1) if self.features.room_part_furthest_distance else None
        )
        self.room_part_save_distance_embedding = (
            torch.nn.Embedding(256, 1) if self.features.room_part_save_distance else None
        )
        self.room_part_refill_distance_embedding = (
            torch.nn.Embedding(256, 1) if self.features.room_part_refill_distance else None
        )
        self.room_part_frontier_distance_embedding = (
            torch.nn.Embedding(256, 1) if self.features.room_part_frontier_distance else None
        )
        self.known_distance_embedding = torch.nn.Embedding(256, 1)
        self.room_part_node_identity_embedding = (
            torch.nn.Embedding(self.num_room_parts, room_part_embedding_width)
            if self.features.room_part_nodes
            else None
        )
        self.room_part_node_flag_linear = (
            torch.nn.Linear(self.room_part_node_flag_width, room_part_embedding_width, bias=False)
            if self.features.room_part_nodes
            else None
        )
        self.room_part_node_distance_embedding = (
            torch.nn.Embedding(256, room_part_embedding_width)
            if self.features.room_part_nodes
            else None
        )
        self.frontier_pos_embedding_x = (
            torch.nn.Parameter(
                torch.randn([NUM_COORD_VALUES, frontier_embedding_width])
                / math.sqrt(frontier_embedding_width)
            )
            if self.features.frontier_position
            else None
        )
        self.frontier_pos_embedding_y = (
            torch.nn.Parameter(
                torch.randn([NUM_COORD_VALUES, frontier_embedding_width])
                / math.sqrt(frontier_embedding_width)
            )
            if self.features.frontier_position
            else None
        )
        self.frontier_relative_pos_embedding_x = (
            torch.nn.Parameter(
                torch.randn([NUM_COORD_VALUES, frontier_message_hidden_width])
                / math.sqrt(frontier_message_hidden_width)
            )
            if self.features.frontier_neighbor_position_embedding
            else None
        )
        self.frontier_relative_pos_embedding_y = (
            torch.nn.Parameter(
                torch.randn([NUM_COORD_VALUES, frontier_message_hidden_width])
                / math.sqrt(frontier_message_hidden_width)
            )
            if self.features.frontier_neighbor_position_embedding
            else None
        )
        self.global_room_pos_embedding_x = (
            torch.nn.Parameter(
                torch.randn(
                    [
                        output_metadata.num_room_connection_variants,
                        NUM_COORD_VALUES,
                        global_room_position_embedding_width,
                    ]
                )
                / math.sqrt(global_room_position_embedding_width)
            )
            if self.features.global_room_position
            else None
        )
        self.global_room_pos_embedding_y = (
            torch.nn.Parameter(
                torch.randn(
                    [
                        output_metadata.num_room_connection_variants,
                        NUM_COORD_VALUES,
                        global_room_position_embedding_width,
                    ]
                )
                / math.sqrt(global_room_position_embedding_width)
            )
            if self.features.global_room_position
            else None
        )
        self.pos_embedding_x = torch.nn.Parameter(
            torch.randn([NUM_COORD_VALUES, embedding_width]) / math.sqrt(embedding_width)
        )
        self.pos_embedding_y = torch.nn.Parameter(
            torch.randn([NUM_COORD_VALUES, embedding_width]) / math.sqrt(embedding_width)
        )
        door_output_metadata = torch.tensor(output_metadata.door, dtype=torch.int64).reshape(
            door_output_size,
            2,
        )
        self.register_buffer("door_variant_outcome_idx", door_output_metadata[:, 1])
        connection_room_part_idx = torch.tensor(
            output_metadata.connection_room_part_idx,
            dtype=torch.int64,
        ).reshape(self.num_connection_outputs, 2)
        self.register_buffer("connection_room_part_idx", connection_room_part_idx)
        self.door_output = torch.nn.Linear(embedding_width, output_metadata.num_door_variants)
        self.frontier_door_invalid_output = torch.nn.Linear(frontier_embedding_width, 1)
        self.connection_output = FactorizedOutcomeHead(
            output_metadata.connection, output_metadata.num_connection_variants, embedding_width
        )
        self.toilet_output = torch.nn.Linear(embedding_width, 1)
        self.balance_score_output = FactorizedOutcomeHead(
            output_metadata.door, output_metadata.num_door_variants, embedding_width
        )
        self.toilet_balance_score_output = torch.nn.Linear(embedding_width, 1)
        self.avg_frontiers_output = torch.nn.Linear(embedding_width, 1)
        self.graph_diameter_output = torch.nn.Linear(embedding_width, 1)
        self.save_to_room_utility_output = torch.nn.Linear(embedding_width, self.num_room_parts)
        self.save_from_room_utility_output = torch.nn.Linear(embedding_width, self.num_room_parts)
        self.refill_to_room_utility_output = torch.nn.Linear(embedding_width, self.num_room_parts)
        self.refill_from_room_utility_output = torch.nn.Linear(
            embedding_width,
            self.num_room_parts,
        )
        self.missing_connect_distance_output = torch.nn.Linear(
            embedding_width,
            self.num_connection_outputs,
        )
        self.local_save_refill_utility_output = torch.nn.Linear(room_part_embedding_width, 4)
        self.local_missing_connect_output = torch.nn.Linear(
            2 * room_part_embedding_width + global_embedding_width,
            2,
        )
        self.proposal_output = ProposalOutput(
            frontier_embedding_width,
            proposal_hidden_width,
            output_metadata.num_door_variants,
        )

    def _door_match_embedding(
        self,
        source_count: int,
        partner_count: int,
        width: int,
    ) -> torch.nn.Parameter | None:
        if not self.features.lookahead_outcomes:
            return None
        return torch.nn.Parameter(
            torch.randn([source_count, partner_count + 1, width]) / math.sqrt(width)
        )

    def _position_embedding(self, x, y, embedding_x, embedding_y, dtype, offset=0):
        x = x.to(torch.int64) + offset
        y = y.to(torch.int64) + offset
        return embedding_x[x].to(dtype) + embedding_y[y].to(dtype)

    def _global_room_position_features(self, features: Features, dtype):
        if not self.features.global_room_position:
            return None
        room_x = features.global_features.room_x.to(torch.int64) + COORD_OFFSET
        room_y = features.global_features.room_y.to(torch.int64) + COORD_OFFSET
        room_connection_variant_idx = (
            self.room_connection_variant_idx.to(room_x.device).unsqueeze(0).expand_as(room_x)
        )
        room_position = (
            self.global_room_pos_embedding_x[room_connection_variant_idx, room_x]
            + self.global_room_pos_embedding_y[room_connection_variant_idx, room_y]
        ).to(dtype)
        placed = features.global_features.room_placed.to(dtype).unsqueeze(-1)
        placed_count = placed.sum(dim=1).clamp_min(1)
        return (room_position * placed).sum(dim=1) / torch.sqrt(placed_count)

    def _pair_features(self, features, dtype):
        values = []
        if self.features.frontier_neighbor_flags:
            flags = features.frontier_features.frontier_neighbor_pair
            values.append(
                torch.stack(
                    [
                        (flags & 1 != 0).to(dtype),
                        (flags & 2 != 0).to(dtype),
                        (flags & 4 != 0).to(dtype),
                    ],
                    dim=-1,
                )
            )
        return torch.cat(values, dim=-1) if values else None

    def _part_frontier_edge_features(self, edge: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        edge = edge.view(edge.shape[0], edge.shape[1] // 10, 10)
        encoded_distance = edge[..., :2].to(dtype)
        distance_features = torch.where(
            encoded_distance > 0,
            encoded_distance.clamp_min(1).reciprocal(),
            torch.zeros_like(encoded_distance),
        )
        return torch.cat([distance_features, edge[..., 2:].to(dtype)], dim=-1)

    def _snapshot_local_neighbor_indices(
        self,
        raw_neighbor: torch.Tensor,
        target_snapshot_idx: torch.Tensor,
        source_snapshot_idx: torch.Tensor,
        snapshot_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_neighbor = raw_neighbor.clamp_min(0).to(torch.int64)
        source_count_by_snapshot = torch.bincount(
            source_snapshot_idx,
            minlength=snapshot_count,
        )
        source_start_by_snapshot = (
            source_count_by_snapshot.cumsum(0) - source_count_by_snapshot
        )
        source_neighbor_count = source_count_by_snapshot[target_snapshot_idx].unsqueeze(1)
        neighbor_valid = (raw_neighbor >= 0) & (local_neighbor < source_neighbor_count)
        neighbor = source_start_by_snapshot[target_snapshot_idx].unsqueeze(1) + local_neighbor
        neighbor = torch.where(neighbor_valid, neighbor, torch.zeros_like(neighbor))
        return neighbor, neighbor_valid.unsqueeze(-1)

    def _local_room_part_lookup(
        self,
        row_snapshot_idx: torch.Tensor,
        row_room_part_idx: torch.Tensor,
        snapshot_count: int,
    ) -> torch.Tensor:
        lookup = row_snapshot_idx.new_full(
            [snapshot_count, self.num_room_parts],
            -1,
            dtype=torch.int64,
        )
        row_count = row_room_part_idx.shape[0]
        if row_count == 0:
            return lookup
        row_snapshot_idx = row_snapshot_idx.to(torch.int64)
        row_room_part_idx = row_room_part_idx.to(torch.int64)
        torch._assert(
            torch.all((row_snapshot_idx >= 0) & (row_snapshot_idx < snapshot_count)),
            "room-part row snapshot index out of bounds",
        )
        torch._assert(
            torch.all((row_room_part_idx >= 0) & (row_room_part_idx < self.num_room_parts)),
            "room-part row part index out of bounds",
        )
        flat_lookup = lookup.flatten()
        flat_idx = row_snapshot_idx * self.num_room_parts + row_room_part_idx
        row_idx = torch.arange(row_count, device=flat_idx.device, dtype=torch.int64)
        flat_lookup.scatter_(0, flat_idx, row_idx)
        return lookup

    def _overlay_local_utility(
        self,
        base_utility: torch.Tensor,
        local_utility: torch.Tensor,
        flat_idx: torch.Tensor,
        enabled: torch.Tensor,
    ) -> torch.Tensor:
        if flat_idx.shape[0] == 0:
            return base_utility
        utility = base_utility.squeeze(1).clone()
        flat_utility = utility.flatten()
        existing = flat_utility.detach().gather(0, flat_idx)
        scatter_values = torch.where(enabled, local_utility, existing)
        flat_utility.scatter_(0, flat_idx, scatter_values.to(flat_utility.dtype))
        return utility.unsqueeze(1)

    def _overlay_local_save_refill_utilities(
        self,
        room_part_X: torch.Tensor,
        features: Features,
        save_to_room_utility: torch.Tensor,
        save_from_room_utility: torch.Tensor,
        refill_to_room_utility: torch.Tensor,
        refill_from_room_utility: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row_count = room_part_X.shape[0]
        if row_count == 0:
            return (
                save_to_room_utility,
                save_from_room_utility,
                refill_to_room_utility,
                refill_from_room_utility,
            )
        row_snapshot_idx = features.room_part_features.row_snapshot_idx.to(torch.int64)
        row_room_part_idx = features.room_part_features.row_room_part_idx.to(torch.int64)
        flat_idx = row_snapshot_idx * self.num_room_parts + row_room_part_idx
        flags = features.room_part_features.row_flags
        local_utility = torch.sigmoid(
            self.local_save_refill_utility_output(room_part_X).to(torch.float32)
        )
        return (
            self._overlay_local_utility(
                save_to_room_utility,
                local_utility[:, 0],
                flat_idx,
                flags & 2 != 0,
            ),
            self._overlay_local_utility(
                save_from_room_utility,
                local_utility[:, 1],
                flat_idx,
                flags & 1 != 0,
            ),
            self._overlay_local_utility(
                refill_to_room_utility,
                local_utility[:, 2],
                flat_idx,
                flags & 8 != 0,
            ),
            self._overlay_local_utility(
                refill_from_room_utility,
                local_utility[:, 3],
                flat_idx,
                flags & 4 != 0,
            ),
        )

    def _overlay_local_missing_connect_outputs(
        self,
        room_part_X: torch.Tensor,
        global_state: torch.Tensor,
        features: Features,
        connection_invalid: torch.Tensor,
        missing_connect_distance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.num_connection_outputs == 0 or room_part_X.shape[0] == 0:
            return connection_invalid, missing_connect_distance
        snapshot_count = global_state.shape[0]
        lookup = self._local_room_part_lookup(
            features.room_part_features.row_snapshot_idx,
            features.room_part_features.row_room_part_idx,
            snapshot_count,
        )
        endpoints = self.connection_room_part_idx.to(connection_invalid.device)
        source_row = lookup[:, endpoints[:, 0]]
        destination_row = lookup[:, endpoints[:, 1]]
        one_sided = (source_row >= 0) ^ (destination_row >= 0)
        torch._assert(
            torch.logical_not(one_sided).all(),
            "missing-connect local query has only one endpoint row",
        )
        query_valid = (source_row >= 0) & (destination_row >= 0)
        safe_source_row = source_row.clamp_min(0)
        safe_destination_row = destination_row.clamp_min(0)
        source_state = room_part_X[safe_source_row.reshape(-1)].view(
            snapshot_count,
            self.num_connection_outputs,
            self.room_part_embedding_width,
        )
        destination_state = room_part_X[safe_destination_row.reshape(-1)].view(
            snapshot_count,
            self.num_connection_outputs,
            self.room_part_embedding_width,
        )
        query_global_state = global_state.unsqueeze(1).expand(
            -1,
            self.num_connection_outputs,
            -1,
        )
        local_outputs = self.local_missing_connect_output(
            torch.cat([source_state, destination_state, query_global_state], dim=-1)
        )
        connection = connection_invalid.squeeze(1)
        distance = missing_connect_distance.squeeze(1)
        connection = torch.where(
            query_valid,
            local_outputs[..., 0].to(connection.dtype),
            connection,
        )
        distance = torch.where(
            query_valid,
            local_outputs[..., 1].to(distance.dtype),
            distance,
        )
        return connection.unsqueeze(1), distance.unsqueeze(1)

    def _activation_dtype(self, device: torch.device) -> torch.dtype:
        return activation_dtype(device, next(self.parameters()).dtype)

    def _direction_door_match_features(
        self,
        matches: torch.Tensor,
        embedding: torch.nn.Parameter | None,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if embedding is None or matches.shape[-1] == 0:
            return matches.new_zeros(
                [matches.shape[0], self.door_match_embedding_width],
                dtype=dtype,
            )
        known = matches >= 0
        safe_matches = matches.clamp(min=0).to(torch.int64)
        source_idx = torch.arange(
            embedding.shape[0],
            dtype=torch.int64,
            device=matches.device,
        ).unsqueeze(0)
        values = embedding.to(dtype)[source_idx, safe_matches]
        return torch.sum(values * known.unsqueeze(-1), dim=1)

    def _lookahead_outcome_features(
        self,
        features: Features,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        left, right, up, down = torch.split(
            features.global_features.lookahead_door_match,
            [self.left_count, self.right_count, self.up_count, self.down_count],
            dim=-1,
        )
        door_match_features = (
            self._direction_door_match_features(left, self.left_door_match_embedding, dtype)
            + self._direction_door_match_features(right, self.right_door_match_embedding, dtype)
            + self._direction_door_match_features(up, self.up_door_match_embedding, dtype)
            + self._direction_door_match_features(down, self.down_door_match_embedding, dtype)
        )
        connection_features = torch.stack(
            [
                (features.global_features.lookahead_connection_invalid == 0).to(dtype),
                (features.global_features.lookahead_connection_invalid == 1).to(dtype),
            ],
            dim=-1,
        ).flatten(1)
        toilet_features = torch.stack(
            [
                (features.global_features.lookahead_toilet_invalid == 0).to(dtype),
                (features.global_features.lookahead_toilet_invalid == 1).to(dtype),
            ],
            dim=-1,
        ).flatten(1)
        return torch.cat([door_match_features, connection_features, toilet_features], dim=-1)

    def _toilet_crossed_room_features(
        self,
        features: Features,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self.toilet_crossed_room_embedding is None:
            return None
        crossed_room = features.global_features.toilet_crossed_room_idx.to(torch.int64)
        torch._assert(
            torch.all((crossed_room >= -1) & (crossed_room < self.num_rooms)),
            "toilet_crossed_room_idx must be -1 or a valid room index",
        )
        return self.toilet_crossed_room_embedding(crossed_room + 1).squeeze(-2).to(dtype)

    def _room_part_furthest_distance_features(
        self,
        features: Features,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self.room_part_furthest_distance_embedding is None:
            return None
        distances = torch.cat(
            [
                features.global_features.room_part_furthest_destination,
                features.global_features.room_part_furthest_source,
            ],
            dim=-1,
        ).to(torch.int64)
        return self.room_part_furthest_distance_embedding(distances).flatten(1).to(dtype)

    def _room_part_save_distance_features(
        self,
        features: Features,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self.room_part_save_distance_embedding is None:
            return None
        distances = torch.cat(
            [
                features.global_features.room_part_save_from_room_distance,
                features.global_features.room_part_save_to_room_distance,
            ],
            dim=-1,
        ).to(torch.int64)
        return self.room_part_save_distance_embedding(distances).flatten(1).to(dtype)

    def _room_part_refill_distance_features(
        self,
        features: Features,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self.room_part_refill_distance_embedding is None:
            return None
        distances = torch.cat(
            [
                features.global_features.room_part_refill_from_room_distance,
                features.global_features.room_part_refill_to_room_distance,
            ],
            dim=-1,
        ).to(torch.int64)
        return self.room_part_refill_distance_embedding(distances).flatten(1).to(dtype)

    def _room_part_frontier_distance_features(
        self,
        features: Features,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self.room_part_frontier_distance_embedding is None:
            return None
        distances = torch.cat(
            [
                features.global_features.room_part_frontier_from_room_distance,
                features.global_features.room_part_frontier_to_room_distance,
            ],
            dim=-1,
        ).to(torch.int64)
        return self.room_part_frontier_distance_embedding(distances).flatten(1).to(dtype)

    def _known_distance_features(
        self,
        features: Features,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        distances = torch.cat(
            [
                features.global_features.known_save_from_room_distance,
                features.global_features.known_save_to_room_distance,
                features.global_features.known_refill_from_room_distance,
                features.global_features.known_refill_to_room_distance,
            ],
            dim=-1,
        ).to(torch.int64)
        return self.known_distance_embedding(distances).flatten(1).to(dtype)

    def _room_part_node_state(
        self,
        features: Features,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        room_part_idx = features.room_part_features.row_room_part_idx.to(torch.int64)
        row_count = room_part_idx.shape[0]
        if self.room_part_node_identity_embedding is None:
            return room_part_idx.new_zeros([row_count, self.room_part_embedding_width], dtype=dtype)
        snapshot_idx = features.room_part_features.row_snapshot_idx.to(torch.int64)
        state = self.room_part_node_identity_embedding(room_part_idx).to(dtype)
        flags = features.room_part_features.row_flags
        flag_features = torch.stack(
            [
                (flags & 1 != 0).to(dtype),
                (flags & 2 != 0).to(dtype),
                (flags & 4 != 0).to(dtype),
                (flags & 8 != 0).to(dtype),
                (flags & 16 != 0).to(dtype),
            ],
            dim=-1,
        )
        state = state + self.room_part_node_flag_linear(flag_features)
        distance_inputs = [
            features.global_features.known_save_from_room_distance,
            features.global_features.known_save_to_room_distance,
            features.global_features.known_refill_from_room_distance,
            features.global_features.known_refill_to_room_distance,
        ]
        if self.features.room_part_furthest_distance:
            distance_inputs.extend(
                [
                    features.global_features.room_part_furthest_destination,
                    features.global_features.room_part_furthest_source,
                ]
            )
        if self.features.room_part_save_distance:
            distance_inputs.extend(
                [
                    features.global_features.room_part_save_from_room_distance,
                    features.global_features.room_part_save_to_room_distance,
                ]
            )
        if self.features.room_part_refill_distance:
            distance_inputs.extend(
                [
                    features.global_features.room_part_refill_from_room_distance,
                    features.global_features.room_part_refill_to_room_distance,
                ]
            )
        if self.features.room_part_frontier_distance:
            distance_inputs.extend(
                [
                    features.global_features.room_part_frontier_from_room_distance,
                    features.global_features.room_part_frontier_to_room_distance,
                ]
            )
        distances = torch.stack(
            [values[snapshot_idx, room_part_idx] for values in distance_inputs],
            dim=-1,
        ).to(torch.int64)
        return state + self.room_part_node_distance_embedding(distances).sum(dim=1).to(dtype)

    def _relative_position_features(self, features, neighbor):
        if self.frontier_relative_pos_embedding_x is None:
            return None
        node = features.frontier_features.frontier
        raw_x = node[:, 1].to(torch.int64)
        raw_y = node[:, 2].to(torch.int64)
        raw_x0, raw_x1 = raw_x.unsqueeze(1), raw_x[neighbor]
        raw_y0, raw_y1 = raw_y.unsqueeze(1), raw_y[neighbor]
        return self._position_embedding(
            raw_x1 - raw_x0,
            raw_y1 - raw_y0,
            self.frontier_relative_pos_embedding_x,
            self.frontier_relative_pos_embedding_y,
            self._activation_dtype(features.frontier_features.frontier.device),
            COORD_OFFSET,
        )

    def forward(
        self,
        features: Features,
        return_proposal_state: bool,
    ):
        # Shapes below use: s=snapshot, r=frontier row, k=neighbors, e=embedding width,
        # h=message hidden width.
        # node: [r, 5]
        node = features.frontier_features.frontier
        row_snapshot_idx = features.frontier_features.row_snapshot_idx.to(torch.int64)
        snapshot_count = features.global_features.inventory.shape[0]
        row_count = node.shape[0]
        # numeric: [r, numeric_width]
        numeric = []
        dtype = self._activation_dtype(node.device)
        room_part_X = self._room_part_node_state(features, dtype)
        if self.features.frontier_occupancy:
            numeric.append(
                features.frontier_features.frontier_occupancy.unsqueeze(-1)
                .bitwise_and(self.frontier_occupancy_bits)
                .ne(0)
                .flatten(-2)[..., : self.frontier_window_area]
                .to(dtype)
            )
        if self.features.frontier_connection_reachability:
            flags = features.frontier_features.frontier_connection_reachability
            numeric.append(
                torch.stack(
                    [
                        (flags & 1 != 0).to(dtype),
                        (flags & 2 != 0).to(dtype),
                    ],
                    dim=-1,
                ).flatten(-2)
            )
        # X: [r, e]
        X = node.new_zeros([row_count, self.frontier_embedding_width], dtype=dtype)
        if self.node_numeric is not None:
            X = X + self.node_numeric(torch.cat(numeric, dim=-1))
        if self.frontier_pos_embedding_x is not None:
            X = X + self._position_embedding(
                node[:, 1],
                node[:, 2],
                self.frontier_pos_embedding_x,
                self.frontier_pos_embedding_y,
                dtype,
            )
        if self.orientation_embedding is not None:
            X = X + self.orientation_embedding(node[:, 3].to(torch.int64)).to(dtype)
        if self.kind_embedding is not None:
            X = X + self.kind_embedding(node[:, 4].to(torch.int64)).to(dtype)
        # if self.inventory_embedding is not None:
        #     X = X + torch.matmul(
        #         features.global_features.inventory.to(torch.float32), self.inventory_embedding
        #     ).unsqueeze(1)
        # if self.connection_reachability_embedding is not None:
        #     X = X + self.connection_reachability_embedding(
        #         features.global_features.connection_reachability.to(torch.float32)
        #     ).unsqueeze(1)
        inventory_features = features.global_features.inventory.to(X.dtype) if self.include_inventory else None
        connection_features = (
            self.connection_reachability_embedding(features.global_features.connection_reachability.to(X.dtype))
            if self.connection_reachability_embedding is not None
            else None
        )
        temperature_features = (
            features.global_features.log_temperature.to(X.dtype).unsqueeze(-1)
            if self.features.temperature
            else None
        )
        recommended_candidate_features = (
            features.global_features.log_recommended_candidates.to(X.dtype).unsqueeze(-1)
            if self.features.recommended_candidates
            else None
        )
        lookahead_features = (
            self._lookahead_outcome_features(features, X.dtype)
            if self.features.lookahead_outcomes
            else None
        )
        toilet_crossed_room_features = self._toilet_crossed_room_features(features, X.dtype)
        global_room_position_features = self._global_room_position_features(features, X.dtype)
        room_part_furthest_distance_features = self._room_part_furthest_distance_features(
            features,
            X.dtype,
        )
        room_part_save_distance_features = self._room_part_save_distance_features(
            features,
            X.dtype,
        )
        room_part_refill_distance_features = self._room_part_refill_distance_features(
            features,
            X.dtype,
        )
        room_part_frontier_distance_features = self._room_part_frontier_distance_features(
            features,
            X.dtype,
        )
        known_distance_features = self._known_distance_features(features, X.dtype)
        global_inputs = []
        if inventory_features is not None:
            global_inputs.append(inventory_features)
        if connection_features is not None:
            global_inputs.append(connection_features)
        if temperature_features is not None:
            global_inputs.append(temperature_features)
        if recommended_candidate_features is not None:
            global_inputs.append(recommended_candidate_features)
        if lookahead_features is not None:
            global_inputs.append(lookahead_features)
        if toilet_crossed_room_features is not None:
            global_inputs.append(toilet_crossed_room_features)
        if global_room_position_features is not None:
            global_inputs.append(global_room_position_features)
        if room_part_furthest_distance_features is not None:
            global_inputs.append(room_part_furthest_distance_features)
        if room_part_save_distance_features is not None:
            global_inputs.append(room_part_save_distance_features)
        if room_part_refill_distance_features is not None:
            global_inputs.append(room_part_refill_distance_features)
        if room_part_frontier_distance_features is not None:
            global_inputs.append(room_part_frontier_distance_features)
        global_inputs.append(known_distance_features)
        global_state = (
            self.global_mlp(torch.cat(global_inputs, dim=-1))
            if self.global_mlp is not None
            else X.new_zeros([snapshot_count, self.global_embedding_width])
        )
        if row_count == 0:
            mean_pool = max_pool = X.new_zeros([snapshot_count, self.frontier_embedding_width])
        else:
            row_count_by_snapshot = torch.bincount(
                row_snapshot_idx,
                minlength=snapshot_count,
            )
            row_start_by_snapshot = row_count_by_snapshot.cumsum(0) - row_count_by_snapshot
            # pair: [r, k, pair_width], neighbor: [r, k], pair_mask: [r, k, 1]
            pair = self._pair_features(features, dtype)
            frontier_neighbor = features.frontier_features.frontier_neighbor
            local_neighbor = frontier_neighbor.clamp_min(0).to(torch.int64)
            row_neighbor_count = row_count_by_snapshot[row_snapshot_idx].unsqueeze(1)
            neighbor_valid = (frontier_neighbor >= 0) & (local_neighbor < row_neighbor_count)
            neighbor = row_start_by_snapshot[row_snapshot_idx].unsqueeze(1) + local_neighbor
            pair_mask = neighbor_valid.unsqueeze(-1)
            relative_position = self._relative_position_features(features, neighbor)
            single_neighbor = neighbor.shape[1] == 1
            if single_neighbor:
                neighbor = neighbor[:, 0]
                pair_mask = pair_mask[:, 0]
                if pair is not None:
                    pair = pair[:, 0]
                if relative_position is not None:
                    relative_position = relative_position[:, 0]
            global_rows = global_state[row_snapshot_idx]
            room_part_row_count = room_part_X.shape[0]
            room_part_snapshot_idx = features.room_part_features.row_snapshot_idx.to(torch.int64)
            part_global_rows = global_state[room_part_snapshot_idx]
            part_frontier_neighbor = features.room_part_features.part_frontier_neighbor
            frontier_room_part_neighbor = features.room_part_features.frontier_room_part_neighbor
            part_frontier_edge = self._part_frontier_edge_features(
                features.room_part_features.part_frontier_edge,
                dtype,
            )
            frontier_room_part_edge = self._part_frontier_edge_features(
                features.room_part_features.frontier_room_part_edge,
                dtype,
            )
            layer_count = max(
                len(self.frontier_message_layers),
                len(self.part_from_frontier_message_layers),
                len(self.frontier_from_part_message_layers),
            )
            for layer_idx in range(layer_count):
                if layer_idx < len(self.frontier_message_layers):
                    X = self.frontier_message_layers[layer_idx](
                        target_state=X,
                        source_state=X,
                        neighbor=neighbor,
                        neighbor_mask=pair_mask,
                        target_global_state=global_rows,
                        edge_features=pair,
                        extra_message_features=relative_position,
                    )
                if (
                    layer_idx < len(self.part_from_frontier_message_layers)
                    and row_count > 0
                    and room_part_row_count > 0
                    and part_frontier_neighbor.shape[1] > 0
                ):
                    part_neighbor, part_neighbor_mask = self._snapshot_local_neighbor_indices(
                        part_frontier_neighbor,
                        room_part_snapshot_idx,
                        row_snapshot_idx,
                        snapshot_count,
                    )
                    room_part_X = self.part_from_frontier_message_layers[layer_idx](
                        target_state=room_part_X,
                        source_state=X,
                        neighbor=part_neighbor,
                        neighbor_mask=part_neighbor_mask,
                        target_global_state=part_global_rows,
                        edge_features=part_frontier_edge,
                        extra_message_features=None,
                    )
                if (
                    layer_idx < len(self.frontier_from_part_message_layers)
                    and row_count > 0
                    and room_part_row_count > 0
                    and frontier_room_part_neighbor.shape[1] > 0
                ):
                    frontier_part_neighbor, frontier_part_neighbor_mask = (
                        self._snapshot_local_neighbor_indices(
                            frontier_room_part_neighbor,
                            row_snapshot_idx,
                            room_part_snapshot_idx,
                            snapshot_count,
                        )
                    )
                    X = self.frontier_from_part_message_layers[layer_idx](
                        target_state=X,
                        source_state=room_part_X,
                        neighbor=frontier_part_neighbor,
                        neighbor_mask=frontier_part_neighbor_mask,
                        target_global_state=global_rows,
                        edge_features=frontier_room_part_edge,
                        extra_message_features=None,
                    )
            if X.device.type == "cuda":
                mean_pool = X.new_zeros([snapshot_count, self.frontier_embedding_width])
                mean_pool.index_add_(0, row_snapshot_idx, X)
                count = row_count_by_snapshot.to(X.dtype).unsqueeze(1).clamp_min(1)
                mean_pool = mean_pool / count
                max_pool = X.new_full([snapshot_count, self.frontier_embedding_width], -torch.inf)
                max_pool.scatter_reduce_(
                    0,
                    row_snapshot_idx.unsqueeze(1).expand(-1, self.frontier_embedding_width),
                    X,
                    reduce="amax",
                    include_self=True,
                )
            else:
                count = row_count_by_snapshot.to(X.dtype).unsqueeze(1).clamp_min(1)
                mean_pool = (
                    torch.segment_reduce(
                        X,
                        "sum",
                        lengths=row_count_by_snapshot,
                        axis=0,
                    )
                    / count
                )
                max_pool = torch.segment_reduce(
                    X,
                    "max",
                    lengths=row_count_by_snapshot,
                    axis=0,
                )
            max_pool = torch.where(torch.isfinite(max_pool), max_pool, 0)
        frontier_door_invalid = self.frontier_door_invalid_output(X)
        proposal_state = X if return_proposal_state else X.new_empty([row_count, 0])
        # mean_pool, max_pool, pooled_state: [s, e]
        pooled_inputs = []
        if inventory_features is not None:
            # pooled_inputs.append(torch.matmul(features.global_features.inventory.to(torch.float32), self.inventory_embedding))
            pooled_inputs.append(inventory_features)
        if self.features.frontier_mask:
            pooled_inputs.extend([mean_pool, max_pool])
        if connection_features is not None:
            pooled_inputs.append(connection_features)
        if temperature_features is not None:
            pooled_inputs.append(temperature_features)
        if recommended_candidate_features is not None:
            pooled_inputs.append(recommended_candidate_features)
        if lookahead_features is not None:
            pooled_inputs.append(lookahead_features)
        if toilet_crossed_room_features is not None:
            pooled_inputs.append(toilet_crossed_room_features)
        if global_room_position_features is not None:
            pooled_inputs.append(global_room_position_features)
        pooled_inputs.append(known_distance_features)
        pooled_state = (
            self.pooled_mlp(torch.cat(pooled_inputs, dim=-1))
            if self.pooled_mlp is not None
            else X.new_zeros([snapshot_count, self.embedding_width])
        )
        if self.features.room_position:
            room_x = (features.global_features.room_x.to(torch.int64) + COORD_OFFSET).unsqueeze(1)
            room_y = (features.global_features.room_y.to(torch.int64) + COORD_OFFSET).unsqueeze(1)
            room_placed = features.global_features.room_placed.to(torch.bool).unsqueeze(1)
        else:
            room_x = torch.full(
                [snapshot_count, 1, self.num_rooms],
                COORD_OFFSET,
                dtype=torch.int64,
                device=X.device,
            )
            room_y = room_x
            room_placed = torch.zeros(
                [snapshot_count, 1, self.num_rooms], dtype=torch.bool, device=X.device
            )
        # X: [s, 1, e]
        X = pooled_state.unsqueeze(1)
        door_variant = self.door_output(X)
        door = door_variant[..., self.door_variant_outcome_idx]
        connection = self.connection_output(
            X, room_x, room_y, room_placed, self.pos_embedding_x, self.pos_embedding_y
        )
        toilet = self.toilet_output(X)
        balance_score = self.balance_score_output(
            X,
            room_x,
            room_y,
            room_placed,
            self.pos_embedding_x,
            self.pos_embedding_y,
        )
        toilet_balance_score = self.toilet_balance_score_output(X)
        avg_frontiers = self.avg_frontiers_output(X).squeeze(-1).to(torch.float32)
        graph_diameter = self.graph_diameter_output(X).squeeze(-1).to(torch.float32)
        save_to_room_utility = torch.sigmoid(
            self.save_to_room_utility_output(X).to(torch.float32)
        )
        save_from_room_utility = torch.sigmoid(
            self.save_from_room_utility_output(X).to(torch.float32)
        )
        refill_to_room_utility = torch.sigmoid(
            self.refill_to_room_utility_output(X).to(torch.float32)
        )
        refill_from_room_utility = torch.sigmoid(
            self.refill_from_room_utility_output(X).to(torch.float32)
        )
        missing_connect_distance = self.missing_connect_distance_output(X).to(torch.float32)
        (
            save_to_room_utility,
            save_from_room_utility,
            refill_to_room_utility,
            refill_from_room_utility,
        ) = self._overlay_local_save_refill_utilities(
            room_part_X,
            features,
            save_to_room_utility,
            save_from_room_utility,
            refill_to_room_utility,
            refill_from_room_utility,
        )
        preds = get_predictions(
            torch.cat([door, connection, toilet, balance_score, toilet_balance_score], dim=-1),
            self.output_sizes,
        )
        door_invalid = apply_frontier_door_invalid_logits(
            preds.door_invalid,
            frontier_door_invalid,
            row_snapshot_idx,
            features.frontier_features.row_door_output_idx,
        )
        connection_invalid, missing_connect_distance = (
            self._overlay_local_missing_connect_outputs(
                room_part_X,
                global_state,
                features,
                preds.connection_invalid,
                missing_connect_distance,
            )
        )
        door_invalid = apply_known_invalid_logits(
            door_invalid,
            features.global_features.lookahead_door_invalid,
            "door",
        )
        connection_invalid = apply_known_invalid_logits(
            connection_invalid,
            features.global_features.lookahead_connection_invalid,
            "connection",
        )
        return Predictions(
            door_invalid=door_invalid,
            connection_invalid=connection_invalid,
            toilet_invalid=preds.toilet_invalid,
            balance_score=preds.balance_score,
            toilet_balance_score=preds.toilet_balance_score,
            avg_frontiers=avg_frontiers,
            graph_diameter=graph_diameter,
            save_to_room_utility=save_to_room_utility,
            save_from_room_utility=save_from_room_utility,
            refill_to_room_utility=refill_to_room_utility,
            refill_from_room_utility=refill_from_room_utility,
            missing_connect_distance=missing_connect_distance,
            proposal_score=X.new_empty([row_count, 0]),
            proposal_state=proposal_state,
            proposal_row_snapshot_idx=(
                row_snapshot_idx if return_proposal_state else row_snapshot_idx.new_empty([0])
            ),
            proposal_row_frontier_idx=(
                features.frontier_features.row_frontier_idx
                if return_proposal_state
                else features.frontier_features.row_frontier_idx.new_empty([0])
            ),
        )


class BalanceModel(torch.nn.Module):
    def __init__(
        self,
        left_count: int,
        right_count: int,
        up_count: int,
        down_count: int,
        num_rooms: int,
        hidden_width: int,
        num_layers: int,
    ):
        super().__init__()
        if hidden_width <= 0:
            raise ValueError("balance model hidden_width must be greater than zero")
        if num_layers <= 0:
            raise ValueError("balance model num_layers must be greater than zero")
        self.left_count = left_count
        self.right_count = right_count
        self.up_count = up_count
        self.down_count = down_count
        self.num_rooms = num_rooms
        self.output_width = (
            left_count * right_count
            + right_count * left_count
            + up_count * down_count
            + down_count * up_count
            + num_rooms
        )

        layers: list[torch.nn.Module] = []
        input_width = 1
        for _ in range(num_layers):
            layers.extend(
                [
                    torch.nn.Linear(input_width, hidden_width),
                    torch.nn.GELU(),
                ]
            )
            input_width = hidden_width
        output_layer = torch.nn.Linear(input_width, self.output_width)
        output_layer.weight.data.zero_()
        layers.append(output_layer)
        self.net = torch.nn.Sequential(*layers)

    def forward(self, log_temperature: torch.Tensor) -> BalancePredictions:
        parameter_dtype = next(self.parameters()).dtype
        raw = self.net(
            log_temperature.to(
                activation_dtype(log_temperature.device, parameter_dtype)
            ).unsqueeze(-1)
        ).to(torch.float32)
        offset = 0
        left_size = self.left_count * self.right_count
        right_size = self.right_count * self.left_count
        up_size = self.up_count * self.down_count
        down_size = self.down_count * self.up_count
        left = raw[:, offset : offset + left_size].reshape(
            log_temperature.shape[0], self.left_count, self.right_count
        )
        offset += left_size
        right = raw[:, offset : offset + right_size].reshape(
            log_temperature.shape[0], self.right_count, self.left_count
        )
        offset += right_size
        up = raw[:, offset : offset + up_size].reshape(
            log_temperature.shape[0], self.up_count, self.down_count
        )
        offset += up_size
        down = raw[:, offset : offset + down_size].reshape(
            log_temperature.shape[0], self.down_count, self.up_count
        )
        offset += down_size
        toilet_crossed_room = raw[:, offset : offset + self.num_rooms]
        return BalancePredictions(
            left=left,
            right=right,
            up=up,
            down=down,
            toilet_crossed_room=toilet_crossed_room,
        )
