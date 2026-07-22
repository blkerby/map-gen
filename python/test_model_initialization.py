import torch

from env import OutputMetadata
from model import FrontierModel
from train_config import FeatureConfig


def query_features() -> FeatureConfig:
    return FeatureConfig(
        inventory=False,
        temperature=False,
        recommended_candidates=False,
        generation_variable_floats=False,
        lookahead_outcomes=0,
        room_position=False,
        global_room_position=0,
        room_part_furthest_distance=0,
        room_part_save_distance=0,
        room_part_refill_distance=0,
        room_part_frontier_distance=0,
        frontier_mask=False,
        frontier_position=0,
        frontier_orientation=0,
        frontier_kind=0,
        frontier_door_variant=0,
        frontier_occupancy=False,
        frontier_neighbor=False,
        frontier_neighbor_position_embedding=0,
        frontier_neighbor_flags=False,
        connection_reachability=0,
        frontier_connection_reachability=False,
        area_state=False,
        frontier_area=0,
        missing_connect_query=True,
        save_utility_query=True,
        refill_utility_query=True,
        toilet_crossed_room=0,
        known_distance=0,
    )


def output_metadata() -> OutputMetadata:
    return OutputMetadata(
        door=[(0, 0), (1, 1)],
        connection=[(0, 0)],
        num_door_variants=2,
        num_connection_variants=1,
        room_connection_variant_idx=[0, 0],
        num_room_connection_variants=1,
        num_room_parts=2,
        door_variant_compatibility=torch.ones([2, 2], dtype=torch.bool),
        door_variant_connection_variant_idx=torch.zeros([2], dtype=torch.int64),
    )


def test_frontier_model_output_heads_are_zero_initialized() -> None:
    model = FrontierModel(
        num_rooms=2,
        output_metadata=output_metadata(),
        map_x=4,
        map_y=4,
        embedding_width=8,
        global_embedding_width=8,
        hidden_width=8,
        proposal_hidden_widths=[4],
        missing_connect_query_hidden_width=8,
        missing_connect_query_frontier_width=4,
        missing_connect_query_distance_width=2,
        utility_query_hidden_width=8,
        utility_query_frontier_width=4,
        known_save_refill_utility_override=False,
        distance_proximity_scale=1.0,
        num_layers=1,
        door_counts=(1, 1, 0, 0),
        frontier_window_size=2,
        area_bounding_box_width=4,
        area_bounding_box_height=4,
        max_area_size=16,
        features=query_features(),
    )
    assert model.missing_connect_query_output is not None
    assert model.save_refill_utility_query_output is not None
    output_layers = (
        model.door_output,
        model.frontier_door_invalid_output,
        model.frontier_balance_score_output,
        model.connection_output,
        model.missing_connect_query_output.output_layers[-1],
        model.toilet_output,
        model.phantoon_output,
        model.balance_score_output,
        model.toilet_balance_score_output,
        model.avg_frontiers_output,
        model.graph_diameter_output,
        model.save_to_room_utility_output,
        model.save_from_room_utility_output,
        model.refill_to_room_utility_output,
        model.refill_from_room_utility_output,
        model.save_refill_utility_query_output.output_layers[-1],
        model.missing_connect_utility_output,
        model.area_crossings_output,
        model.area_size_output,
        model.area_map_station_count_output,
        model.proposal_output.layers[-1],
    )

    for output_layer in output_layers:
        assert torch.count_nonzero(output_layer.weight) == 0
        if output_layer.bias is not None:
            assert torch.count_nonzero(output_layer.bias) == 0
