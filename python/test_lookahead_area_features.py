from types import SimpleNamespace

import torch

from features import LookaheadFeature


def test_lookahead_area_buckets_use_unknown_zero_and_known_one_hot() -> None:
    feature = LookaheadFeature(
        left_count=0,
        right_count=0,
        up_count=0,
        down_count=0,
        door_match_width=2,
    )
    features = SimpleNamespace(
        global_features=SimpleNamespace(
            lookahead_door_match=torch.empty([1, 0], dtype=torch.int16),
            lookahead_connection_invalid=torch.empty([1, 0], dtype=torch.int8),
            lookahead_toilet_invalid=torch.tensor([-1], dtype=torch.int8),
            lookahead_phantoon_invalid=torch.tensor([-1], dtype=torch.int8),
            lookahead_area_size_bucket=torch.tensor([[-1, 0, 1, 2, -1, 1]], dtype=torch.int8),
            lookahead_area_map_station_count_bucket=torch.tensor(
                [[2, 1, 0, -1, 1, 2]], dtype=torch.int8
            ),
        )
    )

    result = feature(features, torch.float32)

    assert result.shape == (1, 42)
    area_features = result[0, 6:]
    assert torch.equal(area_features[:3], torch.zeros(3))
    assert torch.equal(area_features[3:6], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(area_features[6:9], torch.tensor([0.0, 1.0, 0.0]))
    assert torch.equal(area_features[9:12], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.equal(area_features[18:21], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.equal(area_features[27:30], torch.zeros(3))
