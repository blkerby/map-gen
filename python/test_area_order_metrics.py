import torch

from train import compute_area_order_ss


def test_area_order_ss_corrects_finite_sample_bias() -> None:
    # Every rank has two observations of each area: the raw SS is 1/6,
    # whereas the unbiased estimate is 6 * 2 * 1 / (12 * 11) = 1/11.
    areas = torch.arange(6)
    order = (areas[:, None] + areas[None, :]) % 6
    rank_ss = compute_area_order_ss(order.repeat(2, 1))
    torch.testing.assert_close(rank_ss, torch.full((6,), 1 / 11, dtype=torch.float64))


def test_area_order_ss_averages_ranks_equally_and_excludes_missing() -> None:
    order = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2, 3, 4, 5],
            [1, 0, 2, 3, 4, -1],
            [1, 2, 0, 3, -1, -1],
        ]
    )
    # Ranks 5 and 6 are always the same area when reached, so both score 1.
    rank_ss = compute_area_order_ss(order)
    expected = torch.tensor([1 / 3, 1 / 6, 1 / 2, 1, 1, 1], dtype=torch.float64)
    torch.testing.assert_close(rank_ss, expected)
    torch.testing.assert_close(rank_ss.mean(), torch.tensor(2 / 3, dtype=torch.float64))


def test_area_order_ss_requires_two_observations_per_rank() -> None:
    order = torch.tensor([[0, 1, -1, -1, -1, -1], [0, -1, -1, -1, -1, -1]])
    rank_ss = compute_area_order_ss(order)
    assert rank_ss[0] == 1
    assert torch.isnan(rank_ss[1:]).all()
    assert torch.isnan(rank_ss.mean())
    assert torch.isnan(compute_area_order_ss(torch.empty((0, 6), dtype=torch.int64))).all()


if __name__ == "__main__":
    test_area_order_ss_corrects_finite_sample_bias()
    test_area_order_ss_averages_ranks_equally_and_excludes_missing()
    test_area_order_ss_requires_two_observations_per_rank()
