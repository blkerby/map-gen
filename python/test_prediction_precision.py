import unittest

import torch

from env import MissingConnectQueryFeatures, SaveRefillUtilityQueryFeatures
from model import (
    Float32Linear,
    MissingConnectQueryHead,
    ProposalOutput,
    SaveRefillUtilityQueryHead,
    balance_price_network,
)


def missing_queries() -> MissingConnectQueryFeatures:
    return MissingConnectQueryFeatures(
        query_snapshot_idx=torch.tensor([0, 1]),
        query_connection_idx=torch.tensor([0, 0]),
        source_frontier=torch.tensor([[0], [0]]),
        target_frontier=torch.tensor([[1], [1]]),
        source_distance=torch.tensor([[2], [3]], dtype=torch.uint8),
        target_distance=torch.tensor([[4], [5]], dtype=torch.uint8),
        current_distance=torch.tensor([255, 8], dtype=torch.uint8),
    )


def utility_queries() -> SaveRefillUtilityQueryFeatures:
    return SaveRefillUtilityQueryFeatures(
        query_snapshot_idx=torch.tensor([0, 1]),
        query_room_part_idx=torch.tensor([0, 1]),
        target_mask=torch.tensor([15, 5]),
        frontier=torch.tensor([0, 1]),
        frontier_distance=torch.tensor([2, 3], dtype=torch.uint8),
        save_to_current_distance=torch.tensor([255, 8], dtype=torch.uint8),
        save_from_current_distance=torch.tensor([8, 255], dtype=torch.uint8),
        refill_to_current_distance=torch.tensor([255, 12], dtype=torch.uint8),
        refill_from_current_distance=torch.tensor([12, 255], dtype=torch.uint8),
    )


class PredictionPrecisionTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(37)
        self.state = torch.randn(4, 8).to(torch.bfloat16).requires_grad_()
        self.counts = torch.tensor([2, 2])
        self.starts = torch.tensor([0, 2])

    def check_head(self, head, args) -> None:
        # Nonzero output weights expose rounding errors and exercise backward.
        with torch.no_grad():
            for parameter in head.parameters():
                parameter.uniform_(-0.3, 0.3)
        expected = head(self.state.float(), *args)
        expected = (expected,) if isinstance(expected, torch.Tensor) else expected
        for runner in (head, torch.compile(head, backend="aot_eager", fullgraph=True)):
            head.zero_grad(set_to_none=True)
            self.state.grad = None
            with torch.amp.autocast("cpu", dtype=torch.bfloat16):
                actual = runner(self.state, *args)
            actual = (actual,) if isinstance(actual, torch.Tensor) else actual
            loss = torch.zeros(())
            for result, reference in zip(actual, expected, strict=True):
                torch.testing.assert_close(result, reference, rtol=0, atol=0)
                if result.is_floating_point():
                    self.assertEqual(result.dtype, torch.float32)
                    loss = loss + result.sum()
            loss.backward()
            self.assertIsNotNone(self.state.grad)
            gradients = [p.grad for p in head.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(g.dtype == torch.float32 for g in gradients))
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertTrue(any(torch.count_nonzero(g) > 0 for g in gradients))

    def test_linear_head(self) -> None:
        self.check_head(Float32Linear(8, 3), ())

    def test_proposal_head(self) -> None:
        self.check_head(ProposalOutput(8, [12], 3), ())

    def test_missing_connection_head(self) -> None:
        head = MissingConnectQueryHead(8, 4, 2, 12)
        self.check_head(head, (2, self.counts, self.starts, missing_queries(), 1))

    def test_save_refill_head(self) -> None:
        head = SaveRefillUtilityQueryHead(8, 12, 4)
        self.check_head(head, (self.counts, self.starts, utility_queries(), 2))

    def test_empty_query_heads_return_float32(self) -> None:
        state = self.state[:0]
        counts = torch.zeros(2, dtype=torch.int64)
        starts = torch.zeros(2, dtype=torch.int64)
        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            missing = MissingConnectQueryHead(8, 4, 2, 12)(
                state, 2, counts, starts, missing_queries(), 1
            )
            utility = SaveRefillUtilityQueryHead(8, 12, 4)(
                state, counts, starts, utility_queries(), 2
            )
        for result in (*missing, *utility):
            if result.is_floating_point():
                self.assertEqual(result.dtype, torch.float32)
                self.assertEqual(torch.count_nonzero(result), 0)

    def test_balance_output_layer(self) -> None:
        network = balance_price_network(hidden_width=8, num_layers=1, output_width=3)
        self.check_head(network[-1], ())


if __name__ == "__main__":
    unittest.main()
