import unittest

import torch

from env import SaveRefillUtilityQueryFeatures
from model import SaveRefillUtilityQueryHead


def query_features(query_order: list[int]) -> SaveRefillUtilityQueryFeatures:
    # The first room part needs separate queries for its two travel directions.
    # The second snapshot also includes consolidated and partial target masks.
    return SaveRefillUtilityQueryFeatures(
        query_snapshot_idx=torch.tensor([0, 0, 1, 1])[query_order],
        query_room_part_idx=torch.tensor([1, 1, 1, 2])[query_order],
        target_mask=torch.tensor([0b1010, 0b0101, 0b1111, 0b0001])[query_order],
        frontier=torch.tensor([0, 1, 0, 1])[query_order],
        frontier_distance=torch.zeros(4, dtype=torch.uint8)[query_order],
        save_to_current_distance=torch.full((4,), 255, dtype=torch.uint8)[query_order],
        save_from_current_distance=torch.full((4,), 255, dtype=torch.uint8)[query_order],
        refill_to_current_distance=torch.full((4,), 255, dtype=torch.uint8)[query_order],
        refill_from_current_distance=torch.full((4,), 255, dtype=torch.uint8)[query_order],
    )


class SaveRefillUtilityQueryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.head = SaveRefillUtilityQueryHead(
            embedding_width=4,
            hidden_width=4,
            frontier_width=4,
        )
        # Expose each frontier's input values as its four raw query predictions.
        output_layer = torch.nn.Linear(self.head.output_layers[0].in_features, 4, bias=False)
        with torch.no_grad():
            self.head.frontier_projection.weight.copy_(torch.eye(4))
            output_layer.weight.zero_()
            output_layer.weight[:, :4].copy_(torch.eye(4))
        self.head.output_layers = output_layer
        self.frontier_state = torch.tensor(
            [
                [11.0, 12.0, 13.0, 14.0],
                [21.0, 22.0, 23.0, 24.0],
                [31.0, 32.0, 33.0, 34.0],
                [41.0, 42.0, 43.0, 44.0],
            ],
            requires_grad=True,
        )
        self.row_counts = torch.tensor([2, 2])
        self.row_starts = torch.tensor([0, 2])
        self.expected_output = torch.zeros(4, 2, 1, 3)
        self.expected_output[:, 0, 0, 1] = torch.tensor([21.0, 12.0, 23.0, 14.0])
        self.expected_output[:, 1, 0, 1] = torch.tensor([31.0, 32.0, 33.0, 34.0])
        self.expected_output[0, 1, 0, 2] = 41.0
        self.expected_mask = torch.zeros(4, 2, 1, 3, dtype=torch.bool)
        self.expected_mask[:, :, 0, 1] = True
        self.expected_mask[0, 1, 0, 2] = True
        self.expected_gradient = torch.tensor(
            [
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        )

    def test_split_queries_preserve_active_outputs_in_either_order(self) -> None:
        for query_order in ([0, 1, 2, 3], [3, 2, 1, 0]):
            with self.subTest(query_order=query_order):
                output, mask = self.head(
                    self.frontier_state,
                    self.row_counts,
                    self.row_starts,
                    query_features(query_order),
                    room_part_count=3,
                )
                torch.testing.assert_close(output, self.expected_output)
                torch.testing.assert_close(mask, self.expected_mask)

    def test_zero_predictions_preserve_active_masks(self) -> None:
        output, mask = self.head(
            torch.zeros_like(self.frontier_state),
            self.row_counts,
            self.row_starts,
            query_features([0, 1, 2, 3]),
            room_part_count=3,
        )
        torch.testing.assert_close(output, torch.zeros_like(self.expected_output))
        torch.testing.assert_close(mask, self.expected_mask)

    def test_gradients_only_flow_to_active_query_targets(self) -> None:
        output, mask = self.head(
            self.frontier_state,
            self.row_counts,
            self.row_starts,
            query_features([0, 1, 2, 3]),
            room_part_count=3,
        )
        output[mask].sum().backward()
        torch.testing.assert_close(self.frontier_state.grad, self.expected_gradient)

    def test_split_queries_compile_without_graph_breaks(self) -> None:
        compiled_head = torch.compile(self.head, backend="aot_eager", fullgraph=True, dynamic=True)
        output, mask = compiled_head(
            self.frontier_state,
            self.row_counts,
            self.row_starts,
            query_features([0, 1, 2, 3]),
            room_part_count=3,
        )
        torch.testing.assert_close(output, self.expected_output)
        torch.testing.assert_close(mask, self.expected_mask)
        output[mask].sum().backward()
        torch.testing.assert_close(self.frontier_state.grad, self.expected_gradient)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_split_queries_eager_and_compiled(self) -> None:
        device = torch.device("cuda")
        head = self.head.to(device)
        frontier_state = self.frontier_state.detach().to(device).requires_grad_()
        row_counts = self.row_counts.to(device)
        row_starts = self.row_starts.to(device)
        for backend in ("eager", "inductor"):
            runner = (
                head
                if backend == "eager"
                else torch.compile(head, backend=backend, fullgraph=True, dynamic=True)
            )
            for query_order in ([0, 1, 2, 3], [3, 2, 1, 0]):
                with self.subTest(backend=backend, query_order=query_order):
                    frontier_state.grad = None
                    output, mask = runner(
                        frontier_state,
                        row_counts,
                        row_starts,
                        query_features(query_order).to(device),
                        room_part_count=3,
                    )
                    torch.testing.assert_close(output.cpu(), self.expected_output)
                    torch.testing.assert_close(mask.cpu(), self.expected_mask)
                    output[mask].sum().backward()
                    torch.testing.assert_close(frontier_state.grad.cpu(), self.expected_gradient)


if __name__ == "__main__":
    unittest.main()
