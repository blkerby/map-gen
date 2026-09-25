import json
from pathlib import Path
import unittest

import torch

from loss import balance_family_loss, balance_objective_terms, terminal_balance_cost
from room_step import (
    candidate_step_balance_score, episode_room_steps, proposal_step_prices, remaining_step_price,
)
from test_balance_variants import example_predictions, uniform_area_targets
from loss import compute_balance_price_tables
from train_config import Config, validate_config
from train import compute_room_step_ss


class RoomStepTest(unittest.TestCase):
    def test_steps_include_initial_room_and_preserve_gaps(self):
        rooms = torch.tensor([[2, 4, 0, 4], [1, 0, 3, 2]], dtype=torch.uint8)
        self.assertEqual(episode_room_steps(rooms, 4).tolist(), [[2, -1, 0, -1], [1, 0, 3, 2]])

    def test_future_target_excludes_placed_rooms_and_keeps_failures(self):
        prices = torch.arange(9.0).reshape(1, 3, 3)
        terminal = terminal_balance_cost(prices, torch.tensor([[10., 20., 30.]]), torch.tensor([[0, -1, 2]]))
        torch.testing.assert_close(
            remaining_step_price(terminal, torch.tensor([[True, False, False]])), torch.tensor([28.]),
        )

    def test_candidate_cost_counts_current_placement_once_and_honors_horizon(self):
        prices = torch.arange(9.0).reshape(1, 3, 3)
        failures = torch.tensor([[10., 20., 30.]])
        placed = torch.tensor([[[True, True, False], [True, False, False], [True, True, True]]])
        candidates = torch.tensor([[1, 3, 2]], dtype=torch.uint8)
        predicted = torch.full((1, 3), 999.)
        torch.testing.assert_close(
            candidate_step_balance_score(predicted, prices, failures, placed, candidates, 1, 3),
            torch.tensor([[1003., 999., 7.]]),
        )
        torch.testing.assert_close(
            candidate_step_balance_score(predicted, prices, failures, placed, candidates, 2, 3),
            torch.tensor([[35., 50., 8.]]),
        )
        # A deliberately short episode still charges all remaining room failures.
        torch.testing.assert_close(
            candidate_step_balance_score(predicted, prices, failures, placed, candidates, 1, 2),
            torch.tensor([[34., 50., 7.]]),
        )

    def test_proposal_averages_only_unused_concrete_rooms(self):
        prices = torch.tensor([[[0., 2., 0.], [0., 8., 0.], [0., -4., 0.]]])
        variants = torch.tensor([0, 0, 1])
        doors = torch.tensor([1, 0, 0])
        torch.testing.assert_close(
            proposal_step_prices(prices, torch.tensor([[False, False, False]]), variants, 2, doors, 1),
            torch.tensor([[-4., 5., 5.]]),
        )
        torch.testing.assert_close(
            proposal_step_prices(prices, torch.tensor([[True, False, True]]), variants, 2, doors, 1),
            torch.tensor([[0., 8., 8.]]),
        )

    def test_opposite_identity_biases_receive_opposite_price_updates(self):
        raw = torch.zeros(1, 2, 2, requires_grad=True)
        failures = torch.zeros(1, 2, requires_grad=True)
        prices = raw - raw.mean(-1, keepdim=True)
        loss = balance_family_loss(
            [balance_objective_terms(prices, failures, torch.tensor([[0, 1]]), torch.ones(1, 2, dtype=torch.bool))],
            1., 1., torch.ones(1),
        )
        loss.backward()
        # Gradient descent raises the observed price and lowers the other step.
        torch.testing.assert_close(raw.grad, torch.tensor([[[-.25, .25], [.25, -.25]]]))
        torch.testing.assert_close(failures.grad, torch.zeros_like(failures))

    def test_uniform_permutations_have_no_pressure_but_missing_rooms_do(self):
        raw = torch.zeros(2, 2, 2, requires_grad=True)
        failures = torch.zeros(2, 2, requires_grad=True)
        terms = balance_objective_terms(
            raw - raw.mean(-1, keepdim=True), failures,
            torch.tensor([[0, 1], [1, 0]]), torch.ones(2, 2, dtype=torch.bool),
        )
        balance_family_loss([terms], 1., 1., torch.ones(2)).backward()
        torch.testing.assert_close(raw.grad.sum(0), torch.zeros(2, 2))
        raw.grad = None
        failures.grad = None
        terms = balance_objective_terms(
            raw - raw.mean(-1, keepdim=True), failures,
            torch.full((2, 2), -1), torch.ones(2, 2, dtype=torch.bool),
        )
        balance_family_loss([terms], 1., 1., torch.ones(2)).backward()
        self.assertTrue((failures.grad < 0).all())

    def test_prices_center_each_concrete_room_separately(self):
        preds = example_predictions()
        preds.room_step = torch.tensor([[[4., 8.], [9., 3.]]])
        preds.room_step_failure = torch.tensor([[7., 11.]])
        tables = compute_balance_price_tables(preds, *uniform_area_targets())
        torch.testing.assert_close(tables.room_step, torch.tensor([[[-2., 2.], [3., -3.]]]))
        torch.testing.assert_close(tables.room_step_failure, preds.room_step_failure)

    def test_config_requires_explicit_valid_settings(self):
        for section, name in [('balance_train', 'step_beta'), ('balance_train', 'step_price_scale'), ('train', 'step_balance_weight')]:
            data = json.loads(Path('configs/debug.json').read_text())
            del data[section][name]
            with self.assertRaises(ValueError):
                Config.model_validate(data)
            for value in [-1., float('nan')]:
                data[section][name] = value
                with self.assertRaises(ValueError):
                    validate_config(Config.model_validate(data))

    def test_distribution_metric_corrects_sampling_bias_and_excludes_missing(self):
        rooms = torch.tensor([[0, 1], [0, 1], [1, 0], [1, 0]])
        torch.testing.assert_close(compute_room_step_ss(rooms, 2), torch.tensor(1 / 3, dtype=torch.float64))
        rooms = torch.tensor([[0, 1], [0, 1], [2, 1]])
        torch.testing.assert_close(compute_room_step_ss(rooms, 2), torch.tensor(1., dtype=torch.float64))


if __name__ == '__main__':
    unittest.main()
