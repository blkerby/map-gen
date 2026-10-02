import unittest

import torch

from train_config import VariableMixture, VariableRange, VariableSchedule
from variable_sampling import sample_variable_float


class VariableSamplingTest(unittest.TestCase):
    def test_mixture_retains_original_episodes_and_samples_a_range(self):
        torch.manual_seed(0)
        value = VariableMixture.model_validate({"mixture": [
            {"weight": 1.0, "value": 0.0},
            {"weight": 1.0, "value": {"linear": {"min": 0.1, "max": 0.3}}},
        ]})
        samples = sample_variable_float(value, "test", 4096, torch.device("cpu"))
        self.assertTrue(1800 < (samples == 0).sum() < 2300)
        nonzero = samples[samples > 0]
        self.assertTrue(torch.all((nonzero >= 0.1) & (nonzero <= 0.3)))
        self.assertAlmostEqual(nonzero.mean().item(), 0.2, delta=0.01)

    def test_scalar_and_instantiated_step_range(self):
        samples = sample_variable_float(0.0, "test", 8, torch.device("cpu"))
        self.assertEqual(samples.tolist(), [0.0] * 8)
        value = VariableSchedule(step=VariableRange(min=0.1, max=0.2))
        samples = sample_variable_float(value, "test", 128, torch.device("cpu"))
        self.assertTrue(torch.all((samples >= 0.1) & (samples <= 0.2)))


if __name__ == "__main__":
    unittest.main()
