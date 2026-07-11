import unittest

import torch

from learn import verify_feature_tensor


class FeatureConsistencyTest(unittest.TestCase):
    def test_exact_match(self) -> None:
        values = torch.tensor([[1, 2], [3, 4]], dtype=torch.int16)
        self.assertIsNone(
            verify_feature_tensor(values, values.clone(), "features.example", 7)
        )

    def test_mismatch_reports_location_and_values(self) -> None:
        generated = torch.tensor([[1, 2], [3, 4]], dtype=torch.int16)
        replayed = generated.clone()
        replayed[1, 0] = 9
        mismatch = verify_feature_tensor(generated, replayed, "features.example", 7)
        self.assertIsNotNone(mismatch)
        assert mismatch is not None
        self.assertEqual(mismatch.path, "features.example")
        self.assertEqual(mismatch.step, 7)
        self.assertEqual(mismatch.mismatched_values, 1)
        self.assertIn("index=(1, 0) generated=3 replayed=9", mismatch.example)

    def test_shape_mismatch_reports_both_shapes(self) -> None:
        mismatch = verify_feature_tensor(
            torch.zeros([2]),
            torch.zeros([1, 2]),
            "features.example",
            3,
        )
        self.assertIsNotNone(mismatch)
        assert mismatch is not None
        self.assertEqual(mismatch.generated_shape, (2,))
        self.assertEqual(mismatch.replayed_shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
