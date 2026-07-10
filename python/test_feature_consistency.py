import unittest

import torch

from learn import verify_feature_tensor


class FeatureConsistencyTest(unittest.TestCase):
    def test_exact_match(self) -> None:
        values = torch.tensor([[1, 2], [3, 4]], dtype=torch.int16)
        verify_feature_tensor(values, values.clone(), "features.example", 7)

    def test_mismatch_reports_location_and_values(self) -> None:
        generated = torch.tensor([[1, 2], [3, 4]], dtype=torch.int16)
        replayed = generated.clone()
        replayed[1, 0] = 9
        with self.assertRaisesRegex(
            RuntimeError,
            r"step 7.*features\.example\(1, 0\).*generated=3, replayed=9",
        ):
            verify_feature_tensor(generated, replayed, "features.example", 7)

    def test_shape_mismatch_reports_both_shapes(self) -> None:
        with self.assertRaisesRegex(RuntimeError, r"shape=\(2,\).*shape=\(1, 2\)"):
            verify_feature_tensor(
                torch.zeros([2]),
                torch.zeros([1, 2]),
                "features.example",
                3,
            )


if __name__ == "__main__":
    unittest.main()
