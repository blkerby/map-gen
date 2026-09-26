from pathlib import Path
import tempfile
import unittest

import safetensors.torch
import torch

from experience import (
    EXPERIENCE_FORMAT,
    PRE_AREA_CONNECTIONS_EXPERIENCE_FORMAT,
    PRE_SUCCESS_EXPERIENCE_FORMAT,
    ExperienceStorage,
    load_balance_experience,
)
from train_config import GENERATION_VARIABLE_FLOAT_FIELDS


class ReplaySuccessCompatibilityTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.column = GENERATION_VARIABLE_FLOAT_FIELDS.index("reward_area_distinct_crossing")
        self.variables = torch.arange(len(GENERATION_VARIABLE_FLOAT_FIELDS)).float().unsqueeze(0)
        self.variables[:, self.column] = 0
        self.tensors = {
            "room_idx": torch.zeros((1, 2), dtype=torch.uint8),
            "room_x": torch.zeros((1, 2), dtype=torch.uint8),
            "room_y": torch.zeros((1, 2), dtype=torch.uint8),
            "room_area": torch.zeros((1, 2), dtype=torch.uint8),
            "temperature": torch.ones(1),
            "recommended_candidates": torch.ones(1, dtype=torch.int64),
            "generation_variable_floats": self.variables,
        }
        self.storage = ExperienceStorage(2, self.directory, 1)

    def write_experience(self, index: int, variables: torch.Tensor, version: str) -> Path:
        path = self.directory / f"{index}.safetensors"
        safetensors.torch.save_file(
            {**self.tensors, "generation_variable_floats": variables.contiguous()},
            path,
            metadata={"format": version},
        )
        return path

    def test_v3_v4_and_v5_files_preserve_all_other_columns(self) -> None:
        old_variables = torch.cat(
            (self.variables[:, :self.column], self.variables[:, self.column + 1:]), dim=1
        )
        legacy_path = self.write_experience(0, old_variables, PRE_AREA_CONNECTIONS_EXPERIENCE_FORMAT)
        original_bytes = legacy_path.read_bytes()
        current_variables = self.variables.clone()
        current_variables[:, self.column] = 2.5
        self.write_experience(1, current_variables, EXPERIENCE_FORMAT)
        success_column = GENERATION_VARIABLE_FLOAT_FIELDS.index("reward_success")
        v3_variables = self.variables.clone()
        v3_variables[:, success_column] = 0
        v3_columns = [
            index for index in range(len(GENERATION_VARIABLE_FLOAT_FIELDS))
            if index not in (self.column, success_column)
        ]
        v3_path = self.write_experience(2, v3_variables[:, v3_columns], PRE_SUCCESS_EXPERIENCE_FORMAT)
        v3_original_bytes = v3_path.read_bytes()
        expected = torch.cat((self.variables, current_variables, v3_variables))
        replay = self.storage.read_files([0, 1, 2], episodes_per_file=1)
        torch.testing.assert_close(replay.generation_variable_floats, expected, rtol=0, atol=0)
        _, balance_variables = self.storage.read_balance_files([0, 1, 2])
        torch.testing.assert_close(balance_variables, expected, rtol=0, atol=0)
        assert legacy_path.read_bytes() == original_bytes
        assert v3_path.read_bytes() == v3_original_bytes

    def test_wrong_schema_width_and_unsupported_versions_are_rejected(self) -> None:
        for version, width in (
            (PRE_SUCCESS_EXPERIENCE_FORMAT, len(GENERATION_VARIABLE_FLOAT_FIELDS) - 1),
            (PRE_SUCCESS_EXPERIENCE_FORMAT, len(GENERATION_VARIABLE_FLOAT_FIELDS) - 3),
            (PRE_AREA_CONNECTIONS_EXPERIENCE_FORMAT, len(GENERATION_VARIABLE_FLOAT_FIELDS)),
            (PRE_AREA_CONNECTIONS_EXPERIENCE_FORMAT, len(GENERATION_VARIABLE_FLOAT_FIELDS) - 2),
            (EXPERIENCE_FORMAT, len(GENERATION_VARIABLE_FLOAT_FIELDS) - 1),
            ("map-gen-experience-v2", len(GENERATION_VARIABLE_FLOAT_FIELDS) - 1),
        ):
            with self.subTest(version=version, width=width):
                path = self.write_experience(0, self.variables[:, :width], version)
                with self.assertRaises(ValueError):
                    load_balance_experience(path, num_rooms=2)
                with self.assertRaises(ValueError):
                    self.storage.read_files([0], episodes_per_file=1)


if __name__ == "__main__":
    unittest.main()
