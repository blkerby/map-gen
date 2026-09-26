import torch

import os
from pathlib import Path
import safetensors.torch
from safetensors import safe_open
from env import Actions, EpisodeData
from train_config import GENERATION_VARIABLE_FLOAT_FIELDS


EXPERIENCE_FORMAT = "map-gen-experience-v5"
PRE_AREA_CONNECTIONS_EXPERIENCE_FORMAT = "map-gen-experience-v4"
PRE_SUCCESS_EXPERIENCE_FORMAT = "map-gen-experience-v3"
EXPERIENCE_MISSING_VARIABLE_FIELDS = {
    EXPERIENCE_FORMAT: (),
    PRE_AREA_CONNECTIONS_EXPERIENCE_FORMAT: ("reward_area_distinct_crossing",),
    PRE_SUCCESS_EXPERIENCE_FORMAT: ("reward_success", "reward_area_distinct_crossing"),
}
REQUIRED_BALANCE_EXPERIENCE_TENSORS = (
    "room_idx",
    "room_x",
    "room_y",
    "room_area",
    "generation_variable_floats",
)
REQUIRED_EXPERIENCE_TENSORS = (
    *REQUIRED_BALANCE_EXPERIENCE_TENSORS,
    "temperature",
    "recommended_candidates",
)


def read_experience_tensors(
    path: str | Path, required_names: tuple[str, ...]
) -> dict[str, torch.Tensor]:
    with safe_open(path, framework="pt", device="cpu") as experience:
        metadata = experience.metadata()
        if (
            metadata is None
            or metadata.get("format") not in EXPERIENCE_MISSING_VARIABLE_FIELDS
        ):
            raise ValueError(f"unsupported experience format in {path}")
        missing = [
            name for name in required_names if name not in experience.keys()
        ]
        if missing:
            raise ValueError(f"{path} missing tensor(s): {', '.join(missing)}")
        tensors = {
            name: experience.get_tensor(name) for name in required_names
        }
    variables = tensors["generation_variable_floats"]
    missing_fields = EXPERIENCE_MISSING_VARIABLE_FIELDS[metadata["format"]]
    expected_shape = (
        tensors["room_idx"].shape[0],
        len(GENERATION_VARIABLE_FLOAT_FIELDS) - len(missing_fields),
    )
    if variables.shape != expected_shape:
        raise ValueError(
            f"{path} generation_variable_floats shape must be {expected_shape}, "
            f"got {tuple(variables.shape)}"
        )
    # Older formats predate these rewards. Insert neutral columns in schema order
    # so that every recorded reward and constraint keeps its original meaning.
    missing_columns = sorted(
        GENERATION_VARIABLE_FLOAT_FIELDS.index(name) for name in missing_fields
    )
    for column in missing_columns:
        variables = torch.cat(
            (
                variables[:, :column],
                variables.new_zeros((variables.shape[0], 1)),
                variables[:, column:],
            ),
            dim=1,
        )
    tensors["generation_variable_floats"] = variables
    return tensors


def load_balance_experience(path: str | Path, num_rooms: int) -> tuple[Actions, torch.Tensor]:
    tensors = read_experience_tensors(path, REQUIRED_BALANCE_EXPERIENCE_TENSORS)

    action_shape = tensors["room_idx"].shape
    if len(action_shape) != 2 or action_shape[1] != num_rooms:
        raise ValueError(
            f"{path} action shape must be (episodes, {num_rooms}), got {tuple(action_shape)}"
        )
    for name in ("room_x", "room_y", "room_area"):
        if tensors[name].shape != action_shape:
            raise ValueError(
                f"{path} {name} shape {tuple(tensors[name].shape)} does not match "
                f"room_idx shape {tuple(action_shape)}"
            )
    return (
        Actions(
            room_idx=tensors["room_idx"],
            room_x=tensors["room_x"],
            room_y=tensors["room_y"],
            room_area=tensors["room_area"],
        ),
        tensors["generation_variable_floats"],
    )


class ExperienceStorage:
    def __init__(self, num_rooms, data_path, episodes_per_file):
        self.num_rooms = num_rooms
        self.data_path = data_path
        self.episodes_per_file = episodes_per_file
        self.num_files = 0
        os.makedirs(data_path, exist_ok=True)

    def store(self, episode_data: EpisodeData):
        next_file_number = self.num_files
        assert episode_data.actions.room_idx.shape[0] == self.episodes_per_file
        assert episode_data.temperature.shape[0] == self.episodes_per_file
        assert episode_data.recommended_candidates.shape[0] == self.episodes_per_file
        assert episode_data.generation_variable_floats.shape[0] == self.episodes_per_file
        file_path = os.path.join(self.data_path, "{}.safetensors".format(next_file_number))
        safetensors.torch.save_file(
            {
                "room_idx": episode_data.actions.room_idx,
                "room_x": episode_data.actions.room_x,
                "room_y": episode_data.actions.room_y,
                "room_area": episode_data.actions.room_area,
                "temperature": episode_data.temperature,
                "recommended_candidates": episode_data.recommended_candidates,
                "generation_variable_floats": episode_data.generation_variable_floats,
            },
            file_path,
            metadata={"format": EXPERIENCE_FORMAT},
        )
        self.num_files += 1

    def read_files(self, file_num_list, episodes_per_file):
        data_list = []
        for file_num in file_num_list:
            file_path = os.path.join(self.data_path, "{}.safetensors".format(file_num))
            tensors = read_experience_tensors(file_path, REQUIRED_EXPERIENCE_TENSORS)
            data = EpisodeData(
                actions=Actions(
                    room_idx=tensors["room_idx"],
                    room_x=tensors["room_x"],
                    room_y=tensors["room_y"],
                    room_area=tensors["room_area"],
                ),
                temperature=tensors["temperature"],
                recommended_candidates=tensors["recommended_candidates"],
                generation_variable_floats=tensors["generation_variable_floats"],
            )
            ind = torch.randperm(data.actions.room_idx.shape[0])[:episodes_per_file]
            data = EpisodeData(
                actions=Actions(
                    room_idx=data.actions.room_idx[ind],
                    room_x=data.actions.room_x[ind],
                    room_y=data.actions.room_y[ind],
                    room_area=data.actions.room_area[ind],
                ),
                temperature=data.temperature[ind],
                recommended_candidates=data.recommended_candidates[ind],
                generation_variable_floats=data.generation_variable_floats[ind],
            )
            data_list.append(data)

        return EpisodeData(
            actions=Actions(
                room_idx=torch.cat([data.actions.room_idx for data in data_list], dim=0),
                room_x=torch.cat([data.actions.room_x for data in data_list], dim=0),
                room_y=torch.cat([data.actions.room_y for data in data_list], dim=0),
                room_area=torch.cat([data.actions.room_area for data in data_list], dim=0),
            ),
            temperature=torch.cat([data.temperature for data in data_list], dim=0),
            recommended_candidates=torch.cat(
                [data.recommended_candidates for data in data_list], dim=0
            ),
            generation_variable_floats=torch.cat(
                [data.generation_variable_floats for data in data_list], dim=0
            ),
        )

    def read_balance_files(self, file_num_list: list[int]) -> tuple[Actions, torch.Tensor]:
        data = [
            load_balance_experience(Path(self.data_path) / f"{file_num}.safetensors", self.num_rooms)
            for file_num in file_num_list
        ]
        return (
            Actions(
                room_idx=torch.cat([actions.room_idx for actions, _ in data]),
                room_x=torch.cat([actions.room_x for actions, _ in data]),
                room_y=torch.cat([actions.room_y for actions, _ in data]),
                room_area=torch.cat([actions.room_area for actions, _ in data]),
            ),
            torch.cat([variables for _, variables in data]),
        )

    def sample(self, batch_size, episodes_per_file, hist_c) -> EpisodeData:
        n = batch_size
        episodes_per_file = min(episodes_per_file, self.episodes_per_file)
        num_files = (n + episodes_per_file - 1) // episodes_per_file

        t = torch.pow(torch.rand([num_files]), 1 / (1 + hist_c))
        file_num_list = (
            torch.floor(t * self.num_files).to(torch.int64).clamp_max(self.num_files - 1).tolist()
        )

        return self.read_files(file_num_list, episodes_per_file).slice(0, n)
