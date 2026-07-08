import tempfile
from pathlib import Path

from PIL import Image

from visualize import save_episode_frames


def test_save_episode_frames_uses_room_area_colors() -> None:
    rooms = [
        {
            "name": "A",
            "map": [[1]],
            "doors": [],
            "connections": [],
            "missing_connections": [],
            "toilet_crossing_x": [],
        }
    ]
    actions = (
        [[0]],
        [[0]],
        [[0]],
        [[2]],
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        paths = save_episode_frames(
            rooms,
            actions,
            Path(temp_dir),
            map_size=(1, 1),
            environment_index=0,
        )
        image = Image.open(paths[0])

    assert image.getpixel((42, 42)) == (208, 0, 0)


def main() -> None:
    test_save_episode_frames_uses_room_area_colors()


if __name__ == "__main__":
    main()
