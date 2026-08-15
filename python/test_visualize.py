import tempfile
from pathlib import Path

from PIL import Image

from visualize import _room_geometries, save_episode_frames


def test_boss_rooms_use_x_marker() -> None:
    boss_types = (
        "kraid_boss",
        "phantoon_boss",
        "draygon_boss",
        "ridley_boss",
        "mother_brain",
    )
    rooms = [
        {
            "name": boss_type,
            "special_type": boss_type,
            "map": [[1]],
            "doors": [],
            "connections": [],
            "missing_connections": [],
            "toilet_crossing_x": [],
        }
        for boss_type in boss_types
    ]

    assert [geometry["special_marker"] for geometry in _room_geometries(rooms)] == [
        "X"
    ] * 5


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


def test_save_episode_frames_styles_heat_and_water_rooms() -> None:
    rooms = [
        {
            "name": "Heat",
            "map": [[1]],
            "heat": 3,
            "doors": [],
            "connections": [],
            "missing_connections": [],
            "toilet_crossing_x": [],
        },
        {
            "name": "Water",
            "map": [[1]],
            "water": 1,
            "doors": [],
            "connections": [],
            "missing_connections": [],
            "toilet_crossing_x": [],
        },
    ]
    actions = ([[0, 1]], [[0, 1]], [[0, 0]], [[3, 4]])

    with tempfile.TemporaryDirectory() as temp_dir:
        paths = save_episode_frames(
            rooms,
            actions,
            Path(temp_dir),
            map_size=(2, 1),
            environment_index=0,
        )
        image = Image.open(paths[-1])

    heat_pixel = image.getpixel((42, 42))
    water_pixels = {image.getpixel((52, 40)), image.getpixel((55, 40))}
    assert heat_pixel[0] > 181 and heat_pixel[1] > 155 and heat_pixel[2] > 0
    assert len(water_pixels) == 2
    assert all(sum(pixel) < sum((34, 139, 230)) for pixel in water_pixels)


def main() -> None:
    test_boss_rooms_use_x_marker()
    test_save_episode_frames_uses_room_area_colors()
    test_save_episode_frames_styles_heat_and_water_rooms()


if __name__ == "__main__":
    main()
