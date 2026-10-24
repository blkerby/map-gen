#!/usr/bin/env python3
import json
import urllib.error
import urllib.request
import sys

URL = "http://127.0.0.1:5000/generate"
AREA_COUNT = 6


def area_room_counts(area_assignments: list[list[int]]) -> list[list[int]]:
    return [
        [sum(1 for area in map_areas if area == area_id) for area_id in range(AREA_COUNT)]
        for map_areas in area_assignments
    ]


def main() -> int:
    payload = {
        "max_num_maps": 1,
        "episode_length": 253,
        "recommended_candidates": 8,
        "shortlist_candidates": 1024,
        "num_scored_invalid_candidates": 0,
        "max_candidate_areas_per_placement": 2,
        "recommended_candidates_same_frontier": False,
        "guarantee_phantoon_counterpart": True,
        "temperature": 0.04,
        "proposal_temperature": 0.03,
        "reward_door": 1.0,
        "reward_connection": 1.0,
        "reward_toilet": 1.0,
        "reward_phantoon_pair": 1.0,
        "reward_phantoon_area": 1.0,
        "reward_success": 0.0,
        "reward_initial_item": 0.0,
        "initial_item_start": None,
        "initial_item_difficulty": "Basic",
        "initial_item_wall_jump": True,
        "reward_ship_in_crateria": 1.0,
        "reward_kraid_in_brinstar": 1.0,
        "reward_ridley_in_norfair": 1.0,
        "reward_phantoon_in_wrecked_ship": 1.0,
        "reward_draygon_in_maridia": 1.0,
        "reward_mother_brain_in_tourian": 1.0,
        "force_ship_in_crateria": True,
        "force_kraid_in_brinstar": True,
        "force_ridley_in_norfair": True,
        "force_phantoon_in_wrecked_ship": True,
        "force_draygon_in_maridia": True,
        "force_mother_brain_in_tourian": True,
        "maridia_water_preferred_probability_1": 0.3,
        "maridia_water_preferred_probability_2": 0.45,
        "maridia_water_preferred_probability_3": 0.6,
        "norfair_heat_preferred_probability_1": 0.3,
        "norfair_heat_preferred_probability_2": 0.45,
        "norfair_heat_preferred_probability_3": 0.6,
        "reward_frontier": 0.0,
        "reward_graph_diameter": 0.1,
        "reward_save_distance": 0.1,
        "reward_refill_distance": 0.1,
        "reward_missing_connect_utility": 1.0,
        "max_return_distance": None,
        "reward_area_distinct_crossing": 1.0,
        "reward_area_crossing": 0.0,
        "reward_area_size_valid": 1.0,
        "reward_area_map_station": 1.0,
        "reward_area_x": 5.0,
        "reward_area_y": 5.0,
        "target_area_rooms": [150, 300, 300, 100, 200, 100],
        "target_area_x": [16, 10, 40, 48, 40, 16],
        "target_area_y": [12, 32, 64, 12, 32, 24],
        "small_map": False,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        print(error.read().decode("utf-8"))
        return 1
    print(response_body)

    response_data = json.loads(response_body)
    print(response_data["stats"], file=sys.stderr)
    # print(json.dumps(area_room_counts(response_data["area"])))
    # for key in ["num_generated", "num_pre_valid", "num_valid"]:
    #     print(key, response_data[key])
    # print(json.dumps(response_data["area_crossings"]))
    # print(response_data["avg_area_crossings"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
