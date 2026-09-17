#!/usr/bin/env python3
# Benchmark a running serve.py instance with repeated full-size generate
# requests (the same payload as serve.py's warmup requests), printing
# per-request wall time and, when the server was started with --profile,
# a per-phase timing summary of the final response.
import argparse
import json
import time
import urllib.request

PROFILE_SUMMARY_ROWS = 12


def generate_request_payload() -> dict:
    return {
        "episode_length": 253,
        "recommended_candidates": 4,
        "shortlist_candidates": 16,
        "temperature": 0.03,
        "proposal_temperature": 0.3,
        "reward_door": 1.0,
        "reward_connection": 1.0,
        "reward_toilet": 1.0,
        "reward_phantoon": 1.0,
        "reward_balance": 0.1,
        "reward_toilet_balance": 0.1,
        "reward_frontier": 0.0,
        "reward_graph_diameter": 0.1,
        "reward_save_distance": 0.1,
        "reward_refill_distance": 0.1,
        "reward_missing_connect_utility": 0.5,
        "area_assignment_base_order": "random",
        "small_map": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a running serve.py with repeated generate requests.",
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:5000/generate",
        help="generate endpoint URL (default: http://127.0.0.1:5000/generate)",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=5,
        help="number of requests to send (default: 5)",
    )
    parser.add_argument(
        "--save-last-response",
        type=argparse.FileType("w"),
        help="write the final response JSON to this file",
    )
    return parser.parse_args()


def send_request(url: str, payload_bytes: bytes) -> dict:
    request = urllib.request.Request(
        url,
        data=payload_bytes,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def print_profile_summary(profile: list) -> None:
    rows = sorted(profile, key=lambda row: -row[2])[:PROFILE_SUMMARY_ROWS]
    print(f"\n{'phase':52s} {'total s':>8s} {'n':>6s} {'ms/call':>8s}")
    for name, count, nanos in rows:
        per_call = f"{nanos / count / 1e6:8.2f}" if count > 1 else f"{'-':>8s}"
        print(f"{name:52s} {nanos / 1e9:8.3f} {count:6d} {per_call}")


def main() -> None:
    args = parse_args()
    payload_bytes = json.dumps(generate_request_payload()).encode()
    wall_times = []
    response = None
    for request_idx in range(args.requests):
        start = time.perf_counter()
        response = send_request(args.url, payload_bytes)
        elapsed = time.perf_counter() - start
        wall_times.append(elapsed)
        print(f"request {request_idx + 1}/{args.requests}: {elapsed:.2f}s stats={response['stats']}")
    print(f"mean: {sum(wall_times) / len(wall_times):.2f}s min: {min(wall_times):.2f}s")
    if response is None:
        raise RuntimeError("no requests were sent")
    if "profile" in response:
        print_profile_summary(response["profile"])
    if args.save_last_response is not None:
        json.dump(response, args.save_last_response)


if __name__ == "__main__":
    main()
