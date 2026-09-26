#!/usr/bin/env python3
"""Download an inclusive range of experience files with one short command.

Usage (from the repository root, with the map-gen environment active):
  python scripts/download_experience.py rtx map-gen/runs/RUN 8457 8488

The remote path names the run directory, containing experience/. Files go to
this repository's runs/RUN/ unless --destination specifies a local run directory.
Use --log NAME to also refresh a named log in that run. Remote files are read
only. Existing local files are replaced only after all transfers succeed.
Each invocation requires remote-transfer approval when run through Codex.
"""

import argparse
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("host", help="SSH hostname or user@hostname")
    parser.add_argument("remote_run", help="remote run directory (not experience/)")
    parser.add_argument("first", type=int, help="first experience index, inclusive")
    parser.add_argument("last", type=int, help="last experience index, inclusive")
    parser.add_argument("--destination", type=Path, help="local run directory")
    parser.add_argument("--log", help="optional log filename within the remote run")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print transfers without accessing the remote host",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@-]*", args.host):
        parser.error("host must be an SSH hostname or user@hostname")
    remote = PurePosixPath(args.remote_run)
    if (not remote.name or ".." in remote.parts
            or any(c in args.remote_run for c in "\n\r*?[]{}")):
        parser.error("remote_run must name a literal run directory without '..' or wildcards")
    if args.first < 0 or args.last < args.first:
        parser.error("require 0 <= first <= last")
    if args.log is not None and (
        not args.log or PurePosixPath(args.log).name != args.log
        or args.log in (".", "..") or any(c in args.log for c in "\n\r*?[]{}")
    ):
        parser.error("--log must be a literal filename within the run")
    destination = (
        args.destination.resolve() if args.destination is not None
        else Path(__file__).resolve().parents[1] / "runs" / remote.name
    )
    names = [f"{i}.safetensors" for i in range(args.first, args.last + 1)]
    sources = [f"{args.host}:{remote}/experience/{name}" for name in names]
    if args.dry_run:
        print(shlex.join(["scp", *sources, str(destination / "experience") + "/"]))
        if args.log is not None:
            print(shlex.join([
                "scp", f"{args.host}:{remote}/{args.log}", str(destination) + "/",
            ]))
        return
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "experience").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".experience-download-", dir=destination) as temporary:
        staging = Path(temporary)
        experience = staging / "experience"
        experience.mkdir()
        print(f"Downloading {args.first}–{args.last} from {args.host}:{remote}", flush=True)
        subprocess.run(["scp", *sources, str(experience) + "/"], check=True)
        if args.log is not None:
            subprocess.run(
                ["scp", f"{args.host}:{remote}/{args.log}", str(staging) + "/"], check=True,
            )
        files = [(experience / name, destination / "experience" / name) for name in names]
        if args.log is not None:
            files.append((staging / args.log, destination / args.log))
        for source, target in files:
            if not source.is_file() or source.stat().st_size == 0:
                raise RuntimeError(f"missing or empty downloaded file: {source.name}")
        for source, target in files:
            source.replace(target)
    print(f"Downloaded {len(names)} experience files to {destination / 'experience'}", flush=True)


if __name__ == "__main__":
    main()
