#!/usr/bin/env python3
"""Measure Apptainer/Singularity launch overhead with repeated true commands."""

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import time
from pathlib import Path


def run_once(command: list[str]) -> tuple[float, int]:
    start = time.perf_counter()
    completed = subprocess.run(command, check=False)
    elapsed = time.perf_counter() - start
    return elapsed, completed.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure container launch overhead using apptainer exec true."
    )
    parser.add_argument("--image", default="container/nbody_latest.sif")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results/container_launch_overhead.csv"))
    parser.add_argument("--apptainer", default="apptainer")
    args = parser.parse_args()

    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")

    native_command = ["true"]
    container_command = [args.apptainer, "exec", args.image, "true"]
    rows = []

    for label, command in [
        ("native_true", native_command),
        ("container_true", container_command),
    ]:
        for repeat in range(1, args.repeats + 1):
            elapsed, returncode = run_once(command)
            rows.append(
                {
                    "mode": label,
                    "repeat": repeat,
                    "seconds": elapsed,
                    "returncode": returncode,
                }
            )
            if returncode != 0:
                raise SystemExit(f"{label} failed with return code {returncode}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mode", "repeat", "seconds", "returncode"])
        writer.writeheader()
        writer.writerows(rows)

    by_mode: dict[str, list[float]] = {}
    for row in rows:
        by_mode.setdefault(str(row["mode"]), []).append(float(row["seconds"]))

    print(f"# wrote {args.output}")
    for mode, values in by_mode.items():
        sigma = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f"{mode}: median={statistics.median(values):.6f}s sigma={sigma:.6f}s")


if __name__ == "__main__":
    main()
