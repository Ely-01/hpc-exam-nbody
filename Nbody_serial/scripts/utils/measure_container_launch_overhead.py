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


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_mode: dict[str, list[float]] = {}
    for row in rows:
        by_mode.setdefault(str(row["mode"]), []).append(float(row["seconds"]))

    native_median = statistics.median(by_mode.get("native_true", [0.0]))
    summary = []
    for mode in sorted(by_mode):
        values = by_mode[mode]
        median = statistics.median(values)
        overhead = median - native_median if mode != "native_true" else 0.0
        summary.append(
            {
                "mode": mode,
                "repeats": len(values),
                "median_seconds": median,
                "stdev_seconds": stdev(values),
                "min_seconds": min(values),
                "max_seconds": max(values),
                "overhead_vs_native_seconds": overhead,
            }
        )
    return summary


def write_summary_csv(path: Path, summary: list[dict[str, object]]) -> None:
    fieldnames = [
        "mode",
        "repeats",
        "median_seconds",
        "stdev_seconds",
        "min_seconds",
        "max_seconds",
        "overhead_vs_native_seconds",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def write_summary_markdown(path: Path, summary: list[dict[str, object]]) -> None:
    lines = [
        "| Mode | Repeats | Median s | Stdev s | Min s | Max s | Overhead vs native s |",
        "|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {mode} | {repeats} | {median:.6f} | {stdev:.6f} | {min_s:.6f} | {max_s:.6f} | {overhead:.6f} |".format(
                mode=row["mode"],
                repeats=row["repeats"],
                median=float(row["median_seconds"]),
                stdev=float(row["stdev_seconds"]),
                min_s=float(row["min_seconds"]),
                max_s=float(row["max_seconds"]),
                overhead=float(row["overhead_vs_native_seconds"]),
            )
        )
    lines.extend(
        [
            "",
            "The container launch overhead is the fixed one-process startup cost measured with `singularity exec image true`; it is reported separately from long N-body timings because it is amortized by production runs.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure container launch overhead using apptainer exec true."
    )
    parser.add_argument("--image", default="container/nbody_latest.sif")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("results/container_launch_overhead.csv"))
    parser.add_argument("--apptainer", default="apptainer")
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--summary-markdown", type=Path)
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

    summary = summarize(rows)
    if args.summary_csv is not None:
        write_summary_csv(args.summary_csv, summary)
    if args.summary_markdown is not None:
        write_summary_markdown(args.summary_markdown, summary)

    print(f"# wrote {args.output}")
    if args.summary_csv is not None:
        print(f"# wrote {args.summary_csv}")
    if args.summary_markdown is not None:
        print(f"# wrote {args.summary_markdown}")
    for row in summary:
        print(
            "{mode}: median={median:.6f}s sigma={sigma:.6f}s overhead={overhead:.6f}s".format(
                mode=row["mode"],
                median=float(row["median_seconds"]),
                sigma=float(row["stdev_seconds"]),
                overhead=float(row["overhead_vs_native_seconds"]),
            )
        )


if __name__ == "__main__":
    main()
