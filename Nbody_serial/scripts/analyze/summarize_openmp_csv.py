#!/usr/bin/env python3
"""Summarize OpenMP benchmark CSV files produced by benchmark_openmp.sh."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def median(values: list[float]) -> float:
    return statistics.median(values)


def fmt_float(value: float) -> str:
    return f"{value:.6f}"


def fmt_speedup(value: float) -> str:
    return f"{value:.3f}"


def fmt_sci(value: float) -> str:
    return f"{value:.6e}"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[int, list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        groups[int(row["threads"])].append(row)

    if 1 not in groups:
        raise ValueError("input CSV must contain a threads=1 baseline")

    base_total = median([float(row["total_seconds"]) for row in groups[1]])
    base_force = median([float(row["force_seconds"]) for row in groups[1]])

    summary = []
    for threads in sorted(groups):
        group = groups[threads]
        total = median([float(row["total_seconds"]) for row in group])
        force = median([float(row["force_seconds"]) for row in group])
        integration = median([float(row["integration_seconds"]) for row in group])
        energy = median([float(row["energy_seconds"]) for row in group])
        initial_acceleration = median(
            [float(row["initial_acceleration_seconds"]) for row in group]
        )
        max_drift = max(float(row["max_relative_energy_drift"]) for row in group)
        statuses = "|".join(sorted({row["status"] for row in group}))
        total_speedup = base_total / total
        force_speedup = base_force / force

        summary.append(
            {
                "threads": threads,
                "repeats": len(group),
                "total_median_s": total,
                "force_median_s": force,
                "integration_median_s": integration,
                "energy_median_s": energy,
                "initial_acceleration_median_s": initial_acceleration,
                "total_speedup": total_speedup,
                "total_efficiency": total_speedup / threads,
                "force_speedup": force_speedup,
                "force_efficiency": force_speedup / threads,
                "max_relative_energy_drift": max_drift,
                "statuses": statuses,
            }
        )

    return summary


def markdown_table(summary: list[dict[str, object]]) -> str:
    lines = [
        "| Threads | Repeats | Total median s | Force median s | Energy median s | Total speedup | Total efficiency | Force speedup | Force efficiency | Max drift | Status |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|",
    ]

    for row in summary:
        lines.append(
            "| {threads} | {repeats} | {total} | {force} | {energy} | {total_speedup} | {total_efficiency} | {force_speedup} | {force_efficiency} | {drift} | {statuses} |".format(
                threads=row["threads"],
                repeats=row["repeats"],
                total=fmt_float(float(row["total_median_s"])),
                force=fmt_float(float(row["force_median_s"])),
                energy=fmt_float(float(row["energy_median_s"])),
                total_speedup=fmt_speedup(float(row["total_speedup"])),
                total_efficiency=fmt_speedup(float(row["total_efficiency"])),
                force_speedup=fmt_speedup(float(row["force_speedup"])),
                force_efficiency=fmt_speedup(float(row["force_efficiency"])),
                drift=fmt_sci(float(row["max_relative_energy_drift"])),
                statuses=row["statuses"],
            )
        )

    return "\n".join(lines)


def write_summary_csv(path: Path, summary: list[dict[str, object]]) -> None:
    fieldnames = [
        "threads",
        "repeats",
        "total_median_s",
        "force_median_s",
        "integration_median_s",
        "energy_median_s",
        "initial_acceleration_median_s",
        "total_speedup",
        "total_efficiency",
        "force_speedup",
        "force_efficiency",
        "max_relative_energy_drift",
        "statuses",
    ]

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize OpenMP benchmark CSV files."
    )
    parser.add_argument("input_csv", type=Path, help="benchmark CSV to summarize")
    parser.add_argument(
        "--markdown",
        type=Path,
        help="optional path where the Markdown table will be written",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="optional path where the summary CSV will be written",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input_csv)
    summary = summarize(rows)
    table = markdown_table(summary)

    print(table)

    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(table + "\n")

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        write_summary_csv(args.csv, summary)


if __name__ == "__main__":
    main()
