#!/usr/bin/env python3
"""Summarize hybrid MPI+OpenMP benchmark CSV files."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def fmt_float(value: float) -> str:
    return f"{value:.6f}"


def fmt_speedup(value: float) -> str:
    return f"{value:.3f}"


def fmt_sci(value: float) -> str:
    return f"{value:.6e}"


def median(values: list[float]) -> float:
    return statistics.median(values)


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        ranks = int(row["ranks"])
        threads = int(row["threads"])
        groups[(ranks, threads)].append(row)

    if not groups:
        raise ValueError("input CSV does not contain benchmark rows")

    base_key = min(groups, key=lambda key: (key[0] * key[1], key[0], key[1]))
    base_total = median([float(row["total_seconds"]) for row in groups[base_key]])
    base_force = median([float(row["force_seconds"]) for row in groups[base_key]])

    summary = []
    for ranks, threads in sorted(groups, key=lambda key: (key[0] * key[1], key[0], key[1])):
        group = groups[(ranks, threads)]
        workers = ranks * threads
        totals = [float(row["total_seconds"]) for row in group]
        forces = [float(row["force_seconds"]) for row in group]
        communications = [float(row["communication_seconds"]) for row in group]
        integrations = [float(row["integration_seconds"]) for row in group]
        energies = [float(row["energy_seconds"]) for row in group]
        initial_accelerations = [
            float(row["initial_acceleration_seconds"]) for row in group
        ]
        total_median = median(totals)
        force_median = median(forces)
        total_speedup = base_total / total_median
        force_speedup = base_force / force_median
        max_drift = max(float(row["max_relative_energy_drift"]) for row in group)
        statuses = "|".join(sorted({row["status"] for row in group}))

        summary.append(
            {
                "ranks": ranks,
                "threads": threads,
                "total_workers": workers,
                "repeats": len(group),
                "total_median_s": total_median,
                "total_stdev_s": stdev(totals),
                "force_median_s": force_median,
                "force_stdev_s": stdev(forces),
                "communication_median_s": median(communications),
                "integration_median_s": median(integrations),
                "energy_median_s": median(energies),
                "initial_acceleration_median_s": median(initial_accelerations),
                "total_speedup": total_speedup,
                "total_efficiency": total_speedup / workers,
                "force_speedup": force_speedup,
                "force_efficiency": force_speedup / workers,
                "max_relative_energy_drift": max_drift,
                "statuses": statuses,
                "baseline": "yes" if (ranks, threads) == base_key else "no",
            }
        )

    return summary


def markdown_table(summary: list[dict[str, object]]) -> str:
    lines = [
        "| Ranks | Threads | Workers | Repeats | Total median s | Total stdev s | Force median s | Comm median s | Energy median s | Total speedup | Total efficiency | Force speedup | Max drift | Status |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|",
    ]

    for row in summary:
        lines.append(
            "| {ranks} | {threads} | {workers} | {repeats} | {total} | {total_std} | {force} | {comm} | {energy} | {total_speedup} | {total_efficiency} | {force_speedup} | {drift} | {statuses} |".format(
                ranks=row["ranks"],
                threads=row["threads"],
                workers=row["total_workers"],
                repeats=row["repeats"],
                total=fmt_float(float(row["total_median_s"])),
                total_std=fmt_float(float(row["total_stdev_s"])),
                force=fmt_float(float(row["force_median_s"])),
                comm=fmt_float(float(row["communication_median_s"])),
                energy=fmt_float(float(row["energy_median_s"])),
                total_speedup=fmt_speedup(float(row["total_speedup"])),
                total_efficiency=fmt_speedup(float(row["total_efficiency"])),
                force_speedup=fmt_speedup(float(row["force_speedup"])),
                drift=fmt_sci(float(row["max_relative_energy_drift"])),
                statuses=row["statuses"],
            )
        )

    return "\n".join(lines)


def write_summary_csv(path: Path, summary: list[dict[str, object]]) -> None:
    fieldnames = [
        "ranks",
        "threads",
        "total_workers",
        "repeats",
        "total_median_s",
        "total_stdev_s",
        "force_median_s",
        "force_stdev_s",
        "communication_median_s",
        "integration_median_s",
        "energy_median_s",
        "initial_acceleration_median_s",
        "total_speedup",
        "total_efficiency",
        "force_speedup",
        "force_efficiency",
        "max_relative_energy_drift",
        "statuses",
        "baseline",
    ]

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize hybrid MPI+OpenMP benchmark CSV files."
    )
    parser.add_argument("input_csv", type=Path, nargs="+", help="hybrid benchmark CSV file(s)")
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
