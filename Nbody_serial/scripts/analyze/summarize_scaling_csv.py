#!/usr/bin/env python3
"""Summarize strong/weak scaling CSV files produced by benchmark_scaling.sh."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def median(values: list[float]) -> float:
    return statistics.median(values)


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def fmt_float(value: float) -> str:
    return f"{value:.6f}"


def fmt_ratio(value: float) -> str:
    return f"{value:.3f}"


def fmt_sci(value: float) -> str:
    return f"{value:.6e}"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        backend = row.get("backend") or "native"
        ranks = int(row["ranks"])
        threads = int(row["threads"])
        groups[(backend, ranks, threads)].append(row)

    if not groups:
        raise ValueError("input CSV does not contain benchmark rows")

    modes = {row["mode"] for row in rows}
    if len(modes) != 1:
        raise ValueError(f"input CSV mixes multiple modes: {sorted(modes)}")
    mode = next(iter(modes))
    if mode not in {"strong", "weak"}:
        raise ValueError(f"unknown scaling mode: {mode}")

    base_by_backend = {}
    for backend in sorted({key[0] for key in groups}):
        backend_keys = [key for key in groups if key[0] == backend]
        base_key = min(
            backend_keys,
            key=lambda key: (key[1] * key[2], key[1], key[2]),
        )
        _, base_ranks, base_threads = base_key
        base_workers = base_ranks * base_threads
        base_total = median(
            [float(row["total_seconds"]) for row in groups[base_key]]
        )
        base_force = median(
            [float(row["force_seconds"]) for row in groups[base_key]]
        )
        base_by_backend[backend] = {
            "key": base_key,
            "ranks": base_ranks,
            "threads": base_threads,
            "workers": base_workers,
            "total": base_total,
            "force": base_force,
        }

    summary = []
    for backend, ranks, threads in sorted(
        groups, key=lambda key: (key[0], key[1] * key[2], key[1], key[2])
    ):
        group = groups[(backend, ranks, threads)]
        workers = ranks * threads
        base = base_by_backend[backend]
        base_workers = int(base["workers"])
        base_total = float(base["total"])
        base_force = float(base["force"])
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

        communication_median = median(communications)

        if mode == "strong":
            speedup = base_total / total_median
            force_speedup = base_force / force_median
            efficiency = speedup / (workers / base_workers)
            runtime_over_ideal = total_median / (base_total / (workers / base_workers))
        else:
            # Direct N-body weak scaling keeps Nlocal fixed per MPI rank, so the
            # ideal runtime grows linearly with ranks.  The scaled speedup below
            # is throughput-based: total work grows approximately as ranks^2.
            rank_ratio = ranks / int(base["ranks"])
            speedup = (rank_ratio * rank_ratio) * base_total / total_median
            force_speedup = (rank_ratio * rank_ratio) * base_force / force_median
            efficiency = speedup / (workers / base_workers)
            runtime_over_ideal = total_median / (base_total * rank_ratio)

        max_drift = max(float(row["max_relative_energy_drift"]) for row in group)
        statuses = "|".join(sorted({row["status"] for row in group}))

        summary.append(
            {
                "backend": backend,
                "mode": mode,
                "n": int(group[0]["n"]),
                "nlocal": int(group[0]["nlocal"]),
                "ranks": ranks,
                "threads": threads,
                "total_workers": workers,
                "repeats": len(group),
                "total_median_s": total_median,
                "total_stdev_s": stdev(totals),
                "force_median_s": force_median,
                "force_stdev_s": stdev(forces),
                "communication_median_s": communication_median,
                "communication_fraction": communication_median / total_median,
                "runtime_over_ideal": runtime_over_ideal,
                "integration_median_s": median(integrations),
                "energy_median_s": median(energies),
                "initial_acceleration_median_s": median(initial_accelerations),
                "speedup": speedup,
                "parallel_efficiency": efficiency,
                "force_speedup": force_speedup,
                "force_efficiency": force_speedup / (workers / base_workers),
                "max_relative_energy_drift": max_drift,
                "statuses": statuses,
                "baseline": "yes" if (backend, ranks, threads) == base["key"] else "no",
            }
        )

    return summary


def markdown_table(summary: list[dict[str, object]]) -> str:
    mode = str(summary[0]["mode"]) if summary else "scaling"
    speedup_label = "Speedup" if mode == "strong" else "Scaled speedup"
    lines = [
        f"| Backend | Ranks | Threads | Workers | N | Nlocal | Repeats | Total median s | Total stdev s | Force median s | Comm median s | Comm fraction | Runtime/ideal | {speedup_label} | Efficiency | Max drift | Status |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|",
    ]

    for row in summary:
        lines.append(
            "| {backend} | {ranks} | {threads} | {workers} | {n} | {nlocal} | {repeats} | {total} | {total_std} | {force} | {comm} | {comm_frac} | {runtime_over_ideal} | {speedup} | {efficiency} | {drift} | {statuses} |".format(
                backend=row["backend"],
                ranks=row["ranks"],
                threads=row["threads"],
                workers=row["total_workers"],
                n=row["n"],
                nlocal=row["nlocal"],
                repeats=row["repeats"],
                total=fmt_float(float(row["total_median_s"])),
                total_std=fmt_float(float(row["total_stdev_s"])),
                force=fmt_float(float(row["force_median_s"])),
                comm=fmt_float(float(row["communication_median_s"])),
                comm_frac=fmt_ratio(float(row["communication_fraction"])),
                runtime_over_ideal=fmt_ratio(float(row["runtime_over_ideal"])),
                speedup=fmt_ratio(float(row["speedup"])),
                efficiency=fmt_ratio(float(row["parallel_efficiency"])),
                drift=fmt_sci(float(row["max_relative_energy_drift"])),
                statuses=row["statuses"],
            )
        )

    return "\n".join(lines)


def write_summary_csv(path: Path, summary: list[dict[str, object]]) -> None:
    fieldnames = [
        "backend",
        "mode",
        "n",
        "nlocal",
        "ranks",
        "threads",
        "total_workers",
        "repeats",
        "total_median_s",
        "total_stdev_s",
        "force_median_s",
        "force_stdev_s",
        "communication_median_s",
        "communication_fraction",
        "runtime_over_ideal",
        "integration_median_s",
        "energy_median_s",
        "initial_acceleration_median_s",
        "speedup",
        "parallel_efficiency",
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
        description="Summarize strong/weak scaling benchmark CSV files."
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        nargs="+",
        help="one or more scaling benchmark CSV files",
    )
    parser.add_argument("--markdown", type=Path, help="optional Markdown output")
    parser.add_argument("--csv", type=Path, help="optional summary CSV output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for input_csv in args.input_csv:
        rows.extend(read_rows(input_csv))
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
