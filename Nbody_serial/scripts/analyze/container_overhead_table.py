#!/usr/bin/env python3
"""Build native-vs-container overhead tables from scaling summary CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def key_for(row: dict[str, str]) -> tuple[str, int, int, int, int]:
    return (
        row["mode"],
        int(row["ranks"]),
        int(row["threads"]),
        int(row["n"]),
        int(row["nlocal"]),
    )


def fmt_float(value: float) -> str:
    return f"{value:.6f}"


def fmt_overhead(value: float) -> str:
    return f"{value:+.2f}%"


def row_status(row: dict[str, str]) -> str:
    return row.get("statuses") or row.get("status") or "unknown"


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_key: dict[tuple[str, int, int, int, int], dict[str, dict[str, str]]] = {}

    for row in rows:
        backend = row.get("backend", "native")
        if backend not in {"native", "container"}:
            continue
        by_key.setdefault(key_for(row), {})[backend] = row

    summary: list[dict[str, object]] = []
    for key, pair in sorted(by_key.items()):
        if "native" not in pair or "container" not in pair:
            continue

        mode, ranks, threads, n, nlocal = key
        native = pair["native"]
        container = pair["container"]
        native_total = float(native["total_median_s"])
        container_total = float(container["total_median_s"])
        overhead = (container_total / native_total - 1.0) * 100.0

        summary.append(
            {
                "mode": mode,
                "ranks": ranks,
                "threads": threads,
                "workers": ranks * threads,
                "n": n,
                "nlocal": nlocal,
                "native_total_median_s": native_total,
                "native_total_stdev_s": float(native["total_stdev_s"]),
                "container_total_median_s": container_total,
                "container_total_stdev_s": float(container["total_stdev_s"]),
                "overhead_percent": overhead,
                "native_comm_fraction": float(native["communication_fraction"]),
                "container_comm_fraction": float(container["communication_fraction"]),
                "native_max_drift": float(native["max_relative_energy_drift"]),
                "container_max_drift": float(container["max_relative_energy_drift"]),
                "status": f"{row_status(native)}/{row_status(container)}",
            }
        )

    if not summary:
        raise ValueError("no matching native/container rows found")

    return summary


def markdown_table(summary: list[dict[str, object]]) -> str:
    lines = [
        "| Mode | Ranks | Threads | Workers | N | Nlocal | Native median +/- sigma s | Container median +/- sigma s | Overhead | Native comm frac | Container comm frac | Max drift | Status |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|",
    ]

    for row in summary:
        max_drift = max(
            float(row["native_max_drift"]), float(row["container_max_drift"])
        )
        lines.append(
            "| {mode} | {ranks} | {threads} | {workers} | {n} | {nlocal} | {native_med} +/- {native_std} | {container_med} +/- {container_std} | {overhead} | {native_comm:.3f} | {container_comm:.3f} | {drift:.6e} | {status} |".format(
                mode=row["mode"],
                ranks=row["ranks"],
                threads=row["threads"],
                workers=row["workers"],
                n=row["n"],
                nlocal=row["nlocal"],
                native_med=fmt_float(float(row["native_total_median_s"])),
                native_std=fmt_float(float(row["native_total_stdev_s"])),
                container_med=fmt_float(float(row["container_total_median_s"])),
                container_std=fmt_float(float(row["container_total_stdev_s"])),
                overhead=fmt_overhead(float(row["overhead_percent"])),
                native_comm=float(row["native_comm_fraction"]),
                container_comm=float(row["container_comm_fraction"]),
                drift=max_drift,
                status=row["status"],
            )
        )

    return "\n".join(lines)


def write_csv(path: Path, summary: list[dict[str, object]]) -> None:
    fieldnames = [
        "mode",
        "ranks",
        "threads",
        "workers",
        "n",
        "nlocal",
        "native_total_median_s",
        "native_total_stdev_s",
        "container_total_median_s",
        "container_total_stdev_s",
        "overhead_percent",
        "native_comm_fraction",
        "container_comm_fraction",
        "native_max_drift",
        "container_max_drift",
        "status",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create native-vs-container overhead tables from scaling summaries."
    )
    parser.add_argument("summary_csv", type=Path, nargs="+")
    parser.add_argument("--csv", type=Path, help="optional CSV output")
    parser.add_argument("--markdown", type=Path, help="optional Markdown output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize(read_rows(args.summary_csv))
    table = markdown_table(summary)
    print(table)

    if args.csv is not None:
        write_csv(args.csv, summary)
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(table + "\n")


if __name__ == "__main__":
    main()
