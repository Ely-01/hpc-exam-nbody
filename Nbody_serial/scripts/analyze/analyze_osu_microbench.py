#!/usr/bin/env python3
"""Summarize OSU latency/bandwidth native-vs-container outputs."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


def read_osu_table(path: Path) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 2:
                continue
            try:
                size = int(fields[0])
                value = float(fields[1])
            except ValueError:
                continue
            rows.append((size, value))
    if not rows:
        raise ValueError(f"no OSU data rows found in {path}")
    return rows


def pick_latency(rows: list[tuple[int, float]], size: int) -> tuple[int, float]:
    by_size = dict(rows)
    if size in by_size:
        return size, by_size[size]
    return min(rows, key=lambda row: row[0])


def pick_bandwidth(rows: list[tuple[int, float]], size: int | None) -> tuple[int, float]:
    by_size = dict(rows)
    if size is not None and size in by_size:
        return size, by_size[size]
    return max(rows, key=lambda row: row[0])


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def collect_values(
    paths: list[Path],
    metric: str,
    size: int | None,
) -> tuple[int, list[float]]:
    selected_size: int | None = None
    values: list[float] = []
    for path in paths:
        rows = read_osu_table(path)
        if metric == "latency":
            current_size, current_value = pick_latency(rows, 0 if size is None else size)
        elif metric == "bandwidth":
            current_size, current_value = pick_bandwidth(rows, size)
        else:
            raise ValueError(f"unknown metric '{metric}'")

        if selected_size is None:
            selected_size = current_size
        elif selected_size != current_size:
            raise ValueError(
                f"selected message size mismatch for {metric}: "
                f"{selected_size} vs {current_size} in {path}"
            )
        values.append(current_value)

    if selected_size is None:
        raise ValueError(f"no {metric} files provided")
    return selected_size, values


def pct_delta(container: float, native: float) -> float:
    if native == 0.0:
        return 0.0
    return 100.0 * (container / native - 1.0)


def markdown(summary: dict[str, object]) -> str:
    lines = [
        "| Metric | Message size bytes | Repeats | Native median ± sigma | Container median ± sigma | Relative difference |",
        "|:---|---:|---:|---:|---:|---:|",
        "| Latency (us, lower is better) | {lat_size} | {lat_repeats} | {native_lat:.3f} ± {native_lat_std:.3f} | {container_lat:.3f} ± {container_lat_std:.3f} | {lat_delta:+.2f}% |".format(
            lat_size=summary["latency_size_bytes"],
            lat_repeats=summary["latency_repeats"],
            native_lat=float(summary["native_latency_us"]),
            native_lat_std=float(summary["native_latency_stdev_us"]),
            container_lat=float(summary["container_latency_us"]),
            container_lat_std=float(summary["container_latency_stdev_us"]),
            lat_delta=float(summary["latency_overhead_percent"]),
        ),
        "| Bandwidth (MB/s, higher is better) | {bw_size} | {bw_repeats} | {native_bw:.3f} ± {native_bw_std:.3f} | {container_bw:.3f} ± {container_bw_std:.3f} | {bw_delta:+.2f}% |".format(
            bw_size=summary["bandwidth_size_bytes"],
            bw_repeats=summary["bandwidth_repeats"],
            native_bw=float(summary["native_bandwidth_MBps"]),
            native_bw_std=float(summary["native_bandwidth_stdev_MBps"]),
            container_bw=float(summary["container_bandwidth_MBps"]),
            container_bw_std=float(summary["container_bandwidth_stdev_MBps"]),
            bw_delta=float(summary["bandwidth_delta_percent"]),
        ),
        "",
        "The OSU Micro-Benchmarks were built locally in user space because they were not available as Orfeo modules. The container run uses the same host OpenMPI runtime policy as the application container runs. Positive latency difference means the container is slower; positive bandwidth difference means the container measured higher bandwidth.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize OSU latency and bandwidth native/container outputs."
    )
    parser.add_argument("--native-latency", type=Path, nargs="+", required=True)
    parser.add_argument("--container-latency", type=Path, nargs="+", required=True)
    parser.add_argument("--native-bandwidth", type=Path, nargs="+", required=True)
    parser.add_argument("--container-bandwidth", type=Path, nargs="+", required=True)
    parser.add_argument("--latency-size", type=int, default=0)
    parser.add_argument("--bandwidth-size", type=int)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    native_lat_size, native_lat_values = collect_values(
        args.native_latency, "latency", args.latency_size
    )
    container_lat_size, container_lat_values = collect_values(
        args.container_latency, "latency", args.latency_size
    )
    native_bw_size, native_bw_values = collect_values(
        args.native_bandwidth, "bandwidth", args.bandwidth_size
    )
    container_bw_size, container_bw_values = collect_values(
        args.container_bandwidth, "bandwidth", args.bandwidth_size
    )

    if native_lat_size != container_lat_size:
        raise ValueError("native/container latency sizes do not match")
    if native_bw_size != container_bw_size:
        raise ValueError("native/container bandwidth sizes do not match")

    native_lat = statistics.median(native_lat_values)
    container_lat = statistics.median(container_lat_values)
    native_bw = statistics.median(native_bw_values)
    container_bw = statistics.median(container_bw_values)

    summary = {
        "latency_size_bytes": native_lat_size,
        "latency_repeats": min(len(native_lat_values), len(container_lat_values)),
        "native_latency_us": native_lat,
        "native_latency_stdev_us": stdev(native_lat_values),
        "container_latency_us": container_lat,
        "container_latency_stdev_us": stdev(container_lat_values),
        "latency_overhead_percent": pct_delta(container_lat, native_lat),
        "bandwidth_size_bytes": native_bw_size,
        "bandwidth_repeats": min(len(native_bw_values), len(container_bw_values)),
        "native_bandwidth_MBps": native_bw,
        "native_bandwidth_stdev_MBps": stdev(native_bw_values),
        "container_bandwidth_MBps": container_bw,
        "container_bandwidth_stdev_MBps": stdev(container_bw_values),
        "bandwidth_delta_percent": pct_delta(container_bw, native_bw),
    }

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary))
            writer.writeheader()
            writer.writerow(summary)

    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(summary) + "\n", encoding="utf-8")

    if args.csv is None and args.markdown is None:
        print(markdown(summary))


if __name__ == "__main__":
    main()
