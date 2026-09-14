#!/usr/bin/env python3
"""Summarize and plot accumulator-splitting force-kernel benchmarks."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


KERNEL_LANES = {
    "direct": 1,
    "direct-split2": 2,
    "direct-split4": 4,
    "direct-split8": 8,
}


def median(values: list[float]) -> float:
    return statistics.median(values)


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(
    rows: list[dict[str, str]], flops_per_interaction: float
) -> list[dict[str, object]]:
    groups: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        kernel = row["kernel"]
        if kernel not in KERNEL_LANES:
            raise ValueError(f"unknown kernel '{kernel}'")
        groups[(int(row["threads"]), kernel)].append(row)

    if not groups:
        raise ValueError("input CSV does not contain benchmark rows")

    direct_force = {
        threads: median([float(row["force_seconds"]) for row in group])
        for (threads, kernel), group in groups.items()
        if kernel == "direct"
    }
    if not direct_force:
        raise ValueError("input CSV does not contain direct baseline rows")

    summary = []
    for threads, kernel in sorted(
        groups, key=lambda key: (key[0], KERNEL_LANES[key[1]])
    ):
        group = groups[(threads, kernel)]
        n = int(group[0]["n"])
        nsteps = int(group[0]["nsteps"])
        force_values = [float(row["force_seconds"]) for row in group]
        total_values = [float(row["total_seconds"]) for row in group]
        energy_values = [float(row["energy_seconds"]) for row in group]
        force_median = median(force_values)
        force_calls = nsteps + 1
        interactions = float(force_calls) * float(n) * float(n - 1)
        ginteractions_s = interactions / force_median / 1.0e9
        nominal_gflops_s = ginteractions_s * flops_per_interaction
        baseline = direct_force[threads]

        summary.append(
            {
                "kernel": kernel,
                "accumulators": KERNEL_LANES[kernel],
                "inv_sqrt": group[0]["inv_sqrt"],
                "n": n,
                "nsteps": nsteps,
                "threads": threads,
                "repeats": len(group),
                "force_median_s": force_median,
                "force_stdev_s": stdev(force_values),
                "total_median_s": median(total_values),
                "energy_median_s": median(energy_values),
                "speedup_vs_direct": baseline / force_median,
                "ginteractions_s": ginteractions_s,
                "nominal_gflops_s": nominal_gflops_s,
                "max_relative_energy_drift": max(
                    float(row["max_relative_energy_drift"]) for row in group
                ),
                "statuses": "|".join(sorted({row["status"] for row in group})),
            }
        )

    by_thread: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in summary:
        by_thread[int(row["threads"])].append(row)

    for thread_rows in by_thread.values():
        previous_speedup = None
        for row in sorted(thread_rows, key=lambda item: int(item["accumulators"])):
            speedup = float(row["speedup_vs_direct"])
            if previous_speedup is None:
                row["marginal_speedup_gain"] = 0.0
            else:
                row["marginal_speedup_gain"] = speedup / previous_speedup - 1.0
            previous_speedup = speedup

    return summary


def write_csv(path: Path, summary: list[dict[str, object]]) -> None:
    fieldnames = [
        "kernel",
        "accumulators",
        "inv_sqrt",
        "n",
        "nsteps",
        "threads",
        "repeats",
        "force_median_s",
        "force_stdev_s",
        "total_median_s",
        "energy_median_s",
        "speedup_vs_direct",
        "marginal_speedup_gain",
        "ginteractions_s",
        "nominal_gflops_s",
        "max_relative_energy_drift",
        "statuses",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def saturation_notes(
    summary: list[dict[str, object]], threshold: float
) -> list[str]:
    by_thread: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in summary:
        by_thread[int(row["threads"])].append(row)

    notes = []
    for threads in sorted(by_thread):
        rows = sorted(by_thread[threads], key=lambda row: int(row["accumulators"]))
        best = min(rows, key=lambda row: float(row["force_median_s"]))
        saturation = None
        for row in rows:
            if int(row["accumulators"]) == 1:
                continue
            if float(row["marginal_speedup_gain"]) < threshold:
                saturation = row
                break
        if saturation is None:
            sat_text = f"no saturation below {threshold:.0%} marginal gain"
        else:
            sat_text = (
                f"saturates around {int(saturation['accumulators'])} accumulators "
                f"({100.0 * float(saturation['marginal_speedup_gain']):.1f}% "
                "marginal gain)"
            )
        notes.append(
            f"- {threads} thread(s): best {best['kernel']} "
            f"({float(best['speedup_vs_direct']):.3f}x vs direct); {sat_text}."
        )
    return notes


def markdown_table(summary: list[dict[str, object]], threshold: float) -> str:
    lines = [
        "| Threads | Kernel | Accumulators | Repeats | Force median s | Force stdev s | Speedup vs direct | Marginal gain | Gpair/s | Nominal GFLOP/s | Max drift | Status |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for row in summary:
        lines.append(
            "| {threads} | {kernel} | {acc} | {repeats} | {force:.6f} | {stdev:.6f} | {speedup:.3f} | {marginal:.2f}% | {gpair:.3f} | {gflops:.2f} | {drift:.6e} | {status} |".format(
                threads=row["threads"],
                kernel=row["kernel"],
                acc=row["accumulators"],
                repeats=row["repeats"],
                force=float(row["force_median_s"]),
                stdev=float(row["force_stdev_s"]),
                speedup=float(row["speedup_vs_direct"]),
                marginal=100.0 * float(row["marginal_speedup_gain"]),
                gpair=float(row["ginteractions_s"]),
                gflops=float(row["nominal_gflops_s"]),
                drift=float(row["max_relative_energy_drift"]),
                status=row["statuses"],
            )
        )

    lines.extend(
        [
            "",
            "Saturation notes:",
            *saturation_notes(summary, threshold),
            "",
            "The nominal GFLOP/s column uses an approximate flop count per pair; use it as a relative kernel-throughput indicator, not as a hardware-counter replacement.",
        ]
    )
    return "\n".join(lines)


def svg_escape(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def bar(x: float, y: float, width: float, height: float, color: str) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" '
        f'height="{height:.1f}" fill="{color}"/>'
    )


def draw_grouped_panel(
    rows: list[dict[str, object]],
    metric: str,
    title: str,
    y_label: str,
    x0: float,
    y0: float,
    width: float,
    height: float,
    y_max: float,
) -> str:
    colors = ["#2563eb", "#16a34a", "#ca8a04", "#dc2626", "#7c3aed"]
    threads = sorted({int(row["threads"]) for row in rows})
    lanes = sorted({int(row["accumulators"]) for row in rows})
    row_by_key = {
        (int(row["threads"]), int(row["accumulators"])): row for row in rows
    }
    group_width = width / len(lanes)
    bar_width = group_width / (len(threads) + 1.4)

    def sy(value: float) -> float:
        return y0 + height - value / y_max * height

    parts = [
        f'<text x="{x0 + width / 2:.1f}" y="{y0 - 18:.1f}" text-anchor="middle" font-size="16" font-weight="700">{svg_escape(title)}</text>',
        f'<line x1="{x0}" y1="{y0 + height}" x2="{x0 + width}" y2="{y0 + height}" stroke="#222"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + height}" stroke="#222"/>',
        f'<text x="{x0 - 52:.1f}" y="{y0 + height / 2:.1f}" transform="rotate(-90 {x0 - 52:.1f},{y0 + height / 2:.1f})" text-anchor="middle" font-size="12">{svg_escape(y_label)}</text>',
        f'<text x="{x0 + width / 2:.1f}" y="{y0 + height + 42:.1f}" text-anchor="middle" font-size="12">Partial accumulators</text>',
    ]

    for index in range(6):
        value = y_max * index / 5
        py = sy(value)
        parts.append(
            f'<line x1="{x0}" y1="{py:.1f}" x2="{x0 + width}" y2="{py:.1f}" stroke="#e5e7eb"/>'
        )
        parts.append(
            f'<text x="{x0 - 8:.1f}" y="{py + 4:.1f}" text-anchor="end" font-size="10">{value:.2g}</text>'
        )

    for group_index, lane in enumerate(lanes):
        base_x = x0 + group_index * group_width + group_width * 0.12
        center = x0 + group_index * group_width + group_width / 2
        parts.append(
            f'<text x="{center:.1f}" y="{y0 + height + 22:.1f}" text-anchor="middle" font-size="11">{lane}</text>'
        )
        for thread_index, threads_value in enumerate(threads):
            row = row_by_key.get((threads_value, lane))
            if row is None:
                continue
            value = float(row[metric])
            py = sy(value)
            parts.append(
                bar(
                    base_x + thread_index * bar_width,
                    py,
                    bar_width * 0.82,
                    y0 + height - py,
                    colors[thread_index % len(colors)],
                )
            )

    legend_x = x0 + width - 96
    for index, threads_value in enumerate(threads):
        ly = y0 + 18 + index * 18
        parts.append(bar(legend_x, ly - 10, 10, 10, colors[index % len(colors)]))
        parts.append(
            f'<text x="{legend_x + 16:.1f}" y="{ly:.1f}" font-size="11">{threads_value} thread(s)</text>'
        )

    return "\n".join(parts)


def draw_svg(path: Path, summary: list[dict[str, object]]) -> None:
    speedup_max = max(float(row["speedup_vs_direct"]) for row in summary)
    throughput_max = max(float(row["ginteractions_s"]) for row in summary)
    speedup_y = max(1.05, math.ceil(speedup_max * 12.0) / 10.0)
    throughput_y = max(0.01, math.ceil(throughput_max * 12.0) / 10.0)

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="720" viewBox="0 0 1500 720">',
        '<rect width="1500" height="720" fill="white"/>',
        '<text x="750" y="38" text-anchor="middle" font-size="24" font-weight="700">Accumulator splitting and force-kernel critical path</text>',
        '<text x="750" y="66" text-anchor="middle" font-size="13">More independent partial sums shorten the dependency chain; the useful gain saturates once arithmetic throughput or overhead dominates.</text>',
        draw_grouped_panel(
            summary,
            "speedup_vs_direct",
            "Force speedup",
            "x vs direct",
            95,
            135,
            590,
            420,
            speedup_y,
        ),
        draw_grouped_panel(
            summary,
            "ginteractions_s",
            "Nominal interaction throughput",
            "Gpair/s",
            830,
            135,
            590,
            420,
            throughput_y,
        ),
        "</svg>",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze accumulator-splitting force-kernel benchmark CSV files."
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--svg", type=Path)
    parser.add_argument(
        "--flops-per-interaction",
        type=float,
        default=20.0,
        help="Approximate floating-point operations per pair interaction.",
    )
    parser.add_argument(
        "--saturation-threshold",
        type=float,
        default=0.03,
        help="Marginal speedup gain below which a thread case is called saturated.",
    )
    args = parser.parse_args()

    rows = read_rows(args.input_csv)
    summary = summarize(rows, args.flops_per_interaction)

    if args.csv is not None:
        write_csv(args.csv, summary)
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(
            markdown_table(summary, args.saturation_threshold) + "\n",
            encoding="utf-8",
        )
    if args.svg is not None:
        draw_svg(args.svg, summary)

    if args.csv is None and args.markdown is None and args.svg is None:
        print(markdown_table(summary, args.saturation_threshold))


if __name__ == "__main__":
    main()
