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


def draw_svg(path: Path, summary: list[dict[str, object]]) -> None:
    colors = {
        1: "#2563eb",
        2: "#16a34a",
        4: "#ca8a04",
        8: "#dc2626",
        16: "#7c3aed",
    }
    threads = sorted({int(row["threads"]) for row in summary})
    lanes = sorted({int(row["accumulators"]) for row in summary})
    row_by_key = {
        (int(row["threads"]), int(row["accumulators"])): row for row in summary
    }
    speedup_max = max(float(row["speedup_vs_direct"]) for row in summary)
    y_max = max(1.2, math.ceil(speedup_max * 110.0) / 100.0)
    x0 = 122.0
    y0 = 120.0
    width = 820.0
    height = 330.0
    x_min = min(lanes)
    x_max = max(lanes)

    def sx(lane: int) -> float:
      if x_max == x_min:
        return x0 + width / 2.0
      return x0 + (lane - x_min) / (x_max - x_min) * width

    def sy(value: float) -> float:
        return y0 + height - value / y_max * height

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="540" viewBox="0 0 1040 540">',
        '<rect width="1040" height="540" fill="white"/>',
        '<style>text{font-family:Arial, Helvetica, sans-serif; fill:#111827;}</style>',
        '<text x="520" y="36" text-anchor="middle" font-size="22" font-weight="700">Accumulator splitting and critical path</text>',
        '<text x="520" y="62" text-anchor="middle" font-size="12" fill="#4b5563">Speedup is relative to the direct kernel with one accumulator per component.</text>',
        f'<line x1="{x0}" y1="{y0 + height}" x2="{x0 + width}" y2="{y0 + height}" stroke="#222"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + height}" stroke="#222"/>',
        f'<text x="{x0 - 58:.1f}" y="{y0 + height / 2:.1f}" transform="rotate(-90 {x0 - 58:.1f},{y0 + height / 2:.1f})" text-anchor="middle" font-size="12">force-kernel speedup vs direct</text>',
        f'<text x="{x0 + width / 2:.1f}" y="{y0 + height + 52:.1f}" text-anchor="middle" font-size="12">partial accumulators per component</text>',
        f'<line x1="{x0}" y1="{sy(1.0):.1f}" x2="{x0 + width}" y2="{sy(1.0):.1f}" stroke="#9ca3af" stroke-dasharray="6 5"/>',
        f'<text x="{x0 + width - 4:.1f}" y="{sy(1.0) - 8:.1f}" text-anchor="end" font-size="11" fill="#6b7280">direct baseline 1x</text>',
    ]

    for index in range(6):
        value = y_max * index / 5
        py = sy(value)
        parts.append(
            f'<line x1="{x0}" y1="{py:.1f}" x2="{x0 + width}" y2="{py:.1f}" stroke="#e5e7eb"/>'
        )
        parts.append(
            f'<text x="{x0 - 8:.1f}" y="{py + 4:.1f}" text-anchor="end" font-size="10">{value:.2f}x</text>'
        )

    for lane in lanes:
        center = sx(lane)
        parts.append(
            f'<line x1="{center:.1f}" y1="{y0}" x2="{center:.1f}" y2="{y0 + height}" stroke="#f3f4f6"/>'
        )
        parts.append(
            f'<text x="{center:.1f}" y="{y0 + height + 22:.1f}" text-anchor="middle" font-size="11">{lane}</text>'
        )

    for threads_value in threads:
        points = []
        for lane in lanes:
            row = row_by_key.get((threads_value, lane))
            if row is None:
                continue
            points.append((sx(lane), sy(float(row["speedup_vs_direct"]))))
        if len(points) >= 2:
            d = " ".join(
                f"{'M' if index == 0 else 'L'} {px:.1f} {py:.1f}"
                for index, (px, py) in enumerate(points)
            )
            parts.append(
                f'<path d="{d}" fill="none" stroke="{colors.get(threads_value, "#111827")}" stroke-width="2.2"/>'
            )
        for px, py in points:
            parts.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5.2" fill="{colors.get(threads_value, "#111827")}" stroke="white" stroke-width="1.2"/>'
            )

    legend_x = 760
    legend_y = 104
    for index, threads_value in enumerate(threads):
        ly = legend_y + index * 20
        color = colors.get(threads_value, "#111827")
        parts.append(
            f'<line x1="{legend_x}" y1="{ly - 4:.1f}" x2="{legend_x + 22}" y2="{ly - 4:.1f}" stroke="{color}" stroke-width="2.2"/>'
        )
        parts.append(
            f'<circle cx="{legend_x + 11}" cy="{ly - 4:.1f}" r="4.5" fill="{color}" stroke="white" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{legend_x + 30:.1f}" y="{ly:.1f}" font-size="11">{threads_value} thread(s)</text>'
        )

    parts.append("</svg>")
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
