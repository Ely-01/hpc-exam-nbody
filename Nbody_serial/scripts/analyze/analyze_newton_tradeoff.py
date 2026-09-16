#!/usr/bin/env python3
"""Summarize and plot Newton-third-law force-kernel trade-off benchmarks."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


FLOPS_PER_INTERACTION = {
    "direct": 20.0,
    "newton": 23.0,
    "newton-atomic": 23.0,
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


def interaction_count(n: int, kernel: str) -> int:
    if kernel == "direct":
        return n * (n - 1)
    return n * (n - 1) // 2


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["threads"]), row["kernel"])].append(row)

    if not groups:
        raise ValueError("input CSV does not contain benchmark rows")

    direct_by_threads = {
        threads: median([float(row["force_seconds"]) for row in group])
        for (threads, kernel), group in groups.items()
        if kernel == "direct"
    }
    if not direct_by_threads:
        raise ValueError("input CSV does not contain a direct baseline")

    summary = []
    for threads, kernel in sorted(groups, key=lambda key: (key[0], key[1])):
        group = groups[(threads, kernel)]
        n = int(group[0]["n"])
        nsteps = int(group[0]["nsteps"])
        force_evaluations = nsteps + 1
        per_eval_interactions = interaction_count(n, kernel)
        total_interactions = per_eval_interactions * force_evaluations
        approx_flops = total_interactions * FLOPS_PER_INTERACTION[kernel]
        force_values = [float(row["force_seconds"]) for row in group]
        total_values = [float(row["total_seconds"]) for row in group]
        energy_values = [float(row["energy_seconds"]) for row in group]
        force_median = median(force_values)
        direct_force = direct_by_threads.get(threads)
        speedup_vs_direct = direct_force / force_median if direct_force else 1.0
        ideal_half_time = direct_force / 2.0 if direct_force else force_median
        overhead_seconds = 0.0
        overhead_percent = 0.0
        half_time_ratio = 0.0
        if kernel != "direct":
            half_time_ratio = force_median / ideal_half_time
            overhead_seconds = force_median - ideal_half_time
            overhead_percent = 100.0 * (half_time_ratio - 1.0)

        summary.append(
            {
                "kernel": kernel,
                "n": n,
                "nsteps": nsteps,
                "force_evaluations": force_evaluations,
                "threads": threads,
                "repeats": len(group),
                "interactions_per_force_eval": per_eval_interactions,
                "total_interactions": total_interactions,
                "approx_flops": approx_flops,
                "sqrt_count": total_interactions,
                "division_count": total_interactions,
                "force_median_s": force_median,
                "force_stdev_s": stdev(force_values),
                "total_median_s": median(total_values),
                "energy_median_s": median(energy_values),
                "approx_mflop_s": approx_flops / force_median / 1.0e6,
                "speedup_vs_direct": speedup_vs_direct,
                "ideal_half_time_s": ideal_half_time if kernel != "direct" else 0.0,
                "half_time_ratio": half_time_ratio,
                "overhead_vs_half_time_s": overhead_seconds,
                "overhead_vs_half_time_percent": overhead_percent,
                "max_relative_energy_drift": max(
                    float(row["max_relative_energy_drift"]) for row in group
                ),
                "statuses": "|".join(sorted({row["status"] for row in group})),
            }
        )

    return summary


def write_csv(path: Path, summary: list[dict[str, object]]) -> None:
    fieldnames = [
        "kernel",
        "n",
        "nsteps",
        "force_evaluations",
        "threads",
        "repeats",
        "interactions_per_force_eval",
        "total_interactions",
        "approx_flops",
        "sqrt_count",
        "division_count",
        "force_median_s",
        "force_stdev_s",
        "total_median_s",
        "energy_median_s",
        "approx_mflop_s",
        "speedup_vs_direct",
        "ideal_half_time_s",
        "half_time_ratio",
        "overhead_vs_half_time_s",
        "overhead_vs_half_time_percent",
        "max_relative_energy_drift",
        "statuses",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def markdown_table(summary: list[dict[str, object]]) -> str:
    lines = [
        "| Threads | Kernel | Repeats | Force median s | Force stdev s | Approx MFLOP/s | Speedup vs direct | Overhead vs half-time | Max drift | Status |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for row in summary:
        overhead = "-"
        if row["kernel"] != "direct":
            overhead = f'{float(row["overhead_vs_half_time_percent"]):.2f}%'
        lines.append(
            "| {threads} | {kernel} | {repeats} | {force:.6f} | {stdev:.6f} | {mflops:.1f} | {speedup:.3f} | {overhead} | {drift:.6e} | {status} |".format(
                threads=row["threads"],
                kernel=row["kernel"],
                repeats=row["repeats"],
                force=float(row["force_median_s"]),
                stdev=float(row["force_stdev_s"]),
                mflops=float(row["approx_mflop_s"]),
                speedup=float(row["speedup_vs_direct"]),
                overhead=overhead,
                drift=float(row["max_relative_energy_drift"]),
                status=row["statuses"],
            )
        )
    return "\n".join(lines)


def svg_bar(x: float, y: float, width: float, height: float, color: str) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" fill="{color}"/>'


def draw_grouped_bars(
    rows: list[dict[str, object]],
    x0: float,
    y0: float,
    width: float,
    height: float,
    metric: str,
    title: str,
    y_label: str,
    y_max: float,
    kernels: list[str] | None = None,
) -> str:
    colors = {
        "direct": "#2563eb",
        "newton": "#16a34a",
        "newton-atomic": "#dc2626",
    }
    if kernels is None:
        kernels = ["direct", "newton", "newton-atomic"]
    threads = sorted({int(row["threads"]) for row in rows})
    row_by_key = {(int(row["threads"]), str(row["kernel"])): row for row in rows}
    group_width = width / len(threads)
    bar_width = group_width / (len(kernels) + 1.2)

    def sy(value: float) -> float:
        return y0 + height - value / y_max * height

    parts = [
        f'<text x="{x0 + width / 2:.1f}" y="{y0 - 18:.1f}" text-anchor="middle" font-size="16" font-weight="700">{title}</text>',
        f'<line x1="{x0}" y1="{y0 + height}" x2="{x0 + width}" y2="{y0 + height}" stroke="#222"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + height}" stroke="#222"/>',
        f'<text x="{x0 - 48:.1f}" y="{y0 + height / 2:.1f}" transform="rotate(-90 {x0 - 48:.1f},{y0 + height / 2:.1f})" text-anchor="middle" font-size="12">{y_label}</text>',
        f'<text x="{x0 + width / 2:.1f}" y="{y0 + height + 44:.1f}" text-anchor="middle" font-size="12">OpenMP threads</text>',
    ]

    for i in range(6):
        value = y_max * i / 5
        py = sy(value)
        parts.append(f'<line x1="{x0}" y1="{py:.1f}" x2="{x0 + width}" y2="{py:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{x0 - 8:.1f}" y="{py + 4:.1f}" text-anchor="end" font-size="10">{value:.2g}</text>')

    for group_index, thread in enumerate(threads):
        used_width = len(kernels) * bar_width
        base_x = x0 + group_index * group_width + (group_width - used_width) / 2
        center = x0 + group_index * group_width + group_width / 2
        parts.append(f'<text x="{center:.1f}" y="{y0 + height + 22:.1f}" text-anchor="middle" font-size="11">{thread}</text>')
        for kernel_index, kernel in enumerate(kernels):
            row = row_by_key.get((thread, kernel))
            if row is None:
                continue
            value = float(row[metric])
            py = sy(value)
            parts.append(
                svg_bar(
                    base_x + kernel_index * bar_width,
                    py,
                    bar_width * 0.82,
                    y0 + height - py,
                    colors[kernel],
                )
            )

    legend_x = x0 + width - 150
    for index, kernel in enumerate(kernels):
        ly = y0 + 18 + index * 18
        parts.append(svg_bar(legend_x, ly - 10, 10, 10, colors[kernel]))
        parts.append(f'<text x="{legend_x + 16:.1f}" y="{ly:.1f}" font-size="11">{kernel}</text>')

    return "\n".join(parts)


def write_svg(path: Path, summary: list[dict[str, object]]) -> None:
    force_max = max(float(row["force_median_s"]) for row in summary) * 1.15
    speedup_max = max(2.1, max(float(row["speedup_vs_direct"]) for row in summary) * 1.15)
    ideal_speedup_y = 118 + 310 - 2.0 / speedup_max * 310
    direct_baseline_y = 118 + 310 - 1.0 / speedup_max * 310
    newton_rows = [
        row for row in summary if str(row["kernel"]) != "direct"
    ]

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="560" viewBox="0 0 1120 560">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial, Helvetica, sans-serif; fill:#111827;}</style>',
        '<text x="560" y="34" text-anchor="middle" font-size="21" font-weight="700">Newton-third-law force-kernel trade-off</text>',
        '<text x="560" y="58" text-anchor="middle" font-size="12" fill="#4b5563">Direct does more pair work but parallelises cleanly; Newton halves pair evaluations but needs conflict resolution when parallelised.</text>',
        draw_grouped_bars(
            summary,
            88,
            118,
            420,
            310,
            "force_median_s",
            "Force time",
            "seconds",
            force_max,
        ),
        draw_grouped_bars(
            newton_rows,
            622,
            118,
            420,
            310,
            "speedup_vs_direct",
            "Speedup relative to direct",
            "Speedup factor = T_direct / T_kernel",
            speedup_max,
            kernels=["newton", "newton-atomic"],
        ),
        f'<line x1="622" y1="{ideal_speedup_y:.1f}" x2="1042" y2="{ideal_speedup_y:.1f}" stroke="#9ca3af" stroke-dasharray="5 5"/>',
        f'<text x="930" y="{ideal_speedup_y - 8:.1f}" font-size="11" fill="#6b7280">ideal Newton 2x</text>',
        f'<line x1="622" y1="{direct_baseline_y:.1f}" x2="1042" y2="{direct_baseline_y:.1f}" stroke="#6b7280" stroke-dasharray="4 4"/>',
        f'<text x="910" y="{direct_baseline_y - 8:.1f}" font-size="11" fill="#4b5563">direct baseline 1x</text>',
        "</svg>",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Newton-third-law force-kernel trade-off CSV files."
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--csv", type=Path, help="optional summary CSV output")
    parser.add_argument("--markdown", type=Path, help="optional Markdown output")
    parser.add_argument("--svg", type=Path, help="optional SVG figure output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize(read_rows(args.input_csv))
    table = markdown_table(summary)
    print(table)

    if args.csv is not None:
        write_csv(args.csv, summary)
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(table + "\n")
    if args.svg is not None:
        write_svg(args.svg, summary)


if __name__ == "__main__":
    main()
