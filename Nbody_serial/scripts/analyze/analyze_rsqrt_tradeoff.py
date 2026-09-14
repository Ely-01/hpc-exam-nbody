#!/usr/bin/env python3
"""Summarize and plot reciprocal-square-root accuracy/performance trade-offs."""

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


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["threads"]), row["inv_sqrt"])].append(row)

    if not groups:
        raise ValueError("input CSV does not contain benchmark rows")

    libm_force = {
        threads: median([float(row["force_seconds"]) for row in group])
        for (threads, mode), group in groups.items()
        if mode == "libm"
    }
    libm_drift = {
        threads: max(float(row["max_relative_energy_drift"]) for row in group)
        for (threads, mode), group in groups.items()
        if mode == "libm"
    }

    if not libm_force:
        raise ValueError("input CSV does not contain libm baseline rows")

    summary = []
    for threads, mode in sorted(groups, key=lambda key: (key[0], key[1])):
        group = groups[(threads, mode)]
        force_values = [float(row["force_seconds"]) for row in group]
        total_values = [float(row["total_seconds"]) for row in group]
        energy_values = [float(row["energy_seconds"]) for row in group]
        force_median = median(force_values)
        drift = max(float(row["max_relative_energy_drift"]) for row in group)
        baseline_force = libm_force[threads]
        baseline_drift = libm_drift[threads]

        summary.append(
            {
                "inv_sqrt": mode,
                "force_kernel": group[0]["force_kernel"],
                "n": int(group[0]["n"]),
                "nsteps": int(group[0]["nsteps"]),
                "threads": threads,
                "repeats": len(group),
                "force_median_s": force_median,
                "force_stdev_s": stdev(force_values),
                "total_median_s": median(total_values),
                "energy_median_s": median(energy_values),
                "speedup_vs_libm": baseline_force / force_median,
                "max_relative_energy_drift": drift,
                "drift_ratio_vs_libm": drift / baseline_drift if baseline_drift > 0.0 else 0.0,
                "drift_delta_vs_libm": drift - baseline_drift,
                "drift_percent_change_vs_libm": (
                    100.0 * (drift / baseline_drift - 1.0)
                    if baseline_drift > 0.0
                    else 0.0
                ),
                "statuses": "|".join(sorted({row["status"] for row in group})),
            }
        )

    return summary


def write_csv(path: Path, summary: list[dict[str, object]]) -> None:
    fieldnames = [
        "inv_sqrt",
        "force_kernel",
        "n",
        "nsteps",
        "threads",
        "repeats",
        "force_median_s",
        "force_stdev_s",
        "total_median_s",
        "energy_median_s",
        "speedup_vs_libm",
        "max_relative_energy_drift",
        "drift_ratio_vs_libm",
        "drift_delta_vs_libm",
        "drift_percent_change_vs_libm",
        "statuses",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def markdown_table(summary: list[dict[str, object]]) -> str:
    lines = [
        "| Threads | inv sqrt | Repeats | Force median s | Force stdev s | Speedup vs libm | Max drift | Drift change vs libm | Status |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for row in summary:
        lines.append(
            "| {threads} | {mode} | {repeats} | {force:.6f} | {stdev:.6f} | {speedup:.3f} | {drift:.6e} | {change:.3f}% | {status} |".format(
                threads=row["threads"],
                mode=row["inv_sqrt"],
                repeats=row["repeats"],
                force=float(row["force_median_s"]),
                stdev=float(row["force_stdev_s"]),
                speedup=float(row["speedup_vs_libm"]),
                drift=float(row["max_relative_energy_drift"]),
                change=float(row["drift_percent_change_vs_libm"]),
                status=row["statuses"],
            )
        )
    return "\n".join(lines)


def bar(x: float, y: float, width: float, height: float, color: str) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" fill="{color}"/>'


def draw_panel(
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
    colors = {
        "libm": "#2563eb",
        "rsqrt1": "#16a34a",
        "rsqrt2": "#ca8a04",
        "rsqrt3": "#dc2626",
    }
    modes = ["libm", "rsqrt1", "rsqrt2", "rsqrt3"]
    threads = sorted({int(row["threads"]) for row in rows})
    row_by_key = {(int(row["threads"]), str(row["inv_sqrt"])): row for row in rows}
    group_width = width / len(threads)
    bar_width = group_width / 5.2

    def sy(value: float) -> float:
        return y0 + height - value / y_max * height

    parts = [
        f'<text x="{x0 + width / 2:.1f}" y="{y0 - 18:.1f}" text-anchor="middle" font-size="16" font-weight="700">{title}</text>',
        f'<line x1="{x0}" y1="{y0 + height}" x2="{x0 + width}" y2="{y0 + height}" stroke="#222"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + height}" stroke="#222"/>',
        f'<text x="{x0 - 48:.1f}" y="{y0 + height / 2:.1f}" transform="rotate(-90 {x0 - 48:.1f},{y0 + height / 2:.1f})" text-anchor="middle" font-size="12">{y_label}</text>',
        f'<text x="{x0 + width / 2:.1f}" y="{y0 + height + 42:.1f}" text-anchor="middle" font-size="12">OpenMP threads</text>',
    ]

    for i in range(6):
        value = y_max * i / 5
        py = sy(value)
        parts.append(f'<line x1="{x0}" y1="{py:.1f}" x2="{x0 + width}" y2="{py:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{x0 - 8:.1f}" y="{py + 4:.1f}" text-anchor="end" font-size="10">{value:.2g}</text>')

    for group_index, thread in enumerate(threads):
        base_x = x0 + group_index * group_width + group_width * 0.12
        center = x0 + group_index * group_width + group_width / 2
        parts.append(f'<text x="{center:.1f}" y="{y0 + height + 22:.1f}" text-anchor="middle" font-size="11">{thread}</text>')
        for mode_index, mode in enumerate(modes):
            row = row_by_key.get((thread, mode))
            if row is None:
                continue
            value = float(row[metric])
            py = sy(value)
            parts.append(
                bar(
                    base_x + mode_index * bar_width,
                    py,
                    bar_width * 0.82,
                    y0 + height - py,
                    colors[mode],
                )
            )

    legend_x = x0 + width - 112
    for index, mode in enumerate(modes):
        ly = y0 + 18 + index * 18
        parts.append(bar(legend_x, ly - 10, 10, 10, colors[mode]))
        parts.append(f'<text x="{legend_x + 16:.1f}" y="{ly:.1f}" font-size="11">{mode}</text>')

    return "\n".join(parts)


def draw_signed_panel(
    rows: list[dict[str, object]],
    metric: str,
    title: str,
    y_label: str,
    x0: float,
    y0: float,
    width: float,
    height: float,
    y_abs_max: float,
) -> str:
    colors = {
        "libm": "#2563eb",
        "rsqrt1": "#16a34a",
        "rsqrt2": "#ca8a04",
        "rsqrt3": "#dc2626",
    }
    modes = ["libm", "rsqrt1", "rsqrt2", "rsqrt3"]
    threads = sorted({int(row["threads"]) for row in rows})
    row_by_key = {(int(row["threads"]), str(row["inv_sqrt"])): row for row in rows}
    group_width = width / len(threads)
    bar_width = group_width / 5.2

    def sy(value: float) -> float:
        return y0 + height - (value + y_abs_max) / (2.0 * y_abs_max) * height

    zero_y = sy(0.0)
    parts = [
        f'<text x="{x0 + width / 2:.1f}" y="{y0 - 18:.1f}" text-anchor="middle" font-size="16" font-weight="700">{title}</text>',
        f'<line x1="{x0}" y1="{y0 + height}" x2="{x0 + width}" y2="{y0 + height}" stroke="#222"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + height}" stroke="#222"/>',
        f'<line x1="{x0}" y1="{zero_y:.1f}" x2="{x0 + width}" y2="{zero_y:.1f}" stroke="#6b7280" stroke-width="1.5"/>',
        f'<text x="{x0 - 52:.1f}" y="{y0 + height / 2:.1f}" transform="rotate(-90 {x0 - 52:.1f},{y0 + height / 2:.1f})" text-anchor="middle" font-size="12">{y_label}</text>',
        f'<text x="{x0 + width / 2:.1f}" y="{y0 + height + 42:.1f}" text-anchor="middle" font-size="12">OpenMP threads</text>',
    ]

    for i in range(5):
        value = -y_abs_max + 2.0 * y_abs_max * i / 4
        py = sy(value)
        parts.append(f'<line x1="{x0}" y1="{py:.1f}" x2="{x0 + width}" y2="{py:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{x0 - 8:.1f}" y="{py + 4:.1f}" text-anchor="end" font-size="10">{value:.2g}</text>')

    for group_index, thread in enumerate(threads):
        base_x = x0 + group_index * group_width + group_width * 0.12
        center = x0 + group_index * group_width + group_width / 2
        parts.append(f'<text x="{center:.1f}" y="{y0 + height + 22:.1f}" text-anchor="middle" font-size="11">{thread}</text>')
        for mode_index, mode in enumerate(modes):
            row = row_by_key.get((thread, mode))
            if row is None:
                continue
            value = float(row[metric])
            py = sy(value)
            if value >= 0.0:
                y = py
                h = zero_y - py
            else:
                y = zero_y
                h = py - zero_y
            parts.append(
                bar(
                    base_x + mode_index * bar_width,
                    y,
                    bar_width * 0.82,
                    h,
                    colors[mode],
                )
            )

    legend_x = x0 + width - 112
    for index, mode in enumerate(modes):
        ly = y0 + 18 + index * 18
        parts.append(bar(legend_x, ly - 10, 10, 10, colors[mode]))
        parts.append(f'<text x="{legend_x + 16:.1f}" y="{ly:.1f}" font-size="11">{mode}</text>')

    return "\n".join(parts)


def write_svg(path: Path, summary: list[dict[str, object]]) -> None:
    speedup_max = max(1.15, max(float(row["speedup_vs_libm"]) for row in summary) * 1.15)
    drift_change_abs_max = max(
        0.01,
        max(abs(float(row["drift_percent_change_vs_libm"])) for row in summary) * 1.2,
    )

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="530" viewBox="0 0 1040 530">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial, Helvetica, sans-serif; fill:#111827;}</style>',
        '<text x="520" y="34" text-anchor="middle" font-size="21" font-weight="700">Reciprocal square-root trade-off</text>',
        '<text x="520" y="58" text-anchor="middle" font-size="12" fill="#4b5563">rsqrt modes use an approximate reciprocal sqrt estimate plus Newton refinements; libm remains the reference.</text>',
        draw_panel(
            summary,
            "speedup_vs_libm",
            "Force speedup",
            "x vs libm",
            86,
            116,
            390,
            300,
            speedup_max,
        ),
        draw_signed_panel(
            summary,
            "drift_percent_change_vs_libm",
            "Energy drift change",
            "% vs libm",
            596,
            116,
            360,
            300,
            drift_change_abs_max,
        ),
        "</svg>",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze reciprocal-square-root benchmark CSV files."
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
