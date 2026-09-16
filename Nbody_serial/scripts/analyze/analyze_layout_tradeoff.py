#!/usr/bin/env python3
"""Summarize and plot AoS-vs-SoA force-kernel layout benchmarks."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


FLOPS_PER_INTERACTION = 20.0


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
        groups[(int(row["threads"]), row["layout"])].append(row)

    if not groups:
        raise ValueError("input CSV does not contain benchmark rows")

    aos_by_threads = {
        threads: median([float(row["seconds"]) for row in group])
        for (threads, layout), group in groups.items()
        if layout == "aos"
    }

    summary = []
    for threads, layout in sorted(groups, key=lambda key: (key[0], key[1])):
        group = groups[(threads, layout)]
        n = int(group[0]["n"])
        interactions = n * (n - 1)
        approx_flops = interactions * FLOPS_PER_INTERACTION
        seconds = [float(row["seconds"]) for row in group]
        seconds_median = median(seconds)
        aos_baseline = aos_by_threads.get(threads)
        speedup_vs_aos = aos_baseline / seconds_median if aos_baseline else 1.0

        summary.append(
            {
                "layout": layout,
                "n": n,
                "threads": threads,
                "repeats": len(group),
                "interactions": interactions,
                "approx_flops": approx_flops,
                "seconds_median": seconds_median,
                "seconds_stdev": stdev(seconds),
                "approx_mflop_s": approx_flops / seconds_median / 1.0e6,
                "speedup_vs_aos": speedup_vs_aos,
                "max_abs_accel_diff": max(
                    float(row["max_abs_accel_diff"]) for row in group
                ),
                "statuses": "|".join(sorted({row["status"] for row in group})),
            }
        )

    return summary


def write_csv(path: Path, summary: list[dict[str, object]]) -> None:
    fieldnames = [
        "layout",
        "n",
        "threads",
        "repeats",
        "interactions",
        "approx_flops",
        "seconds_median",
        "seconds_stdev",
        "approx_mflop_s",
        "speedup_vs_aos",
        "max_abs_accel_diff",
        "statuses",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def markdown_table(summary: list[dict[str, object]]) -> str:
    lines = [
        "| Threads | Layout | Repeats | Time median s | Time stdev s | Approx MFLOP/s | Speedup vs AoS | Max accel diff | Status |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for row in summary:
        lines.append(
            "| {threads} | {layout} | {repeats} | {seconds:.6f} | {stdev:.6f} | {mflops:.1f} | {speedup:.3f} | {diff:.6e} | {status} |".format(
                threads=row["threads"],
                layout=row["layout"],
                repeats=row["repeats"],
                seconds=float(row["seconds_median"]),
                stdev=float(row["seconds_stdev"]),
                mflops=float(row["approx_mflop_s"]),
                speedup=float(row["speedup_vs_aos"]),
                diff=float(row["max_abs_accel_diff"]),
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
    colors = {"aos": "#dc2626", "soa": "#2563eb"}
    layouts = ["aos", "soa"]
    threads = sorted({int(row["threads"]) for row in rows})
    row_by_key = {(int(row["threads"]), str(row["layout"])): row for row in rows}
    group_width = width / len(threads)
    bar_width = group_width / 3.1

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
        base_x = x0 + group_index * group_width + group_width * 0.24
        center = x0 + group_index * group_width + group_width / 2
        parts.append(f'<text x="{center:.1f}" y="{y0 + height + 22:.1f}" text-anchor="middle" font-size="11">{thread}</text>')
        for layout_index, layout in enumerate(layouts):
            row = row_by_key.get((thread, layout))
            if row is None:
                continue
            value = float(row[metric])
            py = sy(value)
            parts.append(
                bar(
                    base_x + layout_index * bar_width,
                    py,
                    bar_width * 0.82,
                    y0 + height - py,
                    colors[layout],
                )
            )

    legend_x = x0 + width - 90
    for index, layout in enumerate(layouts):
        ly = y0 + 18 + index * 18
        parts.append(bar(legend_x, ly - 10, 10, 10, colors[layout]))
        parts.append(f'<text x="{legend_x + 16:.1f}" y="{ly:.1f}" font-size="11">{layout}</text>')

    return "\n".join(parts)


def write_svg(path: Path, summary: list[dict[str, object]]) -> None:
    soa_rows = [row for row in summary if row["layout"] == "soa"]
    speedups = [float(row["speedup_vs_aos"]) for row in soa_rows]
    y_min = min(speedups + [1.0])
    y_max = max(speedups + [1.0])
    y_pad = max(0.015, (y_max - y_min) * 0.35)
    y_min = max(0.0, y_min - y_pad)
    y_max = y_max + y_pad

    # Keep the ratio plot visibly centered around the AoS baseline even when
    # the measured effect is small. This figure is meant to show whether SoA
    # actually moves away from 1x, not to hide small differences on a 0-based
    # bar chart.
    y_min = min(y_min, 0.97)
    y_max = max(y_max, 1.03)

    threads = [int(row["threads"]) for row in soa_rows]
    row_by_thread = {int(row["threads"]): row for row in soa_rows}
    x0 = 120.0
    y0 = 120.0
    width = 760.0
    height = 300.0
    group_width = width / len(threads)
    bar_width = group_width * 0.36

    def sy(value: float) -> float:
        return y0 + height - (value - y_min) / (y_max - y_min) * height

    baseline_y = sy(1.0)

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="520" viewBox="0 0 1040 520">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial, Helvetica, sans-serif; fill:#111827;}</style>',
        '<text x="520" y="34" text-anchor="middle" font-size="21" font-weight="700">AoS vs SoA force-kernel layout trade-off</text>',
        '<text x="520" y="58" text-anchor="middle" font-size="12" fill="#4b5563">Ratio plot: values above 1 favor SoA; values below 1 favor AoS. Same arithmetic, different memory layout.</text>',
        f'<text x="{x0 + width / 2:.1f}" y="{y0 - 18:.1f}" text-anchor="middle" font-size="16" font-weight="700">SoA speedup relative to AoS</text>',
        f'<line x1="{x0}" y1="{y0 + height}" x2="{x0 + width}" y2="{y0 + height}" stroke="#222"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + height}" stroke="#222"/>',
        f'<text x="{x0 - 58:.1f}" y="{y0 + height / 2:.1f}" transform="rotate(-90 {x0 - 58:.1f},{y0 + height / 2:.1f})" text-anchor="middle" font-size="12">Speedup factor = T_AoS / T_SoA</text>',
        f'<text x="{x0 + width / 2:.1f}" y="{y0 + height + 48:.1f}" text-anchor="middle" font-size="12">OpenMP threads</text>',
        f'<line x1="{x0}" y1="{baseline_y:.1f}" x2="{x0 + width}" y2="{baseline_y:.1f}" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="6 5"/>',
        f'<text x="{x0 + width - 2:.1f}" y="{baseline_y - 8:.1f}" text-anchor="end" font-size="11" fill="#dc2626">AoS baseline 1x</text>',
    ]

    for i in range(6):
        value = y_min + (y_max - y_min) * i / 5
        py = sy(value)
        parts.append(f'<line x1="{x0}" y1="{py:.1f}" x2="{x0 + width}" y2="{py:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{x0 - 8:.1f}" y="{py + 4:.1f}" text-anchor="end" font-size="10">{value:.3f}</text>')

    for group_index, thread in enumerate(threads):
        row = row_by_thread[thread]
        speedup = float(row["speedup_vs_aos"])
        center = x0 + group_index * group_width + group_width / 2
        bar_x = center - bar_width / 2
        bar_y = min(sy(speedup), baseline_y)
        bar_h = abs(sy(speedup) - baseline_y)
        color = "#2563eb" if speedup >= 1.0 else "#f59e0b"
        parts.append(bar(bar_x, bar_y, bar_width, bar_h, color))
        parts.append(f'<text x="{center:.1f}" y="{y0 + height + 24:.1f}" text-anchor="middle" font-size="11">{thread}</text>')
        label_y = bar_y - 8 if speedup >= 1.0 else bar_y + bar_h + 16
        parts.append(f'<text x="{center:.1f}" y="{label_y:.1f}" text-anchor="middle" font-size="10">{speedup:.3f}x</text>')

    parts.extend(
        [
            '<rect x="690" y="88" width="10" height="10" fill="#2563eb"/>',
            '<text x="706" y="98" font-size="11">SoA faster</text>',
            '<rect x="690" y="108" width="10" height="10" fill="#f59e0b"/>',
            '<text x="706" y="118" font-size="11">AoS faster</text>',
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze AoS-vs-SoA force-kernel benchmark CSV files."
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
