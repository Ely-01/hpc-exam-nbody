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
    time_max = max(float(row["seconds_median"]) for row in summary) * 1.15
    speedup_max = max(1.25, max(float(row["speedup_vs_aos"]) for row in summary) * 1.15)

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="530" viewBox="0 0 1040 530">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial, Helvetica, sans-serif; fill:#111827;}</style>',
        '<text x="520" y="34" text-anchor="middle" font-size="21" font-weight="700">AoS vs SoA force-kernel layout trade-off</text>',
        '<text x="520" y="58" text-anchor="middle" font-size="12" fill="#4b5563">Same direct all-pairs arithmetic, different memory layout. SoA is the production layout.</text>',
        draw_panel(
            summary,
            "seconds_median",
            "Force time",
            "seconds",
            86,
            116,
            390,
            300,
            time_max,
        ),
        draw_panel(
            summary,
            "speedup_vs_aos",
            "Speedup vs AoS",
            "x",
            596,
            116,
            360,
            300,
            speedup_max,
        ),
        "</svg>",
    ]
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
