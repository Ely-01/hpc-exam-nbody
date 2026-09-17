#!/usr/bin/env python3
"""Summarize and plot the isolated SIMD rsqrt microbenchmark."""

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
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["method"]].append(row)

    if "sqrtf" not in groups:
        raise ValueError("input CSV does not contain sqrtf baseline rows")

    sqrtf_median = median([float(row["seconds"]) for row in groups["sqrtf"]])
    order = {"sqrtf": 0, "rsqrt0": 1, "rsqrt1": 2, "rsqrt2": 3}
    summary = []

    for method in sorted(groups, key=lambda name: order.get(name, 99)):
        group = groups[method]
        seconds = [float(row["seconds"]) for row in group]
        seconds_median = median(seconds)
        n = int(group[0]["n"])

        summary.append(
            {
                "method": method,
                "n": n,
                "repeats": len(group),
                "seconds_median": seconds_median,
                "seconds_stdev": stdev(seconds),
                "values_per_second": n / seconds_median,
                "speedup_vs_sqrtf": sqrtf_median / seconds_median,
                "max_relative_error": max(
                    float(row["max_relative_error"]) for row in group
                ),
                "statuses": "|".join(sorted({row["status"] for row in group})),
            }
        )

    return summary


def write_csv(path: Path, summary: list[dict[str, object]]) -> None:
    fieldnames = [
        "method",
        "n",
        "repeats",
        "seconds_median",
        "seconds_stdev",
        "values_per_second",
        "speedup_vs_sqrtf",
        "max_relative_error",
        "statuses",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def markdown_table(summary: list[dict[str, object]]) -> str:
    lines = [
        "| Method | Repeats | Median s | Stdev s | Gvalues/s | Speedup vs sqrtf | Max relative error | Status |",
        "|:---|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for row in summary:
        lines.append(
            "| {method} | {repeats} | {seconds:.6f} | {stdev:.6f} | {gvalues:.3f} | {speedup:.3f} | {error:.6e} | {status} |".format(
                method=row["method"],
                repeats=row["repeats"],
                seconds=float(row["seconds_median"]),
                stdev=float(row["seconds_stdev"]),
                gvalues=float(row["values_per_second"]) / 1.0e9,
                speedup=float(row["speedup_vs_sqrtf"]),
                error=float(row["max_relative_error"]),
                status=row["statuses"],
            )
        )
    return "\n".join(lines)


def write_svg(path: Path, summary: list[dict[str, object]]) -> None:
    import math

    colors = {
        "sqrtf": "#2563eb",
        "rsqrt0": "#16a34a",
        "rsqrt1": "#ca8a04",
        "rsqrt2": "#dc2626",
    }
    speedup_max = max(22.0, max(float(row["speedup_vs_sqrtf"]) for row in summary) * 1.08)
    x0 = 122.0
    y0 = 112.0
    width = 770.0
    height = 330.0
    x_min = 1e-8
    x_max = 1e-3

    def sx(value: float) -> float:
        v = min(max(value, x_min), x_max)
        return x0 + (math.log10(v) - math.log10(x_min)) / (
            math.log10(x_max) - math.log10(x_min)
        ) * width

    def sy(value: float) -> float:
        return y0 + height - value / speedup_max * height

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="530" viewBox="0 0 1040 530">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial, Helvetica, sans-serif; fill:#111827;}</style>',
        '<text x="520" y="34" text-anchor="middle" font-size="21" font-weight="700">Isolated reciprocal square-root operation</text>',
        '<text x="520" y="58" text-anchor="middle" font-size="12" fill="#4b5563">Operation-level AVX rsqrt microbenchmark, not full-solver speedup. Left/up is better.</text>',
        f'<line x1="{x0}" y1="{y0 + height}" x2="{x0 + width}" y2="{y0 + height}" stroke="#222"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + height}" stroke="#222"/>',
        f'<text x="{x0 + width / 2:.1f}" y="{y0 + height + 52:.1f}" text-anchor="middle" font-size="12">maximum relative error, log scale</text>',
        f'<text x="{x0 - 62:.1f}" y="{y0 + height / 2:.1f}" transform="rotate(-90 {x0 - 62:.1f},{y0 + height / 2:.1f})" text-anchor="middle" font-size="12">operation speedup vs scalar sqrtf</text>',
        f'<line x1="{x0}" y1="{sy(1.0):.1f}" x2="{x0 + width}" y2="{sy(1.0):.1f}" stroke="#9ca3af" stroke-dasharray="6 5"/>',
        f'<text x="{x0 + width - 4:.1f}" y="{sy(1.0) - 8:.1f}" text-anchor="end" font-size="11" fill="#6b7280">sqrtf baseline 1x</text>',
        "</svg>",
    ]

    for exponent in range(-8, -2):
        value = 10.0 ** exponent
        px = sx(value)
        parts.insert(
            -1,
            f'<line x1="{px:.1f}" y1="{y0}" x2="{px:.1f}" y2="{y0 + height}" stroke="#e5e7eb"/>',
        )
        parts.insert(
            -1,
            f'<text x="{px:.1f}" y="{y0 + height + 22:.1f}" text-anchor="middle" font-size="10">1e{exponent}</text>',
        )

    for value in [0.0, 1.0, 5.0, 10.0, 15.0, 20.0]:
        if value > speedup_max:
            continue
        py = sy(value)
        label = f"{value:.0f}x" if value > 0.0 else "0"
        parts.insert(
            -1,
            f'<line x1="{x0}" y1="{py:.1f}" x2="{x0 + width}" y2="{py:.1f}" stroke="#e5e7eb"/>',
        )
        parts.insert(
            -1,
            f'<text x="{x0 - 8:.1f}" y="{py + 4:.1f}" text-anchor="end" font-size="10">{label}</text>',
        )

    label_offsets = {
        "sqrtf": (10, 18),
        "rsqrt0": (10, -10),
        "rsqrt1": (10, -10),
        "rsqrt2": (10, 18),
    }
    for row in summary:
        method = str(row["method"])
        px = sx(float(row["max_relative_error"]))
        py = sy(float(row["speedup_vs_sqrtf"]))
        dx, dy = label_offsets.get(method, (10, -10))
        parts.insert(
            -1,
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6.5" fill="{colors[method]}" stroke="white" stroke-width="1.5"/>',
        )
        parts.insert(
            -1,
            f'<text x="{px + dx:.1f}" y="{py + dy:.1f}" font-size="12" font-weight="700">{method}</text>',
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze isolated reciprocal-square-root kernel CSV files."
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
