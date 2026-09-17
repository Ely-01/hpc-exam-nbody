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


def bar(x: float, y: float, width: float, height: float, color: str) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" fill="{color}"/>'


def write_svg(path: Path, summary: list[dict[str, object]]) -> None:
    colors = {
        "sqrtf": "#2563eb",
        "rsqrt0": "#16a34a",
        "rsqrt1": "#ca8a04",
        "rsqrt2": "#dc2626",
    }
    methods = [str(row["method"]) for row in summary]
    row_by_method = {str(row["method"]): row for row in summary}
    speedup_max = max(1.2, max(float(row["speedup_vs_sqrtf"]) for row in summary) * 1.2)
    error_max = max(float(row["max_relative_error"]) for row in summary)
    error_max = max(error_max, 1e-8)

    def draw_panel(metric: str, title: str, y_label: str,
                   x0: float, y0: float, width: float, height: float,
                   y_max: float, log_hint: bool = False) -> str:
        group_width = width / len(methods)
        bar_width = group_width * 0.48

        def sy(value: float) -> float:
            if log_hint:
                floor = 1e-9
                import math

                v = max(value, floor)
                lo = math.log10(floor)
                hi = math.log10(y_max)
                return y0 + height - (math.log10(v) - lo) / (hi - lo) * height
            return y0 + height - value / y_max * height

        parts = [
            f'<text x="{x0 + width / 2:.1f}" y="{y0 - 18:.1f}" text-anchor="middle" font-size="16" font-weight="700">{title}</text>',
            f'<line x1="{x0}" y1="{y0 + height}" x2="{x0 + width}" y2="{y0 + height}" stroke="#222"/>',
            f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + height}" stroke="#222"/>',
            f'<text x="{x0 - 52:.1f}" y="{y0 + height / 2:.1f}" transform="rotate(-90 {x0 - 52:.1f},{y0 + height / 2:.1f})" text-anchor="middle" font-size="12">{y_label}</text>',
        ]

        for i in range(6):
            if log_hint:
                value = 10.0 ** (-9 + i * (9 + __import__("math").log10(y_max)) / 5)
                label = f"{value:.0e}"
            else:
                value = y_max * i / 5
                label = f"{value:.2g}"
            py = sy(value)
            parts.append(f'<line x1="{x0}" y1="{py:.1f}" x2="{x0 + width}" y2="{py:.1f}" stroke="#e5e7eb"/>')
            parts.append(f'<text x="{x0 - 8:.1f}" y="{py + 4:.1f}" text-anchor="end" font-size="10">{label}</text>')

        for index, method in enumerate(methods):
            row = row_by_method[method]
            value = float(row[metric])
            center = x0 + index * group_width + group_width / 2
            py = sy(value)
            parts.append(bar(center - bar_width / 2, py, bar_width, y0 + height - py, colors[method]))
            parts.append(f'<text x="{center:.1f}" y="{y0 + height + 22:.1f}" text-anchor="middle" font-size="11">{method}</text>')

        return "\n".join(parts)

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="530" viewBox="0 0 1040 530">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial, Helvetica, sans-serif; fill:#111827;}</style>',
        '<text x="520" y="34" text-anchor="middle" font-size="21" font-weight="700">SIMD reciprocal square-root microbenchmark</text>',
        '<text x="520" y="58" text-anchor="middle" font-size="12" fill="#4b5563">Isolated float inverse-square-root path: scalar sqrtf baseline vs AVX rsqrt plus Newton refinements.</text>',
        draw_panel(
            "speedup_vs_sqrtf",
            "Throughput speedup",
            "x vs sqrtf",
            86,
            116,
            390,
            300,
            speedup_max,
        ),
        draw_panel(
            "max_relative_error",
            "Maximum relative error",
            "relative error",
            596,
            116,
            360,
            300,
            error_max * 1.2,
            log_hint=True,
        ),
        "</svg>",
    ]
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
