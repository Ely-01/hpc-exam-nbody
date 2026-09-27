#!/usr/bin/env python3
"""Summarize and plot reciprocal-square-root accuracy/performance trade-offs."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


PREFERRED_VARIANTS = [
    "libm",
    "rsqrt1",
    "rsqrt2",
    "rsqrt3",
    "rsqrt512-0",
    "rsqrt512-1",
    "rsqrt512-2",
]

COLORS = {
    "libm": "#2563eb",
    "rsqrt1": "#16a34a",
    "rsqrt2": "#ca8a04",
    "rsqrt3": "#dc2626",
    "rsqrt512-0": "#9333ea",
    "rsqrt512-1": "#16a34a",
    "rsqrt512-2": "#ca8a04",
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


def svg_escape(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def variant_label(row: dict[str, str]) -> str:
    mode = row["inv_sqrt"]
    kernel = row.get("force_kernel", "direct")
    if kernel == "direct":
        return mode
    if kernel.startswith("direct-rsqrt512-"):
        return kernel.removeprefix("direct-")
    return f"{kernel}/{mode}"


def variant_sort_key(variant: str) -> tuple[int, str]:
    if variant in PREFERRED_VARIANTS:
        return (PREFERRED_VARIANTS.index(variant), variant)
    return (len(PREFERRED_VARIANTS), variant)


def ordered_variants(rows: list[dict[str, object]]) -> list[str]:
    variants = {str(row["variant"]) for row in rows}
    return sorted(variants, key=variant_sort_key)


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["threads"]), variant_label(row))].append(row)

    if not groups:
        raise ValueError("input CSV does not contain benchmark rows")

    libm_force = {
        threads: median([float(row["force_seconds"]) for row in group])
        for (threads, variant), group in groups.items()
        if variant == "libm"
    }
    libm_drift = {
        threads: max(float(row["max_relative_energy_drift"]) for row in group)
        for (threads, variant), group in groups.items()
        if variant == "libm"
    }

    if not libm_force:
        raise ValueError("input CSV does not contain libm baseline rows")

    summary = []
    for threads, variant in sorted(groups, key=lambda key: (key[0], variant_sort_key(key[1]))):
        group = groups[(threads, variant)]
        force_values = [float(row["force_seconds"]) for row in group]
        total_values = [float(row["total_seconds"]) for row in group]
        energy_values = [float(row["energy_seconds"]) for row in group]
        force_median = median(force_values)
        drift = max(float(row["max_relative_energy_drift"]) for row in group)
        baseline_force = libm_force[threads]
        baseline_drift = libm_drift[threads]

        summary.append(
            {
                "variant": variant,
                "inv_sqrt": group[0]["inv_sqrt"],
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
        "variant",
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
        "| Threads | Variant | Repeats | Force median s | Force stdev s | Speedup vs libm | Max drift | Drift change vs libm | Status |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for row in summary:
        lines.append(
            "| {threads} | {mode} | {repeats} | {force:.6f} | {stdev:.6f} | {speedup:.3f} | {drift:.6e} | {change:.3f}% | {status} |".format(
                threads=row["threads"],
                mode=row["variant"],
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
    variants = ordered_variants(rows)
    threads = sorted({int(row["threads"]) for row in rows})
    row_by_key = {(int(row["threads"]), str(row["variant"])): row for row in rows}
    group_width = width / len(threads)
    bar_width = group_width / (len(variants) + 1.2)

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
        for mode_index, variant in enumerate(variants):
            row = row_by_key.get((thread, variant))
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
                    COLORS.get(variant, "#6b7280"),
                )
            )

    legend_x = x0 + width - 112
    for index, variant in enumerate(variants):
        ly = y0 + 18 + index * 18
        parts.append(bar(legend_x, ly - 10, 10, 10, COLORS.get(variant, "#6b7280")))
        parts.append(f'<text x="{legend_x + 16:.1f}" y="{ly:.1f}" font-size="11">{variant}</text>')

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
    variants = ordered_variants(rows)
    threads = sorted({int(row["threads"]) for row in rows})
    row_by_key = {(int(row["threads"]), str(row["variant"])): row for row in rows}
    group_width = width / len(threads)
    bar_width = group_width / (len(variants) + 1.2)

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
        for mode_index, variant in enumerate(variants):
            row = row_by_key.get((thread, variant))
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
                    COLORS.get(variant, "#6b7280"),
                )
            )

    legend_x = x0 + width - 112
    for index, variant in enumerate(variants):
        ly = y0 + 18 + index * 18
        parts.append(bar(legend_x, ly - 10, 10, 10, COLORS.get(variant, "#6b7280")))
        parts.append(f'<text x="{legend_x + 16:.1f}" y="{ly:.1f}" font-size="11">{variant}</text>')

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
        '<text x="520" y="58" text-anchor="middle" font-size="12" fill="#4b5563">Approximate reciprocal-square-root variants are compared against the direct libm baseline.</text>',
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


def read_accuracy(path: Path) -> dict[str, dict[str, float]]:
    accuracy: dict[str, dict[str, float]] = {}
    for row in read_rows(path):
        accuracy[row["variant"]] = {
            "max_relative_accel_error": float(row["max_relative_accel_error"]),
            "rms_relative_accel_error": float(row["rms_relative_accel_error"]),
        }
    return accuracy


def write_speed_accuracy_svg(
    path: Path,
    summary: list[dict[str, object]],
    accuracy: dict[str, dict[str, float]],
) -> None:
    threads = sorted({int(row["threads"]) for row in summary})
    selected_thread = threads[0]
    rows = [row for row in summary if int(row["threads"]) == selected_thread]
    if not rows:
        raise ValueError(f"no rsqrt summary rows for thread count {selected_thread}")

    points = []
    for row in rows:
        variant = str(row["variant"])
        if variant == "libm" or variant not in accuracy:
            continue
        max_error = accuracy[variant]["max_relative_accel_error"]
        if max_error <= 0.0:
            continue
        points.append(
            {
                "variant": variant,
                "speedup": float(row["speedup_vs_libm"]),
                "max_error": max_error,
                "rms_error": accuracy[variant]["rms_relative_accel_error"],
            }
        )

    if not points:
        raise ValueError("no positive accuracy-error values available for plotting")

    x_min_log = math.floor(math.log10(min(point["max_error"] for point in points))) - 0.4
    x_max_log = math.ceil(math.log10(max(point["max_error"] for point in points))) + 0.4
    y_min = 0.0
    y_max = max(1.2, max(point["speedup"] for point in points) * 1.18)

    width = 1040
    height = 560
    left = 118
    right = 72
    top = 112
    bottom = 92
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(value: float) -> float:
        return left + (math.log10(value) - x_min_log) / (x_max_log - x_min_log) * plot_w

    def sy(value: float) -> float:
        return top + plot_h - (value - y_min) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial, Helvetica, sans-serif; fill:#111827;}</style>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-size="23" font-weight="700">AVX-512 reciprocal square-root trade-off</text>',
        f'<text x="{width / 2}" y="62" text-anchor="middle" font-size="13" fill="#4b5563">Force-kernel speedup versus direct acceleration error, {selected_thread} OpenMP thread(s). Left/up is better.</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222"/>',
        f'<text x="{left + plot_w / 2}" y="{height - 28}" text-anchor="middle" font-size="13">maximum relative acceleration error vs libm, log scale</text>',
        f'<text x="34" y="{top + plot_h / 2}" transform="rotate(-90 34,{top + plot_h / 2})" text-anchor="middle" font-size="13">force-kernel speedup vs libm</text>',
        f'<line x1="{left}" y1="{sy(1.0):.1f}" x2="{left + plot_w}" y2="{sy(1.0):.1f}" stroke="#9ca3af" stroke-dasharray="6 5"/>',
        f'<text x="{left + plot_w - 4}" y="{sy(1.0) - 8:.1f}" text-anchor="end" font-size="12" fill="#6b7280">libm baseline 1x</text>',
    ]

    first_tick = math.ceil(x_min_log)
    last_tick = math.floor(x_max_log)
    for exponent in range(first_tick, last_tick + 1):
        value = 10.0 ** exponent
        px = sx(value)
        parts.append(f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{top + plot_h}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{px:.1f}" y="{top + plot_h + 24}" text-anchor="middle" font-size="11">1e{exponent}</text>')

    for i in range(6):
        value = y_min + (y_max - y_min) * i / 5
        py = sy(value)
        parts.append(f'<line x1="{left}" y1="{py:.1f}" x2="{left + plot_w}" y2="{py:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left - 10}" y="{py + 4:.1f}" text-anchor="end" font-size="11">{value:.1f}x</text>')

    for point in points:
        variant = str(point["variant"])
        px = sx(float(point["max_error"]))
        py = sy(float(point["speedup"]))
        color = COLORS.get(variant, "#6b7280")
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="7.0" fill="{color}" stroke="white" stroke-width="1.5"/>')
        label_dx = 14 if variant != "rsqrt512-2" else -14
        anchor = "start" if label_dx > 0 else "end"
        parts.append(
            f'<text x="{px + label_dx:.1f}" y="{py - 12:.1f}" text-anchor="{anchor}" font-size="13" font-weight="700">{svg_escape(variant)}</text>'
        )
        parts.append(
            f'<text x="{px + label_dx:.1f}" y="{py + 6:.1f}" text-anchor="{anchor}" font-size="11" fill="#4b5563">{point["speedup"]:.2f}x, max err {point["max_error"]:.1e}</text>'
        )

    notes_x = left + 18
    notes_y = top + 24
    parts.extend(
        [
            f'<rect x="{notes_x - 10}" y="{notes_y - 18}" width="292" height="58" fill="white" opacity="0.86" stroke="#e5e7eb"/>',
            f'<text x="{notes_x}" y="{notes_y}" font-size="12" fill="#374151">One Newton step gives the best balance:</text>',
            f'<text x="{notes_x}" y="{notes_y + 18}" font-size="12" fill="#374151">large speedup with ~1e-9 RMS acceleration error.</text>',
        ]
    )

    parts.append("</svg>")
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


def default_accuracy_csv(args: argparse.Namespace) -> Path:
    candidates = []
    if args.csv is not None:
        candidates.append(args.csv.parent / "rsqrt_accuracy_summary.csv")
    if args.markdown is not None:
        candidates.append(args.markdown.parent / "rsqrt_accuracy_summary.csv")
    candidates.append(Path("report/tables/rsqrt_accuracy_summary.csv"))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


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
        accuracy_csv = default_accuracy_csv(args)
        if not accuracy_csv.exists():
            raise SystemExit(
                f"missing acceleration-accuracy table '{accuracy_csv}'; "
                "run scripts/benchmark/benchmark_rsqrt_accuracy.sh first"
            )
        write_speed_accuracy_svg(
            args.svg, summary, read_accuracy(accuracy_csv)
        )


if __name__ == "__main__":
    main()
