#!/usr/bin/env python3
"""Create SVG speedup/efficiency plots from summarize_scaling_csv.py output."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def nice_max(value: float) -> float:
    if value <= 1.0:
        return 1.0
    magnitude = 10 ** (len(str(int(value))) - 1)
    for factor in (1, 2, 5, 10):
        candidate = factor * magnitude
        if candidate >= value:
            return float(candidate)
    return float(10 * magnitude)


def polyline(points: list[tuple[float, float]], color: str) -> str:
    coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5"/>'


def circles(points: list[tuple[float, float]], color: str) -> str:
    return "\n".join(
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>'
        for x, y in points
    )


def draw_panel(
    x_values: list[float],
    y_values: list[float],
    ideal_values: list[float],
    x_tick_labels: list[str],
    x_label: str,
    y_label: str,
    title: str,
    x0: float,
    y0: float,
    width: float,
    height: float,
    y_max: float,
) -> str:
    x_min = min(x_values)
    x_max = max(x_values)
    if x_min == x_max:
        x_max = x_min + 1.0

    def sx(x: float) -> float:
        return x0 + (x - x_min) / (x_max - x_min) * width

    def sy(y: float) -> float:
        return y0 + height - y / y_max * height

    actual = [(sx(x), sy(y)) for x, y in zip(x_values, y_values)]
    ideal = [(sx(x), sy(y)) for x, y in zip(x_values, ideal_values)]

    parts = [
        f'<text x="{x0 + width / 2:.1f}" y="{y0 - 16:.1f}" text-anchor="middle" font-size="16" font-weight="700">{title}</text>',
        f'<line x1="{x0}" y1="{y0 + height}" x2="{x0 + width}" y2="{y0 + height}" stroke="#222"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + height}" stroke="#222"/>',
        f'<text x="{x0 + width / 2:.1f}" y="{y0 + height + 58:.1f}" text-anchor="middle" font-size="13">{x_label}</text>',
        f'<text x="{x0 - 48:.1f}" y="{y0 + height / 2:.1f}" transform="rotate(-90 {x0 - 48:.1f},{y0 + height / 2:.1f})" text-anchor="middle" font-size="13">{y_label}</text>',
    ]

    for i in range(6):
        y = y_max * i / 5
        py = sy(y)
        parts.append(f'<line x1="{x0}" y1="{py:.1f}" x2="{x0 + width}" y2="{py:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{x0 - 8:.1f}" y="{py + 4:.1f}" text-anchor="end" font-size="11">{y:.2g}</text>')

    for index, x in enumerate(x_values):
        px = sx(x)
        parts.append(f'<line x1="{px:.1f}" y1="{y0 + height}" x2="{px:.1f}" y2="{y0 + height + 5}" stroke="#222"/>')
        label_lines = x_tick_labels[index].split("\\n")
        for line_index, label_line in enumerate(label_lines):
            parts.append(f'<text x="{px:.1f}" y="{y0 + height + 22 + 14 * line_index:.1f}" text-anchor="middle" font-size="11">{label_line}</text>')

    parts.append(polyline(ideal, "#9ca3af"))
    parts.append(polyline(actual, "#2563eb"))
    parts.append(circles(actual, "#2563eb"))
    parts.append(f'<text x="{x0 + width - 80:.1f}" y="{y0 + 22:.1f}" font-size="12" fill="#2563eb">measured</text>')
    parts.append(f'<text x="{x0 + width - 80:.1f}" y="{y0 + 40:.1f}" font-size="12" fill="#6b7280">ideal</text>')
    return "\n".join(parts)


def draw_panel_multi(
    x_values: list[float],
    series: list[tuple[str, str, dict[float, float]]],
    ideal_values: list[float],
    x_tick_labels: list[str],
    x_label: str,
    y_label: str,
    title: str,
    x0: float,
    y0: float,
    width: float,
    height: float,
    y_max: float,
) -> str:
    x_min = min(x_values)
    x_max = max(x_values)
    if x_min == x_max:
        x_max = x_min + 1.0

    def sx(x: float) -> float:
        return x0 + (x - x_min) / (x_max - x_min) * width

    def sy(y: float) -> float:
        return y0 + height - y / y_max * height

    ideal = [(sx(x), sy(y)) for x, y in zip(x_values, ideal_values)]

    parts = [
        f'<text x="{x0 + width / 2:.1f}" y="{y0 - 16:.1f}" text-anchor="middle" font-size="16" font-weight="700">{title}</text>',
        f'<line x1="{x0}" y1="{y0 + height}" x2="{x0 + width}" y2="{y0 + height}" stroke="#222"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + height}" stroke="#222"/>',
        f'<text x="{x0 + width / 2:.1f}" y="{y0 + height + 58:.1f}" text-anchor="middle" font-size="13">{x_label}</text>',
        f'<text x="{x0 - 48:.1f}" y="{y0 + height / 2:.1f}" transform="rotate(-90 {x0 - 48:.1f},{y0 + height / 2:.1f})" text-anchor="middle" font-size="13">{y_label}</text>',
    ]

    for i in range(6):
        y = y_max * i / 5
        py = sy(y)
        parts.append(f'<line x1="{x0}" y1="{py:.1f}" x2="{x0 + width}" y2="{py:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{x0 - 8:.1f}" y="{py + 4:.1f}" text-anchor="end" font-size="11">{y:.2g}</text>')

    for index, x in enumerate(x_values):
        px = sx(x)
        parts.append(f'<line x1="{px:.1f}" y1="{y0 + height}" x2="{px:.1f}" y2="{y0 + height + 5}" stroke="#222"/>')
        label_lines = x_tick_labels[index].split("\\n")
        for line_index, label_line in enumerate(label_lines):
            parts.append(f'<text x="{px:.1f}" y="{y0 + height + 22 + 14 * line_index:.1f}" text-anchor="middle" font-size="11">{label_line}</text>')

    parts.append(polyline(ideal, "#9ca3af"))
    for label, color, values_by_x in series:
        points = [
            (sx(x), sy(values_by_x[x]))
            for x in x_values
            if x in values_by_x
        ]
        if points:
            parts.append(polyline(points, color))
            parts.append(circles(points, color))

    legend_x = x0 + width - 104
    legend_y = y0 + 20
    parts.append(f'<text x="{legend_x:.1f}" y="{legend_y:.1f}" font-size="12" fill="#6b7280">ideal</text>')
    for index, (label, color, _) in enumerate(series):
        y = legend_y + 18 * (index + 1)
        parts.append(f'<circle cx="{legend_x - 10:.1f}" cy="{y - 4:.1f}" r="4" fill="{color}"/>')
        parts.append(f'<text x="{legend_x:.1f}" y="{y:.1f}" font-size="12" fill="{color}">{label}</text>')

    return "\n".join(parts)


def make_svg(rows: list[dict[str, str]], output: Path) -> None:
    mode = rows[0]["mode"]
    rows = sorted(
        rows,
        key=lambda row: (
            row.get("backend", "native"),
            float(row["total_workers"]),
            int(row["ranks"]),
            int(row["threads"]),
        ),
    )
    all_workers = sorted({float(row["total_workers"]) for row in rows})
    base_workers = min(all_workers)
    backends = sorted({row.get("backend", "native") for row in rows})
    colors = {
        "native": "#2563eb",
        "container": "#dc2626",
    }

    ranks = [int(row["ranks"]) for row in rows]
    threads = [int(row["threads"]) for row in rows]
    workers = [float(row["total_workers"]) for row in rows]
    has_mpi = any(rank > 1 for rank in ranks)
    has_openmp = any(thread > 1 for thread in threads)

    if len(backends) > 1:
        backend_label = "Native vs container"
    elif has_mpi and has_openmp:
        backend_label = "Hybrid MPI+OpenMP"
    elif has_openmp:
        backend_label = "OpenMP-only"
    else:
        backend_label = "MPI-only"

    show_rank_thread = has_openmp or any(rank != int(worker) for rank, worker in zip(ranks, workers))
    if show_rank_thread:
        labels_by_worker = {}
        for row in rows:
            worker = float(row["total_workers"])
            labels_by_worker.setdefault(
                worker,
                f"{int(worker)}\\n{int(row['ranks'])}x{int(row['threads'])}",
            )
        x_tick_labels = [labels_by_worker[worker] for worker in all_workers]
        x_label = "Total workers (rank x thread)"
    else:
        x_tick_labels = [str(int(worker)) for worker in all_workers]
        x_label = "Total workers"

    def series_for(metric: str) -> list[tuple[str, str, dict[float, float]]]:
        output_series = []
        for backend in backends:
            values = {
                float(row["total_workers"]): float(row[metric])
                for row in rows
                if row.get("backend", "native") == backend
            }
            output_series.append((backend, colors.get(backend, "#16a34a"), values))
        return output_series

    if mode == "strong":
        ideal_speedup = [w / base_workers for w in all_workers]
        note = "Strong scaling: fixed N. Amdahl effects appear as the measured curve bends below ideal speedup."
        speedup_title = f"{backend_label} strong scaling speedup"
    else:
        ideal_speedup = [w / base_workers for w in all_workers]
        note = "Weak scaling: fixed Nlocal per MPI rank. Gustafson-style scaled speedup is shown; efficiency loss indicates communication and runtime overheads."
        speedup_title = f"{backend_label} weak scaling scaled speedup"

    ideal_efficiency = [1.0 for _ in all_workers]
    speedup_values = [float(row["speedup"]) for row in rows]
    efficiency_values = [float(row["parallel_efficiency"]) for row in rows]
    communication_fraction = [
        float(row.get("communication_fraction", "0")) for row in rows
    ]
    y_max_speedup = nice_max(max(max(speedup_values), max(ideal_speedup)) * 1.05)
    y_max_eff = 1.1

    if mode == "weak":
        svg_width = 1560
        panel_width = 390
    else:
        svg_width = 1100
        panel_width = 430
    svg_height = 560
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial, Helvetica, sans-serif; fill:#111827;}</style>',
        f'<text x="{svg_width / 2}" y="34" text-anchor="middle" font-size="21" font-weight="700">{speedup_title}</text>',
        f'<text x="{svg_width / 2}" y="58" text-anchor="middle" font-size="12" fill="#4b5563">{note}</text>',
        draw_panel_multi(
            all_workers,
            series_for("speedup"),
            ideal_speedup,
            x_tick_labels,
            x_label,
            "Speedup",
            "Speedup",
            82,
            112,
            panel_width,
            330,
            y_max_speedup,
        ),
        draw_panel_multi(
            all_workers,
            series_for("parallel_efficiency"),
            ideal_efficiency,
            x_tick_labels,
            x_label,
            "Parallel efficiency",
            "Efficiency",
            622 if mode == "strong" else 570,
            112,
            panel_width,
            330,
            y_max_eff,
        ),
    ]

    if mode == "weak":
        parts.append(
            draw_panel_multi(
                all_workers,
                series_for("communication_fraction"),
                [0.0 for _ in all_workers],
                x_tick_labels,
                x_label,
                "Communication / total",
                "Communication fraction",
                1058,
                112,
                panel_width,
                330,
                max(0.15, nice_max(max(communication_fraction) * 1.15)),
            )
        )

    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create SVG speedup/efficiency plots from scaling summary CSV."
    )
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.summary_csv)
    if not rows:
        raise SystemExit("empty summary CSV")
    make_svg(rows, args.output)


if __name__ == "__main__":
    main()
