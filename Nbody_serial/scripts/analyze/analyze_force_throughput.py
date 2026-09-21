#!/usr/bin/env python3
"""Generate force-kernel throughput tables and plots from scaling summaries."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def f_float(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def f_short(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def summarize(rows: list[dict[str, str]], nsteps: int) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for row in rows:
        n = int(row["n"])
        ranks = int(row["ranks"])
        threads = int(row["threads"])
        workers = int(row["total_workers"])
        total_median_s = float(row["total_median_s"])
        total_stdev_s = float(row["total_stdev_s"])
        force_median_s = float(row["force_median_s"])
        force_stdev_s = float(row.get("force_stdev_s", "nan"))
        interactions = n * (n - 1) * (nsteps + 1)
        interactions_per_second = interactions / force_median_s
        ginteractions_per_second = interactions_per_second / 1.0e9
        force_fraction = force_median_s / total_median_s
        nominal_gflops = ginteractions_per_second * 20.0

        summary.append(
            {
                "backend": row["backend"],
                "mode": row["mode"],
                "n": n,
                "nsteps": nsteps,
                "ranks": ranks,
                "threads": threads,
                "workers": workers,
                "total_median_s": total_median_s,
                "total_stdev_s": total_stdev_s,
                "force_median_s": force_median_s,
                "force_stdev_s": force_stdev_s,
                "force_fraction": force_fraction,
                "interactions": interactions,
                "interactions_per_second": interactions_per_second,
                "ginteractions_per_second": ginteractions_per_second,
                "nominal_gflops_s": nominal_gflops,
            }
        )

    summary.sort(key=lambda item: (int(item["workers"]), int(item["ranks"])))

    if summary:
        baseline = summary[0]
        baseline_rate = float(baseline["ginteractions_per_second"])
        baseline_workers = int(baseline["workers"])
        for row in summary:
            scale = int(row["workers"]) / baseline_workers
            row["ideal_ginteractions_per_second"] = baseline_rate * scale
            row["throughput_efficiency"] = (
                float(row["ginteractions_per_second"])
                / float(row["ideal_ginteractions_per_second"])
            )

    return summary


def write_csv(path: Path, summary: list[dict[str, object]]) -> None:
    fieldnames = [
        "backend",
        "mode",
        "n",
        "nsteps",
        "ranks",
        "threads",
        "workers",
        "total_median_s",
        "total_stdev_s",
        "force_median_s",
        "force_stdev_s",
        "force_fraction",
        "interactions",
        "interactions_per_second",
        "ginteractions_per_second",
        "ideal_ginteractions_per_second",
        "throughput_efficiency",
        "nominal_gflops_s",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)


def markdown_table(summary: list[dict[str, object]]) -> str:
    lines = [
        "| Ranks | Threads | Workers | Force median +/- sigma s | Force / total | Ginteraction/s | Ideal Ginteraction/s | Throughput efficiency | Nominal GFLOP/s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        force = f"{f_float(float(row['force_median_s']))} +/- {f_float(float(row['force_stdev_s']))}"
        lines.append(
            "| {ranks} | {threads} | {workers} | {force} | {frac:.2%} | {gint} | {ideal} | {eff:.3f} | {gflops} |".format(
                ranks=row["ranks"],
                threads=row["threads"],
                workers=row["workers"],
                force=force,
                frac=float(row["force_fraction"]),
                gint=f_short(float(row["ginteractions_per_second"])),
                ideal=f_short(float(row["ideal_ginteractions_per_second"])),
                eff=float(row["throughput_efficiency"]),
                gflops=f_short(float(row["nominal_gflops_s"]), 2),
            )
        )

    lines.extend(
        [
            "",
            "Throughput uses ordered source-target particle interactions:",
            "",
            "`N * (N - 1) * (nsteps + 1) / force_median_s`.",
            "",
            "The nominal GFLOP/s column assumes 20 floating-point operations per interaction and is a relative throughput indicator, not a hardware-counter measurement.",
        ]
    )
    return "\n".join(lines)


def write_markdown(path: Path, summary: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_table(summary) + "\n")


def nice_max(value: float) -> float:
    if value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    base = 10**exponent
    scaled = value / base
    if scaled <= 1:
        nice = 1
    elif scaled <= 2:
        nice = 2
    elif scaled <= 5:
        nice = 5
    else:
        nice = 10
    return nice * base


def polyline(points: list[tuple[float, float]], color: str, dash: bool = False) -> str:
    dash_attr = ' stroke-dasharray="6 5"' if dash else ""
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return (
        f'<polyline points="{coords}" fill="none" stroke="{color}" '
        f'stroke-width="2.5"{dash_attr}/>'
    )


def write_svg(path: Path, summary: list[dict[str, object]]) -> None:
    if not summary:
        raise SystemExit("empty summary")

    width = 980
    height = 560
    left = 90
    right = 40
    top = 95
    bottom = 85
    plot_w = width - left - right
    plot_h = height - top - bottom

    workers = [int(row["workers"]) for row in summary]
    measured = [float(row["ginteractions_per_second"]) for row in summary]
    ideal = [float(row["ideal_ginteractions_per_second"]) for row in summary]
    y_max = nice_max(max(max(measured), max(ideal)) * 1.10)
    x_min = min(workers)
    x_max = max(workers)

    def x_scale(value: float) -> float:
        if x_max == x_min:
            return left + plot_w / 2
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def y_scale(value: float) -> float:
        return top + plot_h - value / y_max * plot_h

    measured_points = [
        (x_scale(w), y_scale(v)) for w, v in zip(workers, measured, strict=True)
    ]
    ideal_points = [(x_scale(w), y_scale(v)) for w, v in zip(workers, ideal, strict=True)]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-size="23" font-weight="700">Force-kernel interaction throughput</text>',
        f'<text x="{width / 2}" y="62" text-anchor="middle" font-size="13">Native strong scaling, ordered interactions, N = {summary[0]["n"]}, steps = {summary[0]["nsteps"]}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#9ca3af"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#9ca3af"/>',
    ]

    for tick in range(0, 6):
        value = y_max * tick / 5
        y = y_scale(value)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>'
        )
        parts.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="11">{value:.1f}</text>'
        )

    for w in workers:
        x = x_scale(w)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 5}" stroke="#374151"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{top + plot_h + 24}" text-anchor="middle" font-size="11">{w}</text>'
        )

    parts.append(polyline(ideal_points, "#9ca3af", dash=True))
    parts.append(polyline(measured_points, "#2563eb"))

    for (x, y), value in zip(measured_points, measured, strict=True):
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="#2563eb"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{y - 10:.1f}" text-anchor="middle" font-size="10">{value:.2f}</text>'
        )

    parts.extend(
        [
            f'<text x="{left + plot_w / 2}" y="{height - 30}" text-anchor="middle" font-size="13">Total workers / MPI ranks</text>',
            f'<text x="25" y="{top + plot_h / 2}" transform="rotate(-90 25 {top + plot_h / 2})" text-anchor="middle" font-size="13">Ginteraction/s</text>',
            f'<line x1="{width - 265}" y1="105" x2="{width - 220}" y2="105" stroke="#2563eb" stroke-width="2.5"/>',
            f'<circle cx="{width - 242}" cy="105" r="4.5" fill="#2563eb"/>',
            f'<text x="{width - 210}" y="109" font-size="12">measured throughput</text>',
            f'<line x1="{width - 265}" y1="128" x2="{width - 220}" y2="128" stroke="#9ca3af" stroke-width="2.5" stroke-dasharray="6 5"/>',
            f'<text x="{width - 210}" y="132" font-size="12">ideal linear from 1 rank</text>',
            "</svg>",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute force-kernel interaction throughput from scaling summaries."
    )
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--nsteps", type=int, required=True)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--svg", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize(read_rows(args.summary_csv), args.nsteps)
    print(markdown_table(summary))
    if args.csv is not None:
        write_csv(args.csv, summary)
    if args.markdown is not None:
        write_markdown(args.markdown, summary)
    if args.svg is not None:
        write_svg(args.svg, summary)


if __name__ == "__main__":
    main()
