#!/usr/bin/env python3
"""Compare blocking and non-blocking MPI ring benchmark CSV files."""

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


def row_key(row: dict[str, str]) -> tuple[str, int, int, int, str, str]:
    return (
        row.get("mode", "hybrid"),
        int(row["n"]),
        int(row["nsteps"]),
        int(row["ranks"]),
        row["threads"],
        row.get("dt", ""),
    )


def group_rows(rows: list[dict[str, str]]) -> dict[tuple[str, int, int, int, str, str], list[dict[str, str]]]:
    groups: dict[tuple[str, int, int, int, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row_key(row)].append(row)
    return groups


def summarize_pair(
    blocking_rows: list[dict[str, str]],
    overlap_rows: list[dict[str, str]],
) -> dict[str, object]:
    first = blocking_rows[0]
    block_total_values = [float(row["total_seconds"]) for row in blocking_rows]
    over_total_values = [float(row["total_seconds"]) for row in overlap_rows]
    block_force_values = [float(row["force_seconds"]) for row in blocking_rows]
    over_force_values = [float(row["force_seconds"]) for row in overlap_rows]
    block_comm_values = [float(row["communication_seconds"]) for row in blocking_rows]
    over_comm_values = [float(row["communication_seconds"]) for row in overlap_rows]
    block_total = median(block_total_values)
    over_total = median(over_total_values)
    block_force = median(block_force_values)
    over_force = median(over_force_values)
    block_comm = median(block_comm_values)
    over_comm = median(over_comm_values)
    hidden_comm = max(0.0, block_comm - over_comm)
    total_saved = block_total - over_total
    overlap_fraction = hidden_comm / block_comm if block_comm > 0.0 else 0.0
    realized_runtime_fraction = total_saved / block_comm if block_comm > 0.0 else 0.0
    napkin_total = block_total - min(block_comm, block_force)
    napkin_total = max(0.0, napkin_total)
    napkin_saved = block_total - napkin_total
    napkin_fraction = napkin_saved / block_comm if block_comm > 0.0 else 0.0
    total_speedup = block_total / over_total if over_total > 0.0 else 0.0

    return {
        "mode": first.get("mode", "hybrid"),
        "n": int(first["n"]),
        "nsteps": int(first["nsteps"]),
        "dt": first.get("dt", ""),
        "eps": first.get("eps", ""),
        "ranks": int(first["ranks"]),
        "threads": int(first["threads"]),
        "total_workers": int(first.get("total_workers", int(first["ranks"]) * int(first["threads"]))),
        "blocking_repeats": len(blocking_rows),
        "overlap_repeats": len(overlap_rows),
        "blocking_total_median_s": block_total,
        "overlap_total_median_s": over_total,
        "blocking_total_stdev_s": stdev(block_total_values),
        "overlap_total_stdev_s": stdev(over_total_values),
        "blocking_force_median_s": block_force,
        "overlap_force_median_s": over_force,
        "blocking_comm_median_s": block_comm,
        "overlap_exposed_comm_median_s": over_comm,
        "hidden_comm_s": hidden_comm,
        "overlap_fraction": overlap_fraction,
        "runtime_saved_s": total_saved,
        "runtime_saved_per_blocking_comm": realized_runtime_fraction,
        "napkin_runtime_saved_s": napkin_saved,
        "napkin_overlap_fraction": napkin_fraction,
        "total_speedup_blocking_over_overlap": total_speedup,
        "max_drift_blocking": max(float(row["max_relative_energy_drift"]) for row in blocking_rows),
        "max_drift_overlap": max(float(row["max_relative_energy_drift"]) for row in overlap_rows),
        "status_blocking": "|".join(sorted({row["status"] for row in blocking_rows})),
        "status_overlap": "|".join(sorted({row["status"] for row in overlap_rows})),
    }


def summarize(
    blocking_rows: list[dict[str, str]],
    overlap_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    blocking_groups = group_rows(blocking_rows)
    overlap_groups = group_rows(overlap_rows)
    common_keys = sorted(set(blocking_groups) & set(overlap_groups))
    if not common_keys:
        raise ValueError("no matching configurations found between blocking and overlap CSV files")

    return [
        summarize_pair(blocking_groups[key], overlap_groups[key])
        for key in common_keys
    ]


def write_csv(path: Path, summary: list[dict[str, object]]) -> None:
    fieldnames = [
        "mode",
        "n",
        "nsteps",
        "dt",
        "eps",
        "ranks",
        "threads",
        "total_workers",
        "blocking_repeats",
        "overlap_repeats",
        "blocking_total_median_s",
        "overlap_total_median_s",
        "blocking_total_stdev_s",
        "overlap_total_stdev_s",
        "blocking_force_median_s",
        "overlap_force_median_s",
        "blocking_comm_median_s",
        "overlap_exposed_comm_median_s",
        "hidden_comm_s",
        "overlap_fraction",
        "runtime_saved_s",
        "runtime_saved_per_blocking_comm",
        "napkin_runtime_saved_s",
        "napkin_overlap_fraction",
        "total_speedup_blocking_over_overlap",
        "max_drift_blocking",
        "max_drift_overlap",
        "status_blocking",
        "status_overlap",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def markdown_table(summary: list[dict[str, object]]) -> str:
    lines = [
        "| N | Ranks | Threads | Repeats | Blocking total s | Overlap total s | Blocking comm s | Exposed overlap comm s | Hidden comm | Overlap achieved | Runtime saved | Napkin hidden | Speedup | Status |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for row in summary:
        lines.append(
            "| {n} | {ranks} | {threads} | {repeats} | {bt:.6f} | {ot:.6f} | {bc:.6f} | {oc:.6f} | {hidden:.6f} | {frac:.1f}% | {saved:.6f} | {napkin:.1f}% | {speedup:.3f} | {status_b}/{status_o} |".format(
                n=row["n"],
                ranks=row["ranks"],
                threads=row["threads"],
                repeats=min(int(row["blocking_repeats"]), int(row["overlap_repeats"])),
                bt=float(row["blocking_total_median_s"]),
                ot=float(row["overlap_total_median_s"]),
                bc=float(row["blocking_comm_median_s"]),
                oc=float(row["overlap_exposed_comm_median_s"]),
                hidden=float(row["hidden_comm_s"]),
                frac=100.0 * float(row["overlap_fraction"]),
                saved=float(row["runtime_saved_s"]),
                napkin=100.0 * float(row["napkin_overlap_fraction"]),
                speedup=float(row["total_speedup_blocking_over_overlap"]),
                status_b=row["status_blocking"],
                status_o=row["status_overlap"],
            )
        )

    lines.extend(
        [
            "",
            "Interpretation:",
            "- `Hidden comm` is the reduction in exposed communication time: blocking communication median minus overlap-mode MPI post/wait median.",
            "- `Overlap achieved` is `hidden_comm / blocking_comm`; 100% would mean the communication cost is fully hidden by useful force work.",
            "- `Napkin hidden` is the optimistic bound from comparing blocking communication with force work. Real measurements are usually lower because MPI progress may require entering MPI, messages have startup/injection costs, OpenMP threads and MPI progress can compete for cores, and phase imbalance leaves some ranks waiting.",
            "- `Runtime saved` can be smaller than hidden communication because total time also includes energy diagnostics, reductions, scheduling noise, and any overhead introduced by non-blocking calls.",
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
    rows = sorted(summary, key=lambda row: (int(row["total_workers"]), int(row["ranks"]), int(row["threads"])))
    ranks = [int(row["ranks"]) for row in rows]
    blocking_comm = [float(row["blocking_comm_median_s"]) for row in rows]
    exposed_comm = [float(row["overlap_exposed_comm_median_s"]) for row in rows]
    overlap_values = [100.0 * float(row["overlap_fraction"]) for row in rows]
    speedups = [float(row["total_speedup_blocking_over_overlap"]) for row in rows]

    def nice_max(value: float) -> float:
        if value <= 0.0:
            return 1.0
        exponent = 10 ** int(f"{value:e}".split("e")[1])
        scaled = value / exponent
        if scaled <= 1.5:
            return 1.5 * exponent
        if scaled <= 2.5:
            return 2.5 * exponent
        if scaled <= 5.0:
            return 5.0 * exponent
        return 10.0 * exponent

    comm_y_min = 0.0
    comm_y_max = nice_max(max(blocking_comm + exposed_comm) * 1.08)
    speedup_y_min = min(0.995, min(speedups) - 0.003)
    speedup_y_max = max(1.008, max(speedups) + 0.003)
    if speedup_y_max <= speedup_y_min:
        speedup_y_max = speedup_y_min + 0.01

    def sx_for(index: int, x0: float, width: float) -> float:
        if len(rows) == 1:
            return x0 + width / 2.0
        return x0 + index / (len(rows) - 1) * width

    def sy_for(value: float, y0: float, height: float, y_min: float, y_max: float) -> float:
        return y0 + height - (value - y_min) / (y_max - y_min) * height

    def axes(x0: float, y0: float, width: float, height: float, y_min: float, y_max: float, title: str, ylabel: str) -> list[str]:
        parts = [
            f'<text x="{x0 + width / 2:.1f}" y="{y0 - 18:.1f}" text-anchor="middle" font-size="16" font-weight="700">{svg_escape(title)}</text>',
            f'<line x1="{x0}" y1="{y0 + height}" x2="{x0 + width}" y2="{y0 + height}" stroke="#222"/>',
            f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + height}" stroke="#222"/>',
            f'<text x="{x0 - 50:.1f}" y="{y0 + height / 2:.1f}" transform="rotate(-90 {x0 - 50:.1f},{y0 + height / 2:.1f})" text-anchor="middle" font-size="12">{svg_escape(ylabel)}</text>',
            f'<text x="{x0 + width / 2:.1f}" y="{y0 + height + 44:.1f}" text-anchor="middle" font-size="12">MPI ranks</text>',
        ]
        for i in range(6):
            value = y_min + (y_max - y_min) * i / 5
            py = sy_for(value, y0, height, y_min, y_max)
            parts.append(f'<line x1="{x0}" y1="{py:.1f}" x2="{x0 + width}" y2="{py:.1f}" stroke="#e5e7eb"/>')
            tick = f"{value:.3f}" if (y_max - y_min) < 0.05 else f"{value:.3g}"
            parts.append(f'<text x="{x0 - 8:.1f}" y="{py + 4:.1f}" text-anchor="end" font-size="10">{tick}</text>')
        for index, rank in enumerate(ranks):
            center = sx_for(index, x0, width)
            parts.append(f'<line x1="{center:.1f}" y1="{y0}" x2="{center:.1f}" y2="{y0 + height}" stroke="#f3f4f6"/>')
            parts.append(f'<text x="{center:.1f}" y="{y0 + height + 23:.1f}" text-anchor="middle" font-size="11">{rank}</text>')
        return parts

    def line_path(values: list[float], x0: float, y0: float, width: float, height: float, y_min: float, y_max: float) -> str:
        tokens = []
        for index, value in enumerate(values):
            px = sx_for(index, x0, width)
            py = sy_for(value, y0, height, y_min, y_max)
            tokens.append(f"{'M' if index == 0 else 'L'} {px:.1f} {py:.1f}")
        return " ".join(tokens)

    def line_points(values: list[float], x0: float, y0: float, width: float, height: float, y_min: float, y_max: float, color: str, label_values: list[str] | None = None) -> list[str]:
        parts = [
            f'<path d="{line_path(values, x0, y0, width, height, y_min, y_max)}" fill="none" stroke="{color}" stroke-width="2.4"/>'
        ]
        for index, value in enumerate(values):
            px = sx_for(index, x0, width)
            py = sy_for(value, y0, height, y_min, y_max)
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5.2" fill="{color}" stroke="white" stroke-width="1.2"/>')
            if label_values is not None:
                parts.append(f'<text x="{px:.1f}" y="{py - 10:.1f}" text-anchor="middle" font-size="10" fill="#374151">{svg_escape(label_values[index])}</text>')
        return parts

    left_x = 105.0
    right_x = 825.0
    y0 = 135.0
    panel_width = 575.0
    panel_height = 395.0

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="690" viewBox="0 0 1500 690">',
        '<rect width="1500" height="690" fill="white"/>',
        '<style>text{font-family:Arial, Helvetica, sans-serif; fill:#111827;}</style>',
        '<text x="750" y="38" text-anchor="middle" font-size="24" font-weight="700">MPI ring communication overlap</text>',
        '<text x="750" y="66" text-anchor="middle" font-size="13" fill="#4b5563">Non-blocking MPI reduces exposed communication only slightly; total runtime stays near 1x.</text>',
        *axes(left_x, y0, panel_width, panel_height, comm_y_min, comm_y_max, "Exposed ring communication time", "communication seconds"),
        *axes(right_x, y0, panel_width, panel_height, speedup_y_min, speedup_y_max, "Total runtime speedup", "blocking / overlap"),
        *line_points(blocking_comm, left_x, y0, panel_width, panel_height, comm_y_min, comm_y_max, "#2563eb"),
        *line_points(exposed_comm, left_x, y0, panel_width, panel_height, comm_y_min, comm_y_max, "#dc2626", [f"{value:.1f}% hidden" for value in overlap_values]),
        f'<line x1="{right_x}" y1="{sy_for(1.0, y0, panel_height, speedup_y_min, speedup_y_max):.1f}" x2="{right_x + panel_width}" y2="{sy_for(1.0, y0, panel_height, speedup_y_min, speedup_y_max):.1f}" stroke="#9ca3af" stroke-dasharray="6 5"/>',
        f'<text x="{right_x + panel_width - 4:.1f}" y="{sy_for(1.0, y0, panel_height, speedup_y_min, speedup_y_max) - 8:.1f}" text-anchor="end" font-size="11" fill="#6b7280">no speedup 1x</text>',
        *line_points(speedups, right_x, y0, panel_width, panel_height, speedup_y_min, speedup_y_max, "#16a34a"),
        '<line x1="520" y1="102" x2="548" y2="102" stroke="#2563eb" stroke-width="2.4"/>',
        '<circle cx="534" cy="102" r="4.5" fill="#2563eb" stroke="white" stroke-width="1"/>',
        '<text x="558" y="106" font-size="11">blocking comm</text>',
        '<line x1="520" y1="124" x2="548" y2="124" stroke="#dc2626" stroke-width="2.4"/>',
        '<circle cx="534" cy="124" r="4.5" fill="#dc2626" stroke="white" stroke-width="1"/>',
        '<text x="558" y="128" font-size="11">overlap exposed comm</text>',
        '<line x1="1040" y1="102" x2="1068" y2="102" stroke="#16a34a" stroke-width="2.4"/>',
        '<circle cx="1054" cy="102" r="4.5" fill="#16a34a" stroke="white" stroke-width="1"/>',
        '<text x="1078" y="106" font-size="11">runtime speedup</text>',
        "</svg>",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare blocking and overlap MPI ring benchmark CSV files."
    )
    parser.add_argument("blocking_csv", type=Path)
    parser.add_argument("overlap_csv", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--svg", type=Path)
    args = parser.parse_args()

    summary = summarize(read_rows(args.blocking_csv), read_rows(args.overlap_csv))

    if args.csv is not None:
        write_csv(args.csv, summary)
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown_table(summary) + "\n", encoding="utf-8")
    if args.svg is not None:
        draw_svg(args.svg, summary)
    if args.csv is None and args.markdown is None and args.svg is None:
        print(markdown_table(summary))


if __name__ == "__main__":
    main()
