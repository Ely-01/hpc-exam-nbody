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


def bar(x: float, y: float, width: float, height: float, color: str) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" fill="{color}"/>'


def draw_svg(path: Path, summary: list[dict[str, object]]) -> None:
    rows = sorted(summary, key=lambda row: (int(row["total_workers"]), int(row["ranks"]), int(row["threads"])))
    labels = [f"{row['ranks']}x{row['threads']}" for row in rows]
    overlap_values = [100.0 * float(row["overlap_fraction"]) for row in rows]
    napkin_values = [100.0 * float(row["napkin_overlap_fraction"]) for row in rows]
    speedups = [float(row["total_speedup_blocking_over_overlap"]) for row in rows]
    y_max_overlap = max(100.0, max(napkin_values + overlap_values) * 1.15)
    y_max_speedup = max(1.05, max(speedups) * 1.12)

    def panel(x0: float, y0: float, width: float, height: float, y_max: float, values_a: list[float], values_b: list[float] | None, title: str, ylabel: str) -> list[str]:
        parts = [
            f'<text x="{x0 + width / 2:.1f}" y="{y0 - 18:.1f}" text-anchor="middle" font-size="16" font-weight="700">{svg_escape(title)}</text>',
            f'<line x1="{x0}" y1="{y0 + height}" x2="{x0 + width}" y2="{y0 + height}" stroke="#222"/>',
            f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + height}" stroke="#222"/>',
            f'<text x="{x0 - 50:.1f}" y="{y0 + height / 2:.1f}" transform="rotate(-90 {x0 - 50:.1f},{y0 + height / 2:.1f})" text-anchor="middle" font-size="12">{svg_escape(ylabel)}</text>',
            f'<text x="{x0 + width / 2:.1f}" y="{y0 + height + 44:.1f}" text-anchor="middle" font-size="12">ranks x threads</text>',
        ]
        for i in range(6):
            value = y_max * i / 5
            py = y0 + height - value / y_max * height
            parts.append(f'<line x1="{x0}" y1="{py:.1f}" x2="{x0 + width}" y2="{py:.1f}" stroke="#e5e7eb"/>')
            parts.append(f'<text x="{x0 - 8:.1f}" y="{py + 4:.1f}" text-anchor="end" font-size="10">{value:.2g}</text>')
        group_width = width / len(rows)
        for index, label in enumerate(labels):
            center = x0 + index * group_width + group_width / 2
            bw = group_width / (3.4 if values_b is not None else 2.2)
            parts.append(f'<text x="{center:.1f}" y="{y0 + height + 23:.1f}" text-anchor="middle" font-size="11">{svg_escape(label)}</text>')
            for value, offset, color in [
                (values_a[index], -0.55 if values_b is not None else -0.35, "#2563eb"),
                (values_b[index] if values_b is not None else None, 0.35, "#ca8a04"),
            ]:
                if value is None:
                    continue
                py = y0 + height - value / y_max * height
                parts.append(bar(center + offset * bw, py, bw * 0.8, y0 + height - py, color))
        return parts

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="690" viewBox="0 0 1500 690">',
        '<rect width="1500" height="690" fill="white"/>',
        '<text x="750" y="38" text-anchor="middle" font-size="24" font-weight="700">MPI ring communication overlap</text>',
        '<text x="750" y="66" text-anchor="middle" font-size="13">Measured exposed communication reduction compared with an optimistic napkin estimate.</text>',
        *panel(95, 135, 590, 400, y_max_overlap, overlap_values, napkin_values, "Overlap achieved", "% of blocking comm hidden"),
        *panel(830, 135, 590, 400, y_max_speedup, speedups, None, "Total runtime speedup", "blocking / overlap"),
        '<rect x="610" y="112" width="10" height="10" fill="#2563eb"/>',
        '<text x="626" y="122" font-size="11">measured</text>',
        '<rect x="610" y="132" width="10" height="10" fill="#ca8a04"/>',
        '<text x="626" y="142" font-size="11">napkin estimate</text>',
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
