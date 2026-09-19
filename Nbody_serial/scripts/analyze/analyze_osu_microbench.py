#!/usr/bin/env python3
"""Summarize OSU latency/bandwidth native-vs-container outputs."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


def read_osu_table(path: Path) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 2:
                continue
            try:
                size = int(fields[0])
                value = float(fields[1])
            except ValueError:
                continue
            rows.append((size, value))
    if not rows:
        raise ValueError(f"no OSU data rows found in {path}")
    return rows


def pick_latency(rows: list[tuple[int, float]], size: int) -> tuple[int, float]:
    by_size = dict(rows)
    if size in by_size:
        return size, by_size[size]
    return min(rows, key=lambda row: row[0])


def pick_bandwidth(rows: list[tuple[int, float]], size: int | None) -> tuple[int, float]:
    by_size = dict(rows)
    if size is not None and size in by_size:
        return size, by_size[size]
    return max(rows, key=lambda row: row[0])


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def collect_values(
    paths: list[Path],
    metric: str,
    size: int | None,
) -> tuple[int, list[float]]:
    selected_size: int | None = None
    values: list[float] = []
    for path in paths:
        rows = read_osu_table(path)
        if metric == "latency":
            current_size, current_value = pick_latency(rows, 0 if size is None else size)
        elif metric == "bandwidth":
            current_size, current_value = pick_bandwidth(rows, size)
        else:
            raise ValueError(f"unknown metric '{metric}'")

        if selected_size is None:
            selected_size = current_size
        elif selected_size != current_size:
            raise ValueError(
                f"selected message size mismatch for {metric}: "
                f"{selected_size} vs {current_size} in {path}"
            )
        values.append(current_value)

    if selected_size is None:
        raise ValueError(f"no {metric} files provided")
    return selected_size, values


def pct_delta(container: float, native: float) -> float:
    if native == 0.0:
        return 0.0
    return 100.0 * (container / native - 1.0)


def summarize_curve(
    native_paths: list[Path],
    container_paths: list[Path],
    metric: str,
) -> list[dict[str, object]]:
    native_by_size: dict[int, list[float]] = {}
    container_by_size: dict[int, list[float]] = {}

    for path in native_paths:
        for size, value in read_osu_table(path):
            native_by_size.setdefault(size, []).append(value)

    for path in container_paths:
        for size, value in read_osu_table(path):
            container_by_size.setdefault(size, []).append(value)

    common_sizes = sorted(set(native_by_size) & set(container_by_size))
    if not common_sizes:
        raise ValueError(f"no common OSU message sizes for {metric}")

    rows: list[dict[str, object]] = []
    for size in common_sizes:
        native_values = native_by_size[size]
        container_values = container_by_size[size]
        native_median = statistics.median(native_values)
        container_median = statistics.median(container_values)
        rows.append(
            {
                "metric": metric,
                "message_size_bytes": size,
                "repeats": min(len(native_values), len(container_values)),
                "native_median": native_median,
                "native_stdev": stdev(native_values),
                "container_median": container_median,
                "container_stdev": stdev(container_values),
                "relative_difference_percent": pct_delta(
                    container_median, native_median
                ),
            }
        )
    return rows


def markdown(summary: dict[str, object]) -> str:
    lines = [
        "| Metric | Message size bytes | Repeats | Native median ± sigma | Container median ± sigma | Relative difference |",
        "|:---|---:|---:|---:|---:|---:|",
        "| Latency (us, lower is better) | {lat_size} | {lat_repeats} | {native_lat:.3f} ± {native_lat_std:.3f} | {container_lat:.3f} ± {container_lat_std:.3f} | {lat_delta:+.2f}% |".format(
            lat_size=summary["latency_size_bytes"],
            lat_repeats=summary["latency_repeats"],
            native_lat=float(summary["native_latency_us"]),
            native_lat_std=float(summary["native_latency_stdev_us"]),
            container_lat=float(summary["container_latency_us"]),
            container_lat_std=float(summary["container_latency_stdev_us"]),
            lat_delta=float(summary["latency_overhead_percent"]),
        ),
        "| Bandwidth (MB/s, higher is better) | {bw_size} | {bw_repeats} | {native_bw:.3f} ± {native_bw_std:.3f} | {container_bw:.3f} ± {container_bw_std:.3f} | {bw_delta:+.2f}% |".format(
            bw_size=summary["bandwidth_size_bytes"],
            bw_repeats=summary["bandwidth_repeats"],
            native_bw=float(summary["native_bandwidth_MBps"]),
            native_bw_std=float(summary["native_bandwidth_stdev_MBps"]),
            container_bw=float(summary["container_bandwidth_MBps"]),
            container_bw_std=float(summary["container_bandwidth_stdev_MBps"]),
            bw_delta=float(summary["bandwidth_delta_percent"]),
        ),
        "",
        "The OSU Micro-Benchmarks were built locally in user space because they were not available as Orfeo modules. The container run uses the same host OpenMPI runtime policy as the application container runs. Positive latency difference means the container is slower; positive bandwidth difference means the container measured higher bandwidth.",
    ]
    return "\n".join(lines)


def curve_markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "| Metric | Message size bytes | Repeats | Native median +/- sigma | Container median +/- sigma | Relative difference |",
        "|:---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {metric} | {size} | {repeats} | {native:.3f} +/- {native_std:.3f} | {container:.3f} +/- {container_std:.3f} | {delta:+.2f}% |".format(
                metric=row["metric"],
                size=row["message_size_bytes"],
                repeats=row["repeats"],
                native=float(row["native_median"]),
                native_std=float(row["native_stdev"]),
                container=float(row["container_median"]),
                container_std=float(row["container_stdev"]),
                delta=float(row["relative_difference_percent"]),
            )
        )
    return "\n".join(lines)


def write_curve_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "metric",
        "message_size_bytes",
        "repeats",
        "native_median",
        "native_stdev",
        "container_median",
        "container_stdev",
        "relative_difference_percent",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def log10(value: float) -> float:
    import math

    return math.log10(max(value, 1.0e-300))


def pretty_size(size: int) -> str:
    if size >= 1024 * 1024 and size % (1024 * 1024) == 0:
        return f"{size // (1024 * 1024)} MiB"
    if size >= 1024 and size % 1024 == 0:
        return f"{size // 1024} KiB"
    return str(size)


def pretty_number(value: float) -> str:
    if value >= 1000.0:
        return f"{value:.0f}"
    if value >= 100.0:
        return f"{value:.1f}"
    if value >= 10.0:
        return f"{value:.2f}"
    return f"{value:.3f}"


def svg_polyline(points: list[tuple[float, float]], color: str) -> str:
    coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5"/>'


def svg_markers(points: list[tuple[float, float]], color: str, marker: str) -> str:
    if marker == "square":
        return "\n".join(
            f'<rect x="{x - 4:.2f}" y="{y - 4:.2f}" width="8" height="8" fill="{color}"/>'
            for x, y in points
        )
    return "\n".join(
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>'
        for x, y in points
    )


def draw_curve_panel(
    rows: list[dict[str, object]],
    metric: str,
    x0: float,
    y0: float,
    width: float,
    height: float,
) -> str:
    metric_rows = [
        row for row in rows if str(row["metric"]) == metric
    ]
    if not metric_rows:
        return ""

    sizes = [float(row["message_size_bytes"]) for row in metric_rows]
    native_values = [float(row["native_median"]) for row in metric_rows]
    container_values = [float(row["container_median"]) for row in metric_rows]
    x_min = log10(min(sizes))
    x_max = log10(max(sizes))
    if x_min == x_max:
        x_max = x_min + 1.0
    use_log_y = metric == "latency"
    if use_log_y:
        positive_values = [value for value in native_values + container_values if value > 0.0]
        y_min_log = log10(min(positive_values) * 0.8)
        y_max_log = log10(max(positive_values) * 1.25)
        if y_min_log == y_max_log:
            y_max_log = y_min_log + 1.0
    else:
        y_min = 0.0
        y_max = max(native_values + container_values) * 1.18
        if y_max <= 0.0:
            y_max = 1.0

    def sx(size: float) -> float:
        return x0 + (log10(size) - x_min) / (x_max - x_min) * width

    def sy(value: float) -> float:
        if use_log_y:
            return y0 + height - (log10(value) - y_min_log) / (y_max_log - y_min_log) * height
        return y0 + height - (value - y_min) / (y_max - y_min) * height

    native_points = [
        (sx(float(row["message_size_bytes"])), sy(float(row["native_median"])))
        for row in metric_rows
    ]
    container_points = [
        (sx(float(row["message_size_bytes"])), sy(float(row["container_median"])))
        for row in metric_rows
    ]

    title = "OSU latency" if metric == "latency" else "OSU bandwidth"
    ylabel = "Latency (us, log scale)" if metric == "latency" else "Bandwidth (MB/s)"
    note = "lower is better" if metric == "latency" else "higher is better"
    tick_sizes = [1, 8, 64, 512, 4096, 32768, 262144, 2097152, 4194304]
    tick_sizes = [size for size in tick_sizes if min(sizes) <= size <= max(sizes)]

    parts = [
        f'<text x="{x0 + width / 2:.1f}" y="{y0 - 20:.1f}" text-anchor="middle" font-size="17" font-weight="700">{title}</text>',
        f'<text x="{x0 + width / 2:.1f}" y="{y0 - 4:.1f}" text-anchor="middle" font-size="11" fill="#4b5563">{note}</text>',
        f'<line x1="{x0}" y1="{y0 + height}" x2="{x0 + width}" y2="{y0 + height}" stroke="#111827"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + height}" stroke="#111827"/>',
        f'<text x="{x0 + width / 2:.1f}" y="{y0 + height + 58:.1f}" text-anchor="middle" font-size="13">Message size (bytes, log scale)</text>',
        f'<text x="{x0 - 56:.1f}" y="{y0 + height / 2:.1f}" transform="rotate(-90 {x0 - 56:.1f},{y0 + height / 2:.1f})" text-anchor="middle" font-size="13">{ylabel}</text>',
    ]

    if use_log_y:
        import math

        start_power = math.floor(y_min_log)
        end_power = math.ceil(y_max_log)
        y_ticks = [10.0 ** power for power in range(start_power, end_power + 1)]
    else:
        y_ticks = [y_max * i / 4 for i in range(5)]

    for y in y_ticks:
        py = sy(y)
        if y0 <= py <= y0 + height:
            parts.append(
                f'<line x1="{x0}" y1="{py:.1f}" x2="{x0 + width}" y2="{py:.1f}" stroke="#e5e7eb"/>'
            )
            parts.append(
                f'<text x="{x0 - 8:.1f}" y="{py + 4:.1f}" text-anchor="end" font-size="11">{pretty_number(y)}</text>'
            )

    for size in tick_sizes:
        px = sx(float(size))
        parts.append(
            f'<line x1="{px:.1f}" y1="{y0 + height}" x2="{px:.1f}" y2="{y0 + height + 5}" stroke="#111827"/>'
        )
        parts.append(
            f'<text x="{px:.1f}" y="{y0 + height + 22:.1f}" text-anchor="middle" font-size="10">{pretty_size(size)}</text>'
        )

    parts.append(svg_polyline(native_points, "#2563eb"))
    parts.append(svg_markers(native_points, "#2563eb", "circle"))
    parts.append(svg_polyline(container_points, "#dc2626"))
    parts.append(svg_markers(container_points, "#dc2626", "square"))

    legend_x = x0 + width - 104
    legend_y = y0 + 22
    parts.extend(
        [
            f'<circle cx="{legend_x - 10:.1f}" cy="{legend_y - 4:.1f}" r="4" fill="#2563eb"/>',
            f'<text x="{legend_x:.1f}" y="{legend_y:.1f}" font-size="12" fill="#2563eb">native</text>',
            f'<rect x="{legend_x - 14:.1f}" y="{legend_y + 10:.1f}" width="8" height="8" fill="#dc2626"/>',
            f'<text x="{legend_x:.1f}" y="{legend_y + 18:.1f}" font-size="12" fill="#dc2626">container</text>',
        ]
    )
    return "\n".join(parts)


def write_curve_svg(path: Path, rows: list[dict[str, object]]) -> None:
    width = 1360
    height = 560
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial, Helvetica, sans-serif; fill:#111827;}</style>',
        '<text x="680" y="36" text-anchor="middle" font-size="23" font-weight="700">OSU MPI microbenchmark: native vs container</text>',
        '<text x="680" y="62" text-anchor="middle" font-size="13" fill="#4b5563">Median over repeated OSU runs using the same host-MPI runtime policy as the controlled container comparison.</text>',
        draw_curve_panel(rows, "latency", 86, 126, 540, 310),
        draw_curve_panel(rows, "bandwidth", 748, 126, 540, 310),
        "</svg>",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(part for part in parts if part) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize OSU latency and bandwidth native/container outputs."
    )
    parser.add_argument("--native-latency", type=Path, nargs="+", required=True)
    parser.add_argument("--container-latency", type=Path, nargs="+", required=True)
    parser.add_argument("--native-bandwidth", type=Path, nargs="+", required=True)
    parser.add_argument("--container-bandwidth", type=Path, nargs="+", required=True)
    parser.add_argument("--latency-size", type=int, default=0)
    parser.add_argument("--bandwidth-size", type=int)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--curve-csv", type=Path)
    parser.add_argument("--curve-markdown", type=Path)
    parser.add_argument("--svg", type=Path)
    args = parser.parse_args()

    native_lat_size, native_lat_values = collect_values(
        args.native_latency, "latency", args.latency_size
    )
    container_lat_size, container_lat_values = collect_values(
        args.container_latency, "latency", args.latency_size
    )
    native_bw_size, native_bw_values = collect_values(
        args.native_bandwidth, "bandwidth", args.bandwidth_size
    )
    container_bw_size, container_bw_values = collect_values(
        args.container_bandwidth, "bandwidth", args.bandwidth_size
    )

    if native_lat_size != container_lat_size:
        raise ValueError("native/container latency sizes do not match")
    if native_bw_size != container_bw_size:
        raise ValueError("native/container bandwidth sizes do not match")

    native_lat = statistics.median(native_lat_values)
    container_lat = statistics.median(container_lat_values)
    native_bw = statistics.median(native_bw_values)
    container_bw = statistics.median(container_bw_values)

    summary = {
        "latency_size_bytes": native_lat_size,
        "latency_repeats": min(len(native_lat_values), len(container_lat_values)),
        "native_latency_us": native_lat,
        "native_latency_stdev_us": stdev(native_lat_values),
        "container_latency_us": container_lat,
        "container_latency_stdev_us": stdev(container_lat_values),
        "latency_overhead_percent": pct_delta(container_lat, native_lat),
        "bandwidth_size_bytes": native_bw_size,
        "bandwidth_repeats": min(len(native_bw_values), len(container_bw_values)),
        "native_bandwidth_MBps": native_bw,
        "native_bandwidth_stdev_MBps": stdev(native_bw_values),
        "container_bandwidth_MBps": container_bw,
        "container_bandwidth_stdev_MBps": stdev(container_bw_values),
        "bandwidth_delta_percent": pct_delta(container_bw, native_bw),
    }

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary))
            writer.writeheader()
            writer.writerow(summary)

    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(summary) + "\n", encoding="utf-8")

    if args.csv is None and args.markdown is None:
        print(markdown(summary))

    if args.curve_csv is not None or args.curve_markdown is not None or args.svg is not None:
        curve_rows = [
            *summarize_curve(args.native_latency, args.container_latency, "latency"),
            *summarize_curve(args.native_bandwidth, args.container_bandwidth, "bandwidth"),
        ]
        if args.curve_csv is not None:
            write_curve_csv(args.curve_csv, curve_rows)
        if args.curve_markdown is not None:
            args.curve_markdown.parent.mkdir(parents=True, exist_ok=True)
            args.curve_markdown.write_text(
                curve_markdown(curve_rows) + "\n", encoding="utf-8"
            )
        if args.svg is not None:
            write_curve_svg(args.svg, curve_rows)


if __name__ == "__main__":
    main()
