#!/usr/bin/env python3

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_METRIC_DIR = ROOT_DIR / "result"
LATENCY_FILE = "latency/total_end_to_end_ms.txt"
RECALL_FILE = "recall/recall_rate_percent.txt"


Predicate = Callable[[float], bool]


def read_metric_values(path: Path) -> list[float]:
    if not path.is_file():
        raise FileNotFoundError(f"metric file not found: {path}")

    values: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                value = float(text)
            except ValueError as exc:
                raise ValueError(f"invalid numeric value in {path} line {line_no}: {text}") from exc
            if not math.isfinite(value):
                raise ValueError(f"non-finite value in {path} line {line_no}: {text}")
            values.append(value)

    if not values:
        raise ValueError(f"no valid values found in {path}")

    return values


def try_read_metric_values(path: Path) -> list[float] | None:
    try:
        return read_metric_values(path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"warning: {exc}", file=sys.stderr)
        return None


def summarize_ranges(values: list[float], bins: list[tuple[str, Predicate]]) -> tuple[list[tuple[str, int, float]], list[float]]:
    counts = [0] * len(bins)
    unmatched: list[float] = []

    for value in values:
        matched = False
        for idx, (_, predicate) in enumerate(bins):
            if predicate(value):
                counts[idx] += 1
                matched = True
                break
        if not matched:
            unmatched.append(value)

    total = len(values)
    summary = []
    for idx, (label, _) in enumerate(bins):
        count = counts[idx]
        ratio = (count / total) * 100.0
        summary.append((label, count, ratio))
    return summary, unmatched


def percentile_nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"percentile must be between 0 and 100: {percentile}")

    sorted_values = sorted(values)
    if percentile == 0.0:
        return sorted_values[0]

    rank = math.ceil((percentile / 100.0) * len(sorted_values))
    return sorted_values[rank - 1]


def print_section(
    title: str,
    values: list[float],
    bins: list[tuple[str, Predicate]],
    extra_stats: list[tuple[str, float]] | None = None,
) -> None:
    average = sum(values) / len(values)
    summary, unmatched = summarize_ranges(values, bins)

    print(title)
    print(f"  count: {len(values)}")
    print(f"  average: {average:.4f}")
    if extra_stats:
        for label, value in extra_stats:
            print(f"  {label}: {value:.4f}")
    print("  distribution:")
    for label, count, ratio in summary:
        print(f"    {label:<18} {count:>8} ({ratio:6.2f}%)")
    if unmatched:
        ratio = (len(unmatched) / len(values)) * 100.0
        print(f"    {'unmatched':<18} {len(unmatched):>8} ({ratio:6.2f}%)")
        preview = ", ".join(f"{value:.4f}" for value in unmatched[:10])
        suffix = "" if len(unmatched) <= 10 else ", ..."
        print(f"  note: unmatched values = [{preview}{suffix}]")
    print()


def print_latency_section(values: list[float], bins: list[tuple[str, Predicate]]) -> None:
    average = sum(values) / len(values)
    qps = 1000.0 / average if average > 0.0 else math.inf
    p80 = percentile_nearest_rank(values, 80.0)
    p50 = percentile_nearest_rank(values, 50.0)
    p90 = percentile_nearest_rank(values, 90.0)
    p99 = percentile_nearest_rank(values, 99.0)
    p100 = percentile_nearest_rank(values, 100.0)
    print_section(
        "Latency (ms)",
        values,
        bins,
        extra_stats=[
            ("qps_by_avg_latency", qps),
            ("p50", p50),
            ("p80", p80),
            ("p90", p90),
            ("p99", p99),
            ("p100", p100),
        ],
    )


def print_recall_section(values: list[float], bins: list[tuple[str, Predicate]]) -> None:
    print_section("Recall (%)", values, bins)


def main() -> int:
    metric_dir = DEFAULT_METRIC_DIR
    latency_path = metric_dir / LATENCY_FILE
    recall_path = metric_dir / RECALL_FILE

    latency_values = try_read_metric_values(latency_path)
    recall_values = try_read_metric_values(recall_path)

    if latency_values is not None and recall_values is not None and len(latency_values) != len(recall_values):
        print(
            "warning: metric file row counts differ: "
            f"{LATENCY_FILE}={len(latency_values)}, {RECALL_FILE}={len(recall_values)}",
            file=sys.stderr,
        )

    latency_bins: list[tuple[str, Predicate]] = [
        ("<2ms", lambda value: value < 2.0),
        ("2-4ms", lambda value: 2.0 <= value < 4.0),
        ("4-6ms", lambda value: 4.0 <= value < 6.0),
        ("6-8ms", lambda value: 6.0 <= value < 8.0),
        ("8-15ms", lambda value: 8.0 <= value < 15.0),
        ("15-30ms", lambda value: 15.0 <= value <= 30.0),
        (">30ms", lambda value: value > 30.0),
    ]

    recall_bins: list[tuple[str, Predicate]] = [
        ("<=50%", lambda value: value <= 50.0),
        ("50%-80%", lambda value: 50.0 < value <= 80.0),
        ("80%-90%", lambda value: 80.0 < value <= 90.0),
        ("90%-95%", lambda value: 90.0 < value <= 95.0),
        ("95%-99%", lambda value: 95.0 < value < 99.0),
        (">=99%", lambda value: value >= 99.0),
    ]

    print(f"metric_dir: {metric_dir}")
    print()

    if latency_values is not None:
        print_latency_section(latency_values, latency_bins)
    if recall_values is not None:
        print_recall_section(recall_values, recall_bins)
    if latency_values is None and recall_values is None:
        print("warning: no analyzable metric files found", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
