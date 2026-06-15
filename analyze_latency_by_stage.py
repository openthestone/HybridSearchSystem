#!/usr/bin/env python3
"""Analyze per-stage latency metrics grouped by query total latency percentile buckets."""

import argparse
import sys
from pathlib import Path
import numpy as np


def load_lines(path: Path):
    return path.read_text().strip().split("\n")


# All stages in pipeline order (fine-grained)
STAGE_ORDER = [
    "bucket_level_ivf_ms",
    "bucket_level_ivf_filter_eval_ms",
    "bucket_level_ivf_centroid_score_ms",
    "bucket_level_ivf_result_pack_ms",
    "candidate_bucket_merge_ms",
    "npu_async_launch_ms",
    "inbucket_attr_filter_overlapped_ms",
    "wait_npu_flag_ms",
    "npu_mask_filter_launch_ms",
    "npu_mask_filter_h2d_ms",
    "npu_mask_filter_kernel_exec_ms",
    "npu_mask_filter_d2h_ms",
    "result_collection_ms",
    "final_merge_ms",
]


def main():
    parser = argparse.ArgumentParser(description="Per-stage latency by query latency bucket")
    parser.add_argument("result_dir", nargs="?", default="result", help="Path to result directory")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    latency_file = result_dir / "latency" / "total_end_to_end_ms.txt"

    if not latency_file.exists():
        sys.exit(f"Missing: {latency_file}")

    latencies = np.array([float(x) for x in load_lines(latency_file)])
    n = len(latencies)

    # Load all available stage metrics
    stage_data = {}
    for stage in STAGE_ORDER:
        f = result_dir / "latency" / f"{stage}.txt"
        if not f.exists():
            continue
        vals = np.array([float(x) for x in load_lines(f)])
        if len(vals) != n:
            print(f"[Warn] Row count mismatch: {stage} ({len(vals)} vs {n})", file=sys.stderr)
            continue
        stage_data[stage] = vals

    if not stage_data:
        sys.exit("No stage data loaded.")

    # Compute percentile boundaries
    p50 = np.percentile(latencies, 50)
    p80 = np.percentile(latencies, 80)
    p90 = np.percentile(latencies, 90)
    p99 = np.percentile(latencies, 99)

    # Define latency buckets
    buckets = [
        (f"< P50  ({p50:.2f}ms)", latencies < p50),
        (f"P50-P80 ({p50:.2f}-{p80:.2f}ms)", (latencies >= p50) & (latencies < p80)),
        (f"P80-P90 ({p80:.2f}-{p90:.2f}ms)", (latencies >= p80) & (latencies < p90)),
        (f"P90-P99 ({p90:.2f}-{p99:.2f}ms)", (latencies >= p90) & (latencies < p99)),
        (f">= P99 ({p99:.2f}ms)", latencies >= p99),
    ]

    stage_col = 38
    hdr = f"{'Stage':<{stage_col}s} {'Queries':>7s} {'Mean':>9s} {'Median':>9s} {'Max':>9s}"
    sep = "-" * len(hdr)

    for label, mask in buckets:
        indices = np.where(mask)[0]
        qcount = len(indices)
        print(f"\n{label}")
        print(hdr)
        print(sep)

        for stage in STAGE_ORDER:
            if stage not in stage_data:
                continue
            vals = stage_data[stage][indices]
            short = stage.replace("_ms", "").replace("_", " ")
            # Fix misleading label: h2d_ms actually measures stream sync wait
            if short == "npu mask filter h2d":
                short = "mf stream sync (h2d)"
            if qcount == 0:
                print(f"{short:<{stage_col}s} {qcount:>7d} {'N/A':>9s} {'N/A':>9s} {'N/A':>9s}")
            else:
                mean = float(np.mean(vals))
                med = float(np.median(vals))
                mx = float(np.max(vals))
                print(f"{short:<{stage_col}s} {qcount:>7d} {mean:>9.4f} {med:>9.4f} {mx:>9.4f}")


if __name__ == "__main__":
    main()
