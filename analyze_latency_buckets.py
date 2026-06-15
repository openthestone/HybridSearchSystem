#!/usr/bin/env python3
"""Analyze L2 bucket metrics grouped by query latency percentile buckets."""

import argparse
import sys
from pathlib import Path
import numpy as np


def load_lines(path: Path):
    return path.read_text().strip().split("\n")


def main():
    parser = argparse.ArgumentParser(description="L2 bucket stats by latency bucket")
    parser.add_argument("result_dir", nargs="?", default="result", help="Path to result directory")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    latency_file = result_dir / "latency" / "total_end_to_end_ms.txt"
    selectivity_file = result_dir / "log" / "l2_bucket_selectivity.txt"
    count_file = result_dir / "log" / "searched_l2_bucket_count.txt"

    for f in [latency_file, selectivity_file, count_file]:
        if not f.exists():
            sys.exit(f"Missing: {f}")

    latencies = np.array([float(x) for x in load_lines(latency_file)])
    sel_lines = load_lines(selectivity_file)
    count_lines = load_lines(count_file)

    n = len(latencies)

    # Parse two-column count file (skip header)
    nonzero_counts = []
    allzero_counts = []
    for line in count_lines[1:]:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            nonzero_counts.append(int(parts[0]))
            allzero_counts.append(int(parts[1]))
        else:
            nonzero_counts.append(int(parts[0]) if parts[0] else 0)
            allzero_counts.append(0)

    nonzero_counts = np.array(nonzero_counts)
    allzero_counts = np.array(allzero_counts)
    total_counts = nonzero_counts + allzero_counts

    if len(nonzero_counts) != n or len(sel_lines) != n:
        sys.exit(f"Row count mismatch: latency={n}, count={len(nonzero_counts)}, selectivity={len(sel_lines)}")

    # Parse selectivity: each line has space-separated floats (may be empty for 0 buckets)
    selectivities = []
    for line in sel_lines:
        parts = line.strip().split()
        selectivities.append(np.array([float(x) for x in parts]) if parts else np.array([]))

    # Compute percentile boundaries
    p50 = np.percentile(latencies, 50)
    p80 = np.percentile(latencies, 80)
    p90 = np.percentile(latencies, 90)
    p99 = np.percentile(latencies, 99)

    # Define buckets: (label, mask)
    buckets = [
        (f"< P50  ({p50:.2f}ms)",    latencies < p50),
        (f"P50-P80 ({p50:.2f}-{p80:.2f}ms)", (latencies >= p50) & (latencies < p80)),
        (f"P80-P90 ({p80:.2f}-{p90:.2f}ms)", (latencies >= p80) & (latencies < p90)),
        (f"P90-P99 ({p90:.2f}-{p99:.2f}ms)", (latencies >= p90) & (latencies < p99)),
        (f">= P99 ({p99:.2f}ms)",    latencies >= p99),
    ]

    hdr = f"{'Bucket':<35s} {'Queries':>7s} {'Mean':>9s} {'Median':>9s} {'Max':>9s}"
    sep = "-" * len(hdr)

    for section, vals_fn in [
        ("Total", lambda idx: total_counts[idx]),
        ("Non0",  lambda idx: nonzero_counts[idx]),
        ("All0",  lambda idx: allzero_counts[idx]),
        ("Sel",   None),
    ]:
        print(f"\n=== {section} ===")
        if section == "Sel":
            sel_hdr = f"{'Bucket':<35s} {'Queries':>7s} {'Mean':>9s} {'Median':>9s} {'Max':>9s}"
            print(sel_hdr)
            print("-" * len(sel_hdr))
        else:
            print(hdr)
            print(sep)

        for label, mask in buckets:
            indices = np.where(mask)[0]
            qcount = len(indices)
            if qcount == 0:
                print(f"{label:<35s} {qcount:>7d} {'N/A':>9s} {'N/A':>9s} {'N/A':>9s}")
                continue

            if section == "Sel":
                all_sel = np.concatenate([selectivities[i] for i in indices])
                if len(all_sel) == 0:
                    print(f"{label:<35s} {qcount:>7d} {'N/A':>9s} {'N/A':>9s} {'N/A':>9s}")
                else:
                    mean = float(np.mean(all_sel))
                    med = float(np.median(all_sel))
                    mx = float(np.max(all_sel))
                    print(f"{label:<35s} {qcount:>7d} {mean:>9.6f} {med:>9.6f} {mx:>9.6f}")
            else:
                vals = vals_fn(indices)
                mean = float(np.mean(vals))
                med = float(np.median(vals))
                mx = int(np.max(vals))
                print(f"{label:<35s} {qcount:>7d} {mean:>9.3f} {med:>9.3f} {mx:>9d}")


if __name__ == "__main__":
    main()
