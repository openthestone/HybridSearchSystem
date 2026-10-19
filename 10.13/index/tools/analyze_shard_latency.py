#!/usr/bin/env python3
"""Decompose the multi-card tail from fr_search --shard_latency_dump.

Answers "why is 2-card p99 worse than single-card": is the tail driven by
(a) one shard systematically slower (LOAD IMBALANCE -> rebalance shards),
(b) a random shard spiking per query (JITTER -> thread pool / affinity), or
(c) both shards slow on the same query (CORRELATED system jitter / the known
    per-shard aggregation over-return spike, hit ~N x by max-of-shards),
and how much the host MERGE and fan-out OVERHEAD (total - max_shard) cost.

TSV columns: idx, shard0_ms..shardN_ms, max_shard_ms, merge_ms, total_ms.
"""
import sys


def pct(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(p * len(s)))]


def main():
    if len(sys.argv) < 2:
        print("usage: analyze_shard_latency.py <shard_latency.tsv>")
        return 1
    rows = []
    with open(sys.argv[1]) as f:
        header = f.readline().rstrip("\n").split("\t")
        nshards = sum(1 for h in header if h.startswith("shard") and h.endswith("_ms"))
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < nshards + 4:
                continue
            shards = [float(p[1 + s]) for s in range(nshards)]
            rows.append({
                "idx": int(p[0]),
                "shards": shards,
                "maxs": float(p[1 + nshards]),
                "merge": float(p[2 + nshards]),
                "total": float(p[3 + nshards]),
            })
    if not rows:
        print("no rows")
        return 1
    n = len(rows)
    print(f"n={n} queries, {nshards} shards\n")

    # 1) is one shard systematically slower? (load imbalance)
    print("== per-shard own time (systematic imbalance?) ==")
    for s in range(nshards):
        col = [r["shards"][s] for r in rows]
        print(f"  shard{s}: mean={sum(col)/n:.3f}  p50={pct(col,.5):.3f}  "
              f"p99={pct(col,.99):.3f}  max={max(col):.3f}")
    straggler_count = [0] * nshards
    for r in rows:
        straggler_count[r["shards"].index(r["maxs"])] += 1
    print("  straggler (the slowest shard) share: "
          + "  ".join(f"shard{s}={100*straggler_count[s]/n:.1f}%" for s in range(nshards)))

    # 2) cost decomposition: max_shard vs merge vs fan-out overhead
    overhead = [r["total"] - r["maxs"] for r in rows]  # thread spawn/join + scheduling
    print("\n== cost decomposition (mean / p99) ==")
    print(f"  max_shard : {sum(r['maxs'] for r in rows)/n:7.3f} / {pct([r['maxs'] for r in rows],.99):7.3f}")
    print(f"  merge     : {sum(r['merge'] for r in rows)/n:7.3f} / {pct([r['merge'] for r in rows],.99):7.3f}")
    print(f"  overhead  : {sum(overhead)/n:7.3f} / {pct(overhead,.99):7.3f}   (total - max_shard: spawn/join/sched)")
    print(f"  total     : {sum(r['total'] for r in rows)/n:7.3f} / {pct([r['total'] for r in rows],.99):7.3f}")

    # 3) straggler tax: max_shard - mean_shard (what fan-out loses vs a balanced ideal)
    tax = [r["maxs"] - sum(r["shards"]) / nshards for r in rows]
    spread = [r["maxs"] - min(r["shards"]) for r in rows]
    print("\n== straggler tax (max_shard - avg_shard) ==")
    print(f"  mean={sum(tax)/n:.3f}  p99={pct(tax,.99):.3f}   (0 = perfectly balanced, no straggler cost)")
    print(f"  max-min spread: mean={sum(spread)/n:.3f}  p99={pct(spread,.99):.3f}")

    # 4) tail decomposition: on the slowest 1% by total, is it one shard or both?
    tail = sorted(rows, key=lambda r: r["total"], reverse=True)[:max(1, n // 100)]
    print(f"\n== slowest {len(tail)} queries (top 1% by total) ==")
    both = one = 0
    thr_hi = pct([r["total"] for r in rows], 0.90)  # "slow" bar = overall p90
    for r in tail:
        slow = sum(1 for x in r["shards"] if x >= thr_hi)
        if slow >= 2:
            both += 1
        else:
            one += 1
    print(f"  slow-shard bar = overall p90 = {thr_hi:.3f} ms")
    print(f"  ONE shard slow (jitter/imbalance): {one}/{len(tail)}  "
          f"({100*one/len(tail):.0f}%)")
    print(f"  BOTH shards slow (correlated / over-return): {both}/{len(tail)}  "
          f"({100*both/len(tail):.0f}%)")
    mt = sum(r["maxs"] for r in tail) / len(tail)
    mg = sum(r["merge"] for r in tail) / len(tail)
    ov = sum(r["total"] - r["maxs"] for r in tail) / len(tail)
    print(f"  tail mean: max_shard={mt:.3f}  merge={mg:.3f}  overhead={ov:.3f}")

    print("\n== reading ==")
    print("  overhead p99 large        -> per-query thread spawn/sched: persistent pool")
    print("  one shard always straggler-> load imbalance: interleave/round-robin shard assignment")
    print("  tail = BOTH shards slow   -> correlated (over-return spike or box jitter): cut per-shard tail")
    print("  tail = ONE shard slow     -> independent jitter amplified by max-of-N: pool/affinity/hedge")
    return 0


if __name__ == "__main__":
    sys.exit(main())
