import argparse
import re
import sys

MB = 1024.0 * 1024.0
HEADER_BYTES = 8

# NPUR_OR_ABLATE on 2 cards, den=0.01, BS=4. The arms are strict subsets, so each piece is a diff:
ABL_FULL = 1372.5   # ablate=0
ABL_PLACE = 937.3   # full - ablate=1, the scalar placement loop
ABL_FIXED = 73.6    # ablate=1 - ablate=3, the input DataCopy plus alloc/queues/free
ABL_DEN0 = 721.4    # the same stage with DENSITY_THRESHOLD=0, every posting dense

# Sparse pair = A * K + f, dense pair = B:
#   A * kSum = ABL_PLACE     f * S = ABL_FIXED
#   B * S = ABL_PLACE + ABL_FIXED - (ABL_FULL - ABL_DEN0)
# S cancels out of the break-even, which is why this needs no operand count.
ABL_AVOIDED = ABL_PLACE + ABL_FIXED - (ABL_FULL - ABL_DEN0)

# Calibrated on synthetic v2, packed layout. K* is only as good as ABL_DEN0, which comes from a
# different run than the other three.


def parse_kv(line):
    """field=value pairs, plus kHist=[...] as a list of ints."""
    out = {}
    hist = re.search(r"kHist=\[([0-9,\s]*)\]", line)
    if hist:
        out["kHist"] = [int(x) for x in hist.group(1).split(",") if x.strip()]
    for key, val in re.findall(r"([A-Za-z][A-Za-z0-9]*)=(-?[0-9.]+)", line):
        out[key] = float(val) if "." in val else int(val)
    return out


def last_line(path, tag):
    found = None
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if line.startswith(tag):
                found = line.rstrip("\n")
    return found


def bucket_range(b):
    """Bucket b holds 2^(b-1) <= K < 2^b; bucket 0 holds K == 0."""
    if b == 0:
        return 0, 1
    return 1 << (b - 1), 1 << b


def cumulative(hist, bytes_per_unit=4):
    """Per bucket: (lo, hi, count, mean K, bytes if compressed). Bucket 0 is dropped."""
    rows = []
    for b, count in enumerate(hist):
        if b == 0 or count == 0:
            continue
        lo, hi = bucket_range(b)
        mean_k = (lo + hi - 1) / 2.0
        rows.append((lo, hi, count, mean_k, count * (HEADER_BYTES + bytes_per_unit * mean_k)))
    return rows


def index_bytes_at(cutoff_k, rows, total_postings, segment_bytes, byte_scale, bytes_per_unit=4):
    """Index bytes when everything with K < cutoff_k is compressed.

    Within the bucket the cutoff falls in, the count is split linearly -- the one approximation in
    here, and on a corpus whose mass sits inside one bucket it is a bad one. Measured on
    frnew_syn_v2, 131072 docs per segment, against indexes actually built:

        threshold   K <     predicted      measured
        0.00017     22.3    -24.9%         -7.9%     (16985 MB/shard of 18439)
        0.00021     27.5    -36.4%         -18.8%    (14966)
        0.01        1310    -85.7%         -82.6%    (3214)

    The bucket edges are powers of two and this corpus keeps both its index mass and every posting
    its queries touch inside [16, 32), so the interpolation is doing all the work exactly where it
    has no resolution. The edges themselves are sound -- den=0 and 0.01 land within 2-3% -- so read
    this column as a ranking between edges, not as a size at a threshold between them. Placing one
    costs a --convert-data 1 build, and two builds bracket it well: on the numbers above the slope
    was 50.5 MB per 1e-6 of threshold, flat enough to interpolate over a range that narrow.
    """
    sparse_count = 0.0
    sparse_bytes = 0.0
    for lo, hi, count, mean_k, cost in rows:
        if hi <= cutoff_k:
            sparse_count += count
            sparse_bytes += cost
        elif lo < cutoff_k:
            frac = (cutoff_k - lo) / float(hi - lo)
            part = count * frac
            part_mean = (lo + cutoff_k - 1) / 2.0
            sparse_count += part
            sparse_bytes += part * (HEADER_BYTES + bytes_per_unit * part_mean)
    dense_count = total_postings - sparse_count
    return dense_count * segment_bytes + sparse_bytes * byte_scale, sparse_count


def touched_at(cutoff_k, deep_rows, total_ksum):
    """(sum of K, pair count) still handled sparsely at this cutoff, and the total pair count."""
    if not deep_rows:
        return None
    kept_k = 0.0
    kept_n = 0.0
    total_n = 0.0
    for lo, hi, count, mean_k, _cost in deep_rows:
        total_n += count
        if hi <= cutoff_k:
            kept_k += count * mean_k
            kept_n += count
        elif lo < cutoff_k:
            frac = (cutoff_k - lo) / float(hi - lo)
            kept_k += count * frac * (lo + cutoff_k - 1) / 2.0
            kept_n += count * frac
    return min(kept_k, total_ksum), kept_n, total_n


def filter_cost_model(deep_rows, total_ksum):
    """Per-unit and per-pair prices, calibrated by the ablation against THIS log's histogram.

    A pair stored sparsely costs A*K + f of filter-kernel time; the same pair stored dense
    costs B. Returns (A, f, B, K*) with A in us per unit of query-touched kSum.
    """
    ref = touched_at(float(1 << 20), deep_rows, total_ksum)  # everything touched is sparse
    if ref is None or ref[1] <= 0:
        return None
    s_ref = ref[1]
    us_per_unit = ABL_PLACE / total_ksum
    us_sparse_fixed = ABL_FIXED / s_ref
    us_dense_pair = ABL_AVOIDED / s_ref
    return us_per_unit, us_sparse_fixed, us_dense_pair, (us_dense_pair - us_sparse_fixed) / us_per_unit


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True, help="a log holding [BITSTAT-INDEX] (BITLIST_STATS=1)")
    ap.add_argument("--doc-num-per-segment", type=int, default=131072)
    ap.add_argument("--layout", choices=("packed", "pairs"), default="packed",
                    help="sparse posting layout: packed is one uint32 per unit (the default since "
                         "SPARSE_PACKED landed), pairs is a uint32 offset plus a uint16 mask")
    ap.add_argument("--target-reduction", type=float, default=0.20,
                    help="index reduction to hit against DENSITY_THRESHOLD=0, e.g. 0.20")
    args = ap.parse_args()
    bytes_per_unit = 4 if args.layout == "packed" else 6

    idx_line = last_line(args.log, "[BITSTAT-INDEX]")
    if idx_line is None:
        sys.exit("no [BITSTAT-INDEX] in %s -- rerun with BITLIST_STATS=1" % args.log)
    idx = parse_kv(idx_line)
    for need in ("postings", "sparse", "sparseBytes", "kHist"):
        if need not in idx:
            sys.exit("[BITSTAT-INDEX] is missing %s" % need)

    total = idx["postings"]
    segment_bytes = args.doc_num_per_segment / 8.0
    rows = cumulative(idx["kHist"], bytes_per_unit)

    # Pin the midpoint-per-bucket estimate to the measured sparseBytes. Which way it errs depends
    # on how the mass sits inside the buckets: 0.79 on one corpus, 1.36 on another.
    est = sum(cost for _lo, _hi, _c, _m, cost in rows)
    byte_scale = (idx["sparseBytes"] / est) if est > 0 else 1.0

    deep_line = last_line(args.log, "[BITSTAT-DEEP]")
    deep_rows, deep_ksum = [], None
    if deep_line:
        deep = parse_kv(deep_line)
        if "kHist" in deep and "kSum" in deep:
            deep_rows = cumulative(deep["kHist"], bytes_per_unit)
            deep_ksum = float(deep["kSum"])

    base_bytes = total * segment_bytes
    print("index at DENSITY_THRESHOLD=0: %.0f MB/shard over %d postings" % (base_bytes / MB, total))
    print("this log: sparse=%d sparseBytes=%.0f MB (bucket-midpoint correction %.4f)"
          % (idx["sparse"], idx["sparseBytes"] / MB, byte_scale))
    if deep_ksum is None:
        print("no usable [BITSTAT-DEEP]: latency column omitted")
    print()

    # Bucket edges only: that is where the histogram answers exactly.
    model = filter_cost_model(deep_rows, deep_ksum) if deep_ksum else None
    if model:
        us_per_unit, us_sparse_fixed, us_dense_pair, kstar = model
        print("filter-kernel prices from the ablation, per batch at BS=4 on 2 cards, scaled to the")
        print("kSum THIS log sampled -- the ratios are what carry, not the absolute us:")
        print("  a sparse pair costs %.4g us * K + %.4g us,  the same pair dense costs %.4g us"
              % (us_per_unit, us_sparse_fixed, us_dense_pair))
        print("  so sparse is the cheaper way to store a posting only below K = %.1f" % kstar)
        ref_k, ref_n, _ = touched_at(float(1 << 20), deep_rows, deep_ksum)
        print("  (query-touched kMean here is %.1f)" % (deep_ksum / ref_n))
        # Self-check: drift means the constants and this log describe different runs.
        closes = us_per_unit * ref_k - (us_dense_pair - us_sparse_fixed) * ref_n
        print("  self-check: all-sparse gives %.1f us against the measured %.1f" % (closes, ABL_FULL - ABL_DEN0))
        print()

    hdr = "%-12s %8s %11s %9s" % ("threshold", "K<", "index MB", "vs den=0")
    if model:
        hdr += " %10s %11s" % ("touched", "filter +us")
    print(hdr)

    want = args.target_reduction
    flat_from = None
    below = None      # last edge under the target
    candidates = []   # every edge at or above it
    for edge in range(1, 14):
        cutoff = float(1 << edge)
        total_bytes, _sparse = index_bytes_at(cutoff, rows, total, segment_bytes, byte_scale,
                                             bytes_per_unit)
        reduction = 1.0 - total_bytes / base_bytes
        if reduction <= 0:
            continue
        t = cutoff / args.doc_num_per_segment
        # Above this density GetMaxPostingLength exceeds FilterOrOp's tile buffer and sparse-direct
        # is refused for the WHOLE index. A cliff, not a slope; nothing here models the far side.
        refused = min(args.doc_num_per_segment * t * bytes_per_unit,
                      args.doc_num_per_segment / 8.0 * (bytes_per_unit / 2)) > segment_bytes
        line = "%-12.6g %8.0f %11.0f %8.1f%%" % (t, cutoff, total_bytes / MB, reduction * 100)
        kept_frac = None
        p50_est = None
        if model:
            kept_k, kept_n, _total_n = touched_at(cutoff, deep_rows, deep_ksum)
            kept_frac = kept_k / deep_ksum
            p50_est = us_per_unit * kept_k - (us_dense_pair - us_sparse_fixed) * kept_n
            line += " %9.0f%% %11.1f" % (100.0 * kept_frac, p50_est)
        print(line + ("   <- sparse-direct refused above here" if refused else ""))
        if refused:
            continue  # cannot be recommended, but it is listed so the cliff is visible
        if reduction < want:
            below = (t, cutoff, total_bytes, reduction, kept_frac)
        else:
            candidates.append((t, cutoff, total_bytes, reduction, kept_frac, p50_est))
        # Past here latency is flat and the index keeps shrinking, so every larger edge DOMINATES.
        if kept_frac is not None and kept_frac >= 0.999 and not flat_from:
            flat_from = t
            print("(latency is flat from here: these queries never touch what a higher threshold")
            print(" compresses, so every larger edge below the sparse-direct cliff dominates)")

    print()
    if not candidates:
        print("no bucket edge reaches %.0f%% -- the whole index would have to compress."
              % (want * 100))
        return

    # Cheapest edge that clears the memory target, ties broken by the largest reduction. Without a
    # [BITSTAT-DEEP] line there is no latency column and this becomes the smallest qualifying edge.
    if candidates[0][5] is None:
        pick = candidates[0]
    else:
        pick = min(candidates, key=lambda c: (round(c[5], 1), -c[3]))
    t, cutoff, total_bytes, reduction, _kept, p50_est = pick
    print("recommended: DENSITY_THRESHOLD=%.6g" % t)
    print("  compresses K < %.0f, index %.0f MB/shard, %.1f%% under den=0"
          % (cutoff, total_bytes / MB, reduction * 100))
    if p50_est is not None:
        print("  adds %.1f us per batch to the filter kernel against den=0" % p50_est)
        ties = [c for c in candidates if round(c[5], 1) == round(p50_est, 1)]
        if len(ties) > 1:
            print("  the estimate is flat across K < %.0f .. %.0f, so this takes the whole"
                  % (min(c[1] for c in ties), max(c[1] for c in ties)))
            print("  memory saving rather than stopping at the %.0f%% asked for." % (want * 100))
    if below is not None and pick is candidates[0]:
        bt, bcut, bbytes, bred, _bk = below
        print("  the edge below it, %.6g (K < %.0f), gives only %.1f%% -- so the smallest"
              % (bt, bcut, bred * 100))
        print("  threshold that clears the target sits between them, in bucket [%d,%d)."
              % (int(bcut), int(cutoff)))
        print("  The histogram cannot place it: within a bucket the mass sits toward the high")
        print("  end, so a uniform split overstates compression (measured 79% of it). Resolving")
        print("  it costs one --convert-data 1 build; the edge above is the safe choice without one.")
    print()
    print("The threshold is an fr_converter argument, not fr_builder's -- it is baked into")
    print("WORK_DIR/builder_input at conversion time, so a new one needs --convert-data 1.")
    print("Reusing another threshold's builder_input silently rebuilds that threshold's index.")


if __name__ == "__main__":
    main()
