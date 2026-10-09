#!/usr/bin/env python3
"""Read the distributions gen_synthetic_data_v2.py was never calibrated against.

v2 was fitted to one statistic -- filter selectivity ~5%, from docs/query-analysis.md --
and it hits it by ORing ~200 values at F_SEL=0.00018. Real data can reach the same 5%
with far fewer, far commoner terms, and then every posting-length distribution differs
by two orders of magnitude while the calibrated number still matches. Measured on the
2-card synthetic corpus against a real one:

    query-referenced terms   synthetic p=0.00018 (K~24)   real p~0.01 (K 1024-2048)
    corpus bulk              synthetic p~0.0016  (K~207)  real p<=0.00012 (K<=16)

Inverted: synthetic query terms are 9x rarer than the corpus bulk, real ones are ~80x
commoner. That inversion is what makes DENSITY_THRESHOLD nearly free on one corpus and
expensive on the other, so a generator that gets it backwards cannot be used to tune it.

Three probes now print what is needed, none of which existed when v2 was written:

    BITLIST_STATS=2 BITLIST_STATS_CALLS=8   ->  [BITSTAT-INDEX]  corpus term frequencies
                                                [BITSTAT-DEEP]   query-referenced ones
    EXPR_STATS=1                            ->  [EXPRSTAT]       expression shape

Run a search with all three against the corpus being copied, then:

    tools/profile_for_synthetic_v3.py --log run.log

Use NUM_QUERIES=0. Anything else fails the harness's topk line-count check before a
single probe prints, and the sampling caps (BITLIST_STATS_CALLS, EXPR_STATS_EVERY) are
what bound the cost, not the query count.

Point it at the highest DENSITY_THRESHOLD index available: both histograms only cover
postings below the threshold the log ran at, so a low one truncates the very tail that
distinguishes one corpus from another.

It prints what each side implies and, at the end, a constants block for v3. It also
checks the two constraints against each other: term frequency and operands-per-OR-node
both drive selectivity, so they cannot be chosen independently. v2's numbers only look
consistent because it picked the frequency to make its own operand count land on 5%.
"""

import argparse
import math
import re
import sys


def parse_kv(line):
    out = {}
    for name in ("kHist", "andOrOperandHist", "conjPartHist"):
        m = re.search(name + r"(?:\([^)]*\))?=\[([0-9,\s]*)\]", line)
        if m:
            out[name] = [int(x) for x in m.group(1).split(",") if x.strip()]
    # nested {nodes=..,postings=..,stacks=..} groups, keyed by the name in front
    for op, body in re.findall(r"\b(and|or|not|conj)=\{([^}]*)\}", line):
        for k, v in re.findall(r"([a-z]+)=([0-9]+)", body):
            out["%s_%s" % (op, k)] = int(v)
    for key, val in re.findall(r"([A-Za-z][A-Za-z0-9]*)=(-?[0-9.]+)", line):
        if key not in out:
            out[key] = float(val) if "." in val else int(val)
    return out


def last_line(paths, tag):
    found = None
    for path in paths:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                if line.startswith(tag):
                    found = line.rstrip("\n")
    return found


def bucket_range(b):
    """kHist bucket b holds 2^(b-1) <= K < 2^b; bucket 0 holds K == 0."""
    return (0, 1) if b == 0 else (1 << (b - 1), 1 << b)


def docs_from_units(k, units):
    """A posting's doc count from the uint16 units it occupies.

    Each unit covers 16 docs, so D docs scattered over `units` slots leave
    units * (1 - (1 - 1/units)^D) of them non-empty. Inverting that is what turns a
    kHist bucket into a term frequency. Near K == units it goes to infinity, which is
    correct -- a saturated posting says nothing about how many docs are in it, only
    that there are many -- so it is clamped and flagged.
    """
    if k <= 0:
        return 0.0
    frac = min(k / float(units), 0.999)
    return -units * math.log(1.0 - frac)


def hist_rows(hist, units, doc_num_per_segment):
    """(K lo, K hi, count, mean K, mean freq) per non-empty bucket."""
    rows = []
    for b, count in enumerate(hist):
        if b == 0 or count == 0:
            continue
        lo, hi = bucket_range(b)
        mean_k = (lo + hi - 1) / 2.0
        freq = docs_from_units(mean_k, units) / doc_num_per_segment
        rows.append((lo, hi, count, mean_k, freq))
    return rows


def split_bulk_tail(rows):
    """Split the query-referenced terms into a bulk and a common tail.

    v2 models them as two populations -- F_SEL for the rare bulk, SEL_COMMON_FRAC of
    them at SEL_COMMON_FREQ -- and reporting one mean folds both into a number that is
    neither. The split is the widest run of empty buckets: a single Poisson population
    spills into at most the neighbouring bucket, so anything separated by two or more
    empty ones is a different population. Returns (bulk rows, tail rows).
    """
    if len(rows) < 2:
        return rows, []
    occupied = [int(math.log2(lo)) + 1 if lo else 0 for lo, _hi, _c, _m, _f in rows]
    gaps = [(occupied[i + 1] - occupied[i], i) for i in range(len(rows) - 1)]
    width, at = max(gaps)
    if width < 3:
        return rows, []
    return rows[:at + 1], rows[at + 1:]


def weighted_freq(rows):
    n = sum(c for _lo, _hi, c, _m, _f in rows)
    if n == 0:
        return 0.0, 0
    return sum(c * f for _lo, _hi, c, _m, f in rows) / n, n


def show_hist(title, rows, total_count):
    print(title)
    print("  %10s %10s %9s %12s" % ("K range", "count", "share", "implied freq"))
    for lo, hi, count, _mean_k, freq in rows:
        print("  %10s %10d %8.1f%% %12.6g"
              % ("[%d,%d)" % (lo, hi), count, 100.0 * count / total_count, freq))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", nargs="+", required=True,
                    help="logs holding [BITSTAT-INDEX], [BITSTAT-DEEP] and [EXPRSTAT]")
    ap.add_argument("--doc-num-per-segment", type=int, default=131072)
    ap.add_argument("--target-selectivity", type=float, default=0.05,
                    help="match fraction the generator has to reproduce (default 0.05)")
    args = ap.parse_args()

    units = args.doc_num_per_segment // 16  # each uint16 unit covers 16 docs
    idx_line = last_line(args.log, "[BITSTAT-INDEX]")
    deep_line = last_line(args.log, "[BITSTAT-DEEP]")
    expr_line = last_line(args.log, "[EXPRSTAT]")
    missing = [t for t, l in (("[BITSTAT-INDEX]", idx_line), ("[BITSTAT-DEEP]", deep_line),
                              ("[EXPRSTAT]", expr_line)) if l is None]
    if missing:
        print("missing %s -- rerun the search with BITLIST_STATS=2 BITLIST_STATS_CALLS=8 "
              "EXPR_STATS=1" % ", ".join(missing), file=sys.stderr)
        if idx_line is None:
            sys.exit(1)

    print("units per segment: %d (docNumPerSegment %d, 16 docs per uint16)"
          % (units, args.doc_num_per_segment))
    print()

    idx = parse_kv(idx_line)
    segments = idx.get("segments", 1)
    postings, dense, sparse = idx["postings"], idx.get("dense", 0), idx.get("sparse", 0)
    idx_rows = hist_rows(idx.get("kHist", []), units, args.doc_num_per_segment)
    print("== corpus (from [BITSTAT-INDEX], per shard) ==")
    print("  %d postings over %d segments -> ~%d terms present per segment"
          % (postings, segments, round(postings / float(max(segments, 1)))))
    if idx_rows:
        show_hist("  compressed terms, which is every term below the threshold this log ran at:",
                  idx_rows, sparse or 1)
    print("  %d terms were above it and are not in the histogram -- rerun at a higher"
          % dense)
    print("  DENSITY_THRESHOLD to see their frequencies, or treat them as one common bucket.")
    print()

    deep_rows, sel_freq, touched = [], None, None
    bulk_rows, tail_rows, bulk_freq, tail_freq, tail_frac = [], [], None, None, 0.0
    if deep_line:
        deep = parse_kv(deep_line)
        deep_rows = hist_rows(deep.get("kHist", []), units, args.doc_num_per_segment)
        touched = deep.get("converted", 0)
        if deep_rows and touched:
            show_hist("== query-referenced terms (from [BITSTAT-DEEP], per shard per query) ==",
                      deep_rows, touched)
            bulk_rows, tail_rows = split_bulk_tail(deep_rows)
            bulk_freq, bulk_n = weighted_freq(bulk_rows)
            tail_freq, tail_n = weighted_freq(tail_rows)
            sel_freq, _ = weighted_freq(deep_rows)
            tail_frac = tail_n / float(bulk_n + tail_n) if (bulk_n + tail_n) else 0.0
            print("  bulk %6.2f%% at freq %.6g          (v2's F_SEL = 0.00018)"
                  % (100.0 * (1 - tail_frac), bulk_freq))
            if tail_rows:
                print("  tail %6.2f%% at freq %.6g          (v2: SEL_COMMON_FRAC 0.004, FREQ 0.028)"
                      % (100.0 * tail_frac, tail_freq))
                print("  Two populations, so the %.6g mean over both is neither of them --"
                      % sel_freq)
                print("  a union of mixed frequencies is not a union at their average.")
            if touched < 200:
                print("  WARNING: only %d sampled. Raise BITLIST_STATS_CALLS and NUM_QUERIES"
                      % touched)
                print("  before trusting this -- it is the number the whole calibration turns on.")
            print()

    operands_per_or = None
    if expr_line:
        e = parse_kv(expr_line)
        exprs = e.get("exprs", 1)
        print("== expression shape (from [EXPRSTAT], per expression) ==")
        for op in ("and", "or", "not", "conj"):
            nodes = e.get("%s_nodes" % op, 0)
            posts = e.get("%s_postings" % op, 0)
            if nodes == 0 and posts == 0:
                continue
            print("  %-5s %6.2f nodes, %7.2f posting operands (%6.2f per node)"
                  % (op, nodes / float(exprs), posts / float(exprs),
                     posts / float(nodes) if nodes else 0.0))
        or_nodes, or_posts = e.get("or_nodes", 0), e.get("or_postings", 0)
        if or_nodes:
            operands_per_or = or_posts / float(or_nodes)
        hist = e.get("andOrOperandHist")
        if hist:
            labels = ["1", "2", "3", "4", "5-8", "9-16", "17-32", "32+"]
            print("  operands per AND/OR node: " +
                  ", ".join("%s=%d" % (l, c) for l, c in zip(labels, hist) if c))
        print()

    if sel_freq is not None and expr_line:
        e = parse_kv(expr_line)
        exprs = float(e.get("exprs", 1))
        or_posts = e.get("or_postings", 0) / exprs
        or_nodes = e.get("or_nodes", 0) / exprs
        or_stacks = e.get("or_stacks", 0) / exprs
        print("== consistency ==")
        # An OR node consuming stacks is taking another OR node's result, so the nodes nest into
        # one union rather than being ANDed together. Reading v2's tree as an AND of per-node
        # unions gives 5e-28 against a measured 5%.
        nested = or_stacks > 0.5 * or_nodes
        # Mixed frequencies: the union is 1 - prod(1 - p_i), not the union at their mean.
        if tail_rows:
            union_all = 1.0 - ((1.0 - bulk_freq) ** (or_posts * (1 - tail_frac)) *
                               (1.0 - tail_freq) ** (or_posts * tail_frac))
        else:
            union_all = 1.0 - (1.0 - sel_freq) ** or_posts
        per_node = 1.0 - (1.0 - sel_freq) ** operands_per_or if operands_per_or else 0.0
        anded = per_node ** or_nodes if or_nodes else per_node
        print("  %.2f OR nodes consuming %.2f stacks -> they %s"
              % (or_nodes, or_stacks, "nest into one union" if nested else "look independent"))
        mix = ("bulk %.6g + %.2f%% at %.6g" % (bulk_freq, 100 * tail_frac, tail_freq)
               if tail_rows else "freq %.6g" % sel_freq)
        print("  one union of all %.1f operands, %s -> %.4g" % (or_posts, mix, union_all))
        print("  the other reading, %.2f separate unions ANDed -> %.4g" % (or_nodes, anded))
        got = union_all if nested else anded
        print("  target %.4g, so the %s reading is the right one and it %s"
              % (args.target_selectivity, "nested" if nested else "independent",
                 "checks out" if 0.5 <= got / args.target_selectivity <= 2.0 else "does NOT"))
        print("  Read it as an upper bound: bucket midpoints inflate these frequencies about")
        print("  30% at this K, since a Poisson population centred in one bucket spills into")
        print("  the next and is then priced at that bucket's own midpoint.")
        if not (0.5 <= got / args.target_selectivity <= 2.0):
            print("  A gap here means independent terms are the wrong model -- real ones are")
            print("  correlated, or the tree differs. v3 has to close it while KEEPING the")
            print("  measured frequencies: selectivity alone does not pin them down, which is")
            print("  exactly the trap v2 fell into by fitting selectivity and nothing else.")
        print()

    print("== for gen_synthetic_data_v3.py ==")
    if sel_freq is not None:
        print("F_SEL = %.6g          # rare bulk, was 0.00018 in v2" % bulk_freq)
        if tail_rows:
            print("SEL_COMMON_FRAC = %.4g   # was 0.004" % tail_frac)
            print("SEL_COMMON_FREQ = %.4g   # was 0.028" % tail_freq)
    if operands_per_or:
        print("SEL_VALUES_PER_OR_NODE = %.1f    # v2 ORs ~200 into one union" % operands_per_or)
    if expr_line:
        e = parse_kv(expr_line)
        exprs = float(e.get("exprs", 1))
        print("OR_NODES_PER_QUERY = %.2f" % (e.get("or_nodes", 0) / exprs))
        print("AND_NODES_PER_QUERY = %.2f" % (e.get("and_nodes", 0) / exprs))
        print("NOT_NODES_PER_QUERY = %.2f    # operands: %.3f postings, %.3f stack"
              % (e.get("not_nodes", 0) / exprs, e.get("not_postings", 0) / exprs,
                 e.get("not_stacks", 0) / exprs))
        print("CONJ_NODES_PER_QUERY = %.2f" % (e.get("conj_nodes", 0) / exprs))
    if idx_rows:
        print("# density pool: (count per segment, freq), replacing v2's B_RARE/B_LOW/B_COMMON")
        print("DENSITY_BUCKETS = [")
        for lo, hi, count, _mean_k, freq in idx_rows:
            print("    (%7d, %.6g),   # K in [%d,%d)"
                  % (round(count / float(max(segments, 1))), freq, lo, hi))
        print("    (%7d, None),      # above the threshold this log ran at -- unmeasured"
              % round(dense / float(max(segments, 1))))
        print("]")


if __name__ == "__main__":
    main()
