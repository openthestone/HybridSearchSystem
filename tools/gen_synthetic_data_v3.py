#!/usr/bin/env python3
"""
Synthetic data generator v3 — calibrated to the referenced-tag FREQUENCY distribution,
not only to filter selectivity. Writes dataset.bin (HYDSET2) + tag_map.bin +
QueryData.txt, same formats as v1/v2, which are kept.

v2 fitted one number: match fraction ~5%. It reached it by ORing ~200 values at a flat
F_SEL=0.00018 and capping every 'common' value at 0.12. That hits 5% and still gets the
posting-length distribution wrong, because many combinations of term frequency and
operand count give 5%. Posting length is what DENSITY_THRESHOLD trades against, so a
corpus that matches selectivity but not lengths cannot be used to tune it -- measured,
the threshold cost 37% of p50 on v2's corpus and was nearly free on a real one.

What docs/archive/query-analysis.md actually says about referenced tags (line 352-365),
against what v2 does:

    referenced-tag freq   p50 0.0003, mean 0.0873, p90 0.297, p99 1, max 1
    'common' (>=5%)       14.8% of references        v2: SEL_COMMON_FRAC 0.004  (37x low)
    top of the range      5.3% of refs at >=0.92     v2: clipped at 0.12
    OR nodes / query      2.77                       v2 measured 11.68          (4.2x high)
    terms nodes / query   7.54                       v2 ~8                      ok
    tag refs / query      p50 784, normal-ish        v2 normal(780,290)         ok
    resolution            39.88%                     v2 40%                     ok
    match fraction        p50 0.0497, none >=90%     v2 calibrated to this      ok

So v2's rare bulk was roughly right (0.00018 against 0.0003) and the real miss is the
common population: 14.8% of references, reaching 1.0, which v2 has almost none of.

The two do not conflict. Per DISTINCT tag the picture is ~30 common tags referenced by
nearly every query -- 292148 common references / ~10000 queries each -- against ~45400
rare ones referenced ~37 times apiece. The common ones are the anchor values (304#0,
305#1, 307#32771, 99017#13 ...), and they sit in PERMISSIVE clauses of the top-level
AND, where a frequency near 1 restricts nothing. That is why the real data can hold
tags at freq 1.0 and still never match >=90% of docs, and it is what v2's 0.12 cap was
working around instead of modelling.

Selectivity therefore comes from the rare OR-union alone: ~154 resolved rare references
at ~0.0003 give 1-(1-0.0003)^154 = 4.5%, against the measured 0.0497.

Needs numpy. 10M dataset.bin ~47GB (run on the server, e.g. under dtach).
Test small first: --docs 20000 --queries 200.

Usage:
  python3 gen_synthetic_data_v3.py --out-dir /root/xihe_v1/npur_syn_v3 \\
      --docs 10000000 --queries 10800

VERIFY BEFORE BUILDING 10M DOCS. Generate small, build an index, and run a search with
BITLIST_STATS=2 BITLIST_STATS_CALLS=8 EXPR_STATS=1 NUM_QUERIES=0, then:

  tools/profile_for_synthetic_v3.py --log run.log

OR_NODES_PER_QUERY has to come back near 2.77 and F_SEL near 0.0003. SEL_OR_NODES below
is the knob for the first: how the engine expands a terms node into OR nodes is not
derivable from the query JSON, so it is set by measurement, not by reading the code.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

# 134 real tag_map section sizes (name, #tags), sum == 35672. Unchanged from v2.
SECTIONS = [("99072",5001),("99078",4146),("99036",4020),("99041",4020),("99009",2996),("99052",2971),("3000",2945),("2000",2133),("99097",1652),("99031",1064),("3001",770),("99011",370),("99013",369),("2002",368),("10000",366),("99004",342),("1006",337),("99088",222),("10015",205),("99033",187),("99038",184),("99003",184),("99095",143),("99043",142),("99051",117),("10003",80),("99002",56),("99059",24),("99034",24),("99017",20),("99058",15),("99070",14),("99076",10),("99090",10),("99092",8),("317",7),("307",7),("305",6),("316",6),("300",6),("99069",6),("99057",5),("99074",4),("309",3),("1",3),("303",3),("99098",2),("99091",2),("99093",2),("99085",2),("310",2),("306",2),("8000",2),("99068",2),("308",2),("8006",2),("10014",2),("99065",2),("99055",2),("304",1),("99049",1),("99039",1),("99075",1),("10004",1),("1008",1),("99056",1),("99077",1),("99050",1),("1002",1),("10008",1),("1009",1),("99001",1),("301",1),("99042",1),("3002",1),("99066",1),("99064",1),("99014",1),("99062",1),("99054",1),("99045",1),("2004",1),("312",1),("8003",1),("99048",1),("99012",1),("99063",1),("99046",1),("10010",1),("2",1),("99010",1),("99005",1),("313",1),("99037",1),("315",1),("99053",1),("10012",1),("314",1),("99035",1),("99000",1),("1004",1),("99071",1),("8004",1),("10007",1),("99060",1),("1001",1),("10006",1),("99040",1),("110002",1),("10011",1),("99087",1),("2003",1),("110001",1),("1000",1),("99073",1),("10005",1),("10009",1),("10013",1),("99067",1),("1003",1),("99047",1),("311",1),("99061",1),("8010",1),("1007",1),("10002",1),("8005",1),("99032",1),("99018",1),("99028",1),("99044",1),("2001",1),("110000",1),("10001",1)]

# real per-section share of a query's tag refs ("sections referenced", line 108-145).
SECTION_REF_WEIGHTS = {"3000":2708023,"3001":2698400,"99002":860316,"10015":290402,"99046":282884,"99047":273403,"99051":232666,"99052":223108,"99036":126848,"305":88326,"307":69628,"99011":42013,"10000":41343,"99017":40720,"2002":36867,"99005":36585,"2000":32696,"99012":31125,"99018":30536,"10004":30455,"99073":28295,"2003":27353,"2001":21808,"10011":21776,"304":21776,"1006":21776,"99000":21776,"99043":20609,"99070":18419,"99001":10888,"10012":10888,"99041":10888,"99033":3029,"10003":654,"99068":486,"99072":146,"99055":88,"99009":52}

# sections whose value lists are mirrored (referenced with identical values).
PAIRS = {"3000": "3001", "99046": "99047", "99051": "99052"}

DIM = 64
TAG_NUM = sum(sz for _, sz in SECTIONS)  # 35672

# SELECTIVE sections carry the rare values whose OR-union is the whole selectivity.
SELECTIVE_SECTIONS = ["3000", "3001", "99002", "99046", "99047", "99051", "99052",
                      "10015", "99036"]
# ANCHOR sections hold the common tags, in PERMISSIVE clauses of the top-level AND, so a
# frequency near 1 restricts nothing.
ANCHOR_SECTIONS = ["304", "305", "307", "99017", "99018", "99000", "99001", "1006",
                   "2000", "2001", "2003", "10000", "10004", "10011", "10012",
                   "99005", "99011", "99012", "99043", "99073"]

# Rare referenced tags: exponential clipped to the measured [5e-5, 0.05]. The mean is set so the
# 58.7th percentile lands on the measured p50 of 0.0003 -- not 50, because 14.8% of references
# are common and sit above the whole distribution's median.
F_RARE_MEAN = 0.00034
F_RARE_MIN, F_RARE_MAX = 5e-05, 0.05

# Common referenced tags: the >=0.0834 part of the measured histogram, as (lo, hi, share).
COMMON_FREQ_HIST = [
    (0.08338, 0.1667, 11367), (0.1667, 0.25, 58707), (0.25, 0.3334, 16199),
    (0.3334, 0.4167, 15067), (0.4167, 0.5, 19109), (0.5, 0.5834, 9558),
    (0.5834, 0.6667, 21782), (0.6667, 0.75, 12911), (0.75, 0.8333, 928),
    (0.8333, 0.9167, 2877), (0.9167, 1.0, 104348),
]

# Corpus-wide density for tags no query references. NOT measured -- carried over from v2. To fix,
# take DENSITY_BUCKETS from profile_for_synthetic_v3.py against the real index.
B_ALWAYS, B_UNIV, B_COMMON, B_LOW, B_RARE = 79, 28, 894, 2080, 15601

# Query shape. OR nodes per query must be checked against [EXPRSTAT] after generating: v2's 3
# selective terms nodes came back as 11.68 OR nodes against the real 2.77.
ANCHOR_COMMON_PER_SECTION = 2
# Distinct resolved tags: a mirrored section pair resolves twice to the same tag.
DISTINCT_RESOLVED = 181

SEL_OR_NODES = 1          # selective terms nodes inside the one OR wrapper
PERM_CLAUSES_MIN, PERM_CLAUSES_MAX = 5, 6
REFS_MEAN, REFS_SD = 780, 290
REFS_MIN, REFS_MAX = 67, 2046
RESOLVE_FRAC = 0.3988     # measured 3356969/8417051
COMMON_REF_FRAC = 0.148   # measured 292148/1969191


def build_tag_map(sections):
    id2term, sec_span, idx = [], {}, 0
    for name, size in sections:
        sec_span[name] = (idx, size)
        for v in range(size):
            id2term.append(f"{name}#{v}")
        idx += size
    return id2term, sec_span


def build_probs(rng, sec_span):
    """Per-tag doc frequency, in three populations rather than v2's one.

    SELECTIVE values are rare and are what the OR-union is built from. ANCHOR values
    are common, drawn from the measured histogram, and are referenced permissively.
    Everything else carries corpus density without being referenced at all.
    """
    p = np.zeros(TAG_NUM, dtype=np.float64)

    for s in SELECTIVE_SECTIONS:
        start, size = sec_span.get(s, (None, 0))
        if start is None or size == 0:
            continue
        p[start:start + size] = np.clip(rng.exponential(F_RARE_MEAN, size),
                                        F_RARE_MIN, F_RARE_MAX)

    # Anchor sections are small (305 has 6 values, 307 has 7, 304 has 1), which is why ~30
    # distinct tags account for 14.8% of all references.
    weights = np.array([c for _lo, _hi, c in COMMON_FREQ_HIST], dtype=np.float64)
    weights /= weights.sum()
    for s in ANCHOR_SECTIONS:
        start, size = sec_span.get(s, (None, 0))
        if start is None or size == 0:
            continue
        n = min(ANCHOR_COMMON_PER_SECTION, size)
        picks = rng.choice(len(COMMON_FREQ_HIST), size=n, p=weights)
        for j, b in enumerate(picks):
            lo, hi, _c = COMMON_FREQ_HIST[b]
            p[start + j] = rng.uniform(lo, hi)

    # Density pool on tags nobody references, so it cannot inflate the match fraction.
    pool = []
    for name, size in SECTIONS:
        if name in set(SELECTIVE_SECTIONS):
            continue
        start, _ = sec_span[name]
        pool.extend(range(start, start + size))
    pool = np.array([gi for gi in pool if p[gi] == 0.0], dtype=np.int64)
    rng.shuffle(pool)
    buckets = [
        (B_ALWAYS, lambda n: np.ones(n)),
        (B_UNIV, lambda n: rng.uniform(0.90, 0.999, n)),
        (B_COMMON, lambda n: np.clip(rng.normal(0.24, 0.16, n), 0.05, 0.90)),
        (B_LOW, lambda n: rng.uniform(0.01, 0.05, n)),
        (B_RARE, lambda n: np.clip(rng.exponential(0.0016, n), 5e-5, 0.0099)),
    ]
    i = 0
    for cnt, gen in buckets:
        cnt = min(cnt, max(0, len(pool) - i))
        if cnt > 0:
            p[pool[i:i + cnt]] = gen(cnt)
            i += cnt
    return p.astype(np.float32)


def write_tag_map(path, id2term):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(id2term)))
        for i, term in enumerate(id2term):
            b = term.encode("utf-8")
            f.write(struct.pack("<I", len(b)))
            f.write(b)
            f.write(struct.pack("<i", i))


def _gen_range_worker(args):
    """Generate docs [start,end) and write them at their file offsets. Vectors and
    bitmaps are separate contiguous regions, so a doc-range maps to two contiguous
    byte-ranges; disjoint ranges => processes never overlap. Own RNG per worker."""
    path, start, end, p_bytes, doc_num, seed, wid, chunk = args
    p = np.frombuffer(p_bytes, dtype=np.float32)
    stride = (TAG_NUM + 63) // 64
    pad = stride * 64 - TAG_NUM
    vec_off = 40
    bmp_off = 40 + doc_num * DIM * 4
    rng = np.random.default_rng([seed, wid])
    with open(path, "r+b") as f:
        f.seek(vec_off + start * DIM * 4)
        d = start
        while d < end:
            n = min(chunk, end - d)
            scale = np.exp(rng.normal(0.0, 0.5, (n, 1))).astype(np.float32)
            f.write(((rng.standard_normal((n, DIM), dtype=np.float32) * np.float32(3.4)) * scale).tobytes())
            d += n
        f.seek(bmp_off + start * stride * 8)
        d = start
        while d < end:
            n = min(chunk, end - d)
            fdoc = np.ones((n, 1), dtype=np.float32)
            rich = rng.random(n) < 0.05
            if rich.any():
                fdoc[rich, 0] = rng.uniform(2.5, 9.0, int(rich.sum())).astype(np.float32)
            peff = np.minimum(p[None, :] * fdoc, np.float32(1.0))
            hits = (rng.random((n, TAG_NUM), dtype=np.float32) < peff).astype(np.uint8)
            if pad:
                hits = np.pad(hits, ((0, 0), (0, pad)))
            f.write(np.packbits(hits, axis=1, bitorder="little").tobytes())
            d += n
            if d % (chunk * 200) < chunk or d >= end:
                sys.stderr.write(f"  [w{wid}] {d - start}/{end - start}\n")
                sys.stderr.flush()
    return end - start


def write_dataset(path, doc_num, p, seed, chunk, jobs, log):
    stride = (TAG_NUM + 63) // 64
    total = 40 + doc_num * (DIM * 4 + stride * 8)
    with open(path, "wb") as f:            # header, then pre-size so workers can seek
        f.write(b"HYDSET2\0")
        f.write(struct.pack("<IIQIIII", 2, 0, doc_num, DIM, TAG_NUM, 0, 0))
    with open(path, "r+b") as f:
        f.truncate(total)
    jobs = max(1, jobs)
    bounds = [(doc_num * i // jobs, doc_num * (i + 1) // jobs) for i in range(jobs)]
    args = [(str(path), s, e, p.tobytes(), doc_num, seed, wid, chunk)
            for wid, (s, e) in enumerate(bounds)]
    if jobs == 1:
        _gen_range_worker(args[0])
        return
    import multiprocessing as mp
    log(f"  generating dataset with {jobs} processes ({total / 1e9:.1f} GB)...")
    with mp.Pool(jobs) as pool:
        done = 0
        for got in pool.imap_unordered(_gen_range_worker, args):
            done += got
            log(f"  dataset: {done}/{doc_num} docs done")


def _sel_values(rng, sizes):
    """Referenced values, one per entry of `sizes`. RESOLVE_FRAC of them are in-range
    indices; the rest are absent huge ids or 'delivery_all', skipped at conversion,
    which is what makes the real resolution rate 39.88%. Absent values still count as
    references.

    Drawn as whole arrays rather than per value: a query carries ~660 of these and the
    scalar version spent all its time in numpy's per-call overhead, ~2000 RNG calls per
    query. Same distributions, ~50x faster.
    """
    n = len(sizes)
    if n == 0:
        return []
    resolves = rng.random(n) < RESOLVE_FRAC
    idx = (rng.random(n) ** 2 * sizes).astype(np.int64)
    huge = rng.integers(10 ** 12, 10 ** 16, n)
    absent_is_id = rng.random(n) < 0.6
    return [str(idx[i]) if resolves[i]
            else (str(huge[i]) if absent_is_id[i] else "delivery_all")
            for i in range(n)]


def build_filter_v3(rng, sec_span):
    """One OR-union of rare selective values, ANDed with permissive anchor clauses.

    The selectivity is the union alone: ~154 resolved rare references at ~0.0003 give
    4.5% against the measured 4.97%. The anchor clauses carry the common population --
    14.8% of references, up to freq 1.0 -- in positions where that restricts nothing,
    which is the whole reason the real data has both freq-1.0 tags and no filter
    matching >=90%.
    """
    total = int(np.clip(rng.normal(REFS_MEAN, REFS_SD), REFS_MIN, REFS_MAX))
    n_common = int(total * COMMON_REF_FRAC)
    n_sel = total - n_common

    selw = np.array([SECTION_REF_WEIGHTS.get(s, 1.0) for s in SELECTIVE_SECTIONS],
                    dtype=np.float64)
    selw /= selw.sum()
    picks = rng.choice(len(SELECTIVE_SECTIONS), size=n_sel, p=selw)
    sizes = np.array([sec_span.get(SELECTIVE_SECTIONS[si], (0, 1))[1] for si in picks],
                     dtype=np.int64)
    values = _sel_values(rng, sizes)
    bysec: dict = {}
    for si, v in zip(picks, values):
        s = SELECTIVE_SECTIONS[si]
        bysec.setdefault(s, []).append(v)
        if s in PAIRS:
            bysec.setdefault(PAIRS[s], []).append(v)

    items = list(bysec.items())
    rng.shuffle(items)
    n_nodes = min(SEL_OR_NODES, len(items)) or 1

    def or_terms(chunk):
        node = {"join_type": "or", "inner_section_join_type": "or"}
        for sec, vals in chunk:
            node[sec] = list(dict.fromkeys(vals))
        return {"terms": node}

    sel_nodes = [or_terms(items[gi::n_nodes]) for gi in range(n_nodes)]

    n_clauses = int(rng.integers(PERM_CLAUSES_MIN, PERM_CLAUSES_MAX + 1))
    per_clause = max(1, n_common // max(n_clauses, 1))
    chosen = rng.choice(ANCHOR_SECTIONS, size=min(n_clauses, len(ANCHOR_SECTIONS)),
                        replace=False)
    perm = []
    for s in chosen:
        _, size = sec_span.get(s, (0, 1))
        vals = sorted({str(v) for v in rng.integers(0, size, per_clause)})
        perm.append({"terms": {"join_type": "or", "inner_section_join_type": "or",
                               s: vals or ["0"]}})

    # Mild NOT on rare selective values -> excludes ~1%, matching the real one-NOT tree.
    ns = str(rng.choice(SELECTIVE_SECTIONS))
    _, nsz = sec_span.get(ns, (0, 1))
    nv = _sel_values(rng, np.full(20, nsz, dtype=np.int64))
    not_node = {"terms": {"join_type": "or", "inner_section_join_type": "or",
                          ns: nv or ["0"]}}

    return {"and": [{"or": sel_nodes}] + perm + [{"not": not_node}]}


def write_queries_v3(path, q, sec_span, rng, log):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(q):
            vec = (rng.standard_normal(DIM) * 0.525).round(6).tolist()   # L2 ~4.2
            inner = {"vector": {
                "relevance_learning2rank": vec,
                "record_name": "relevance_learning2rank",
                "cluster_num": 64, "max_cluster_num": 128, "result_num": 3500,
                "max_distance_score_doc_num": 3500, "parallel_score": True,
                "syntax_filter": build_filter_v3(rng, sec_span),
            }}
            rec = {"corpus": "synthetic-v3", "result_num": 3500,
                   "main_tier": {"json_query": json.dumps(inner, separators=(",", ":")),
                                 "retrieve_num": 3500},
                   "request_id": f"synv3-{i}"}
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            if (i + 1) % 2000 == 0:
                log(f"  queries {i + 1}/{q}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--docs", type=int, default=10_000_000)
    # The real QueryData_10000.txt holds 10908 records, 10888 of them with a filter.
    ap.add_argument("--queries", type=int, default=10_800)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=16,
                    help="parallel processes for dataset.bin (1 = serial)")
    ap.add_argument("--skip-dataset", action="store_true")
    a = ap.parse_args()

    def log(m):
        sys.stderr.write(m + "\n")
        sys.stderr.flush()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    id2term, sec_span = build_tag_map(SECTIONS)
    p = build_probs(rng, sec_span)

    sel_ids = [gi for s in SELECTIVE_SECTIONS
               for gi in range(*(lambda st, sz: (st, st + sz))(*sec_span[s]))]
    anc_ids = [sec_span[s][0] + j for s in ANCHOR_SECTIONS
               for j in range(min(ANCHOR_COMMON_PER_SECTION, sec_span[s][1]))]
    sel_p = np.sort(p[sel_ids])
    log(f"tag_num={TAG_NUM} sections={len(SECTIONS)} E[tags/doc]={float(p.sum()):.1f} "
        f"density={100 * float(p.sum()) / TAG_NUM:.3f}%")
    # The measured p50 is over ALL referenced tags, so the rare population's own 58.7th percentile
    # is what has to be 0.0003. Comparing its median would say the calibration is 22% low.
    log(f"selective tags={len(sel_ids)} freq p50={np.median(sel_p):.6g} "
        f"p58.7={np.quantile(sel_p, 0.587):.6g} (target 0.0003) max={sel_p[-1]:.4g}")
    log(f"anchor common tags={len(anc_ids)} (target ~30) freq p50={np.median(p[anc_ids]):.4g} "
        f">=0.92 share={float((p[anc_ids] >= 0.9167).mean()):.3f}")
    n_rare = int(DISTINCT_RESOLVED * (1 - COMMON_REF_FRAC))
    union = 1.0 - float(np.prod(1.0 - rng.choice(sel_p, size=n_rare)))
    log(f"estimated match fraction: {union:.4f} from {n_rare} distinct rare tags "
        f"(target 0.0497)")

    log(f"writing {out/'tag_map.bin'}")
    write_tag_map(out / "tag_map.bin", id2term)
    log(f"writing {out/'QueryData.txt'} ({a.queries} queries)")
    write_queries_v3(out / "QueryData.txt", a.queries, sec_span, rng, log)

    if not a.skip_dataset:
        stride = (TAG_NUM + 63) // 64
        gb = (40 + a.docs * (DIM * 4 + stride * 8)) / 1e9
        log(f"writing {out/'dataset.bin'} ({a.docs} docs, ~{gb:.1f} GB, jobs={a.jobs})")
        write_dataset(out / "dataset.bin", a.docs, p, a.seed, a.chunk, a.jobs, log)

    log("verify the shape before trusting it:")
    log(f"  python3 tools/analyze_query.py --query {out/'QueryData.txt'} "
        f"--tag-map {out/'tag_map.bin'} --dataset {out/'dataset.bin'} --top 50")
    log("  then build an index and run a search with BITLIST_STATS=2 "
        "BITLIST_STATS_CALLS=8 EXPR_STATS=1 NUM_QUERIES=0,")
    log("  and check OR_NODES_PER_QUERY ~ 2.77 with "
        "tools/profile_for_synthetic_v3.py --log run.log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
