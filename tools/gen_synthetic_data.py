#!/usr/bin/env python3
"""
Generate synthetic data that MIMICS the Huawei internal dataset's distribution,
so indexes can be built and tuned WITHOUT the internal data. Writes to --out-dir:

  dataset.bin     HYDSET2: [40B hdr][doc_num*dim FP32 vectors][doc_num*stride u64 bitmaps]
  tag_map.bin     [size_t n][ u32 len | utf8 "section#value" | i32 id ]...
  QueryData.txt   JSON-lines with the real record shape:
                  {result_num, main_tier:{json_query:"{vector:{relevance_learning2rank,
                   syntax_filter}}", retrieve_num}, ...}

Distribution targets (from docs/data-analysis.md + docs/query-sample.txt, real 10M):
  doc_num=10M, dim=64, tag_num=35672, 134 sections (sizes embedded below).
  density ~1.06% (avg ~378 tags/doc, right-skewed to ~2525).
  tag-freq buckets: ~79 always-on(>=99.9%), ~28 universal(90-99.9%),
    ~894 common(5-90%), ~2080 low(1-5%), ~15601 rare(<1%), ~16990 never-seen.
  set-bit concentration: top-1000 tags ~82.5% of set bits.
  doc vectors: L2 ~30 (UNNORMALIZED, right-skewed), component std ~4.3.
  queries: result_num=3500, vector L2 ~4.2, syntax_filter = AND of ~8 terms-groups
    over ~28-31 sections, ~700 tag refs/query (vals/sec avg ~20, max ~240), with
    some or/not nesting; referenced tags biased to common so filters are permissive.

Needs numpy. For --docs 10000000 the dataset.bin is ~47GB — run on the server
(e.g. under dtach), output into the xihe_v1 dir. Test small first: --docs 20000.

Usage:
  python3 gen_synthetic_data.py --out-dir /root/xihe_v1/npur_syn \
      --docs 10000000 --queries 10000
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

# 134 sections (name, #tags), summing to 35672 — the real tag_map section sizes.
SECTIONS = [("99072",5001),("99078",4146),("99036",4020),("99041",4020),("99009",2996),("99052",2971),("3000",2945),("2000",2133),("99097",1652),("99031",1064),("3001",770),("99011",370),("99013",369),("2002",368),("10000",366),("99004",342),("1006",337),("99088",222),("10015",205),("99033",187),("99038",184),("99003",184),("99095",143),("99043",142),("99051",117),("10003",80),("99002",56),("99059",24),("99034",24),("99017",20),("99058",15),("99070",14),("99076",10),("99090",10),("99092",8),("317",7),("307",7),("305",6),("316",6),("300",6),("99069",6),("99057",5),("99074",4),("309",3),("1",3),("303",3),("99098",2),("99091",2),("99093",2),("99085",2),("310",2),("306",2),("8000",2),("99068",2),("308",2),("8006",2),("10014",2),("99065",2),("99055",2),("304",1),("99049",1),("99039",1),("99075",1),("10004",1),("1008",1),("99056",1),("99077",1),("99050",1),("1002",1),("10008",1),("1009",1),("99001",1),("301",1),("99042",1),("3002",1),("99066",1),("99064",1),("99014",1),("99062",1),("99054",1),("99045",1),("2004",1),("312",1),("8003",1),("99048",1),("99012",1),("99063",1),("99046",1),("10010",1),("2",1),("99010",1),("99005",1),("313",1),("99037",1),("315",1),("99053",1),("10012",1),("314",1),("99035",1),("99000",1),("1004",1),("99071",1),("8004",1),("10007",1),("99060",1),("1001",1),("10006",1),("99040",1),("110002",1),("10011",1),("99087",1),("2003",1),("110001",1),("1000",1),("99073",1),("10005",1),("10009",1),("10013",1),("99067",1),("1003",1),("99047",1),("311",1),("99061",1),("8010",1),("1007",1),("10002",1),("8005",1),("99032",1),("99018",1),("99028",1),("99044",1),("2001",1),("110000",1),("10001",1)]

TAG_NUM = sum(sz for _, sz in SECTIONS)  # 35672
DIM = 64

# tag-frequency bucket counts (sum == TAG_NUM). Frequent tags get the LOW ids,
# which (since ids are assigned section-by-section) land in the big sections.
B_ALWAYS, B_UNIV, B_COMMON, B_LOW, B_RARE = 79, 28, 894, 2080, 15601
B_ABSENT = TAG_NUM - (B_ALWAYS + B_UNIV + B_COMMON + B_LOW + B_RARE)  # 16990


def build_tag_map(sections):
    """Return (id2term list, section->(start_id,size))."""
    id2term = []
    sec_span = {}
    idx = 0
    for name, size in sections:
        sec_span[name] = (idx, size)
        for v in range(size):
            id2term.append(f"{name}#{v}")
        idx += size
    return id2term, sec_span


def build_probs(rng):
    """Per-tag Bernoulli probability, ids ordered frequent-first, tuned to
    density ~378/doc and the real bucket structure. Returns float32 array."""
    p = np.zeros(TAG_NUM, dtype=np.float64)
    i = 0
    p[i:i+B_ALWAYS] = 1.0; i += B_ALWAYS
    p[i:i+B_UNIV] = rng.uniform(0.90, 0.999, B_UNIV); i += B_UNIV
    # common: mean ~0.21, a cluster near 0.30 with spread (matches the many ~30% tags)
    p[i:i+B_COMMON] = np.clip(rng.normal(0.24, 0.16, B_COMMON), 0.05, 0.90); i += B_COMMON
    p[i:i+B_LOW] = rng.uniform(0.01, 0.05, B_LOW); i += B_LOW
    p[i:i+B_RARE] = np.clip(rng.exponential(0.0016, B_RARE), 5e-5, 0.0099); i += B_RARE
    # remaining B_ABSENT stay 0 (never seen)
    # gently rescale the non-fixed part; base target is below 378 because the
    # per-doc "rich" factor (write_dataset) inflates the realized mean back to ~378.
    target = 330.0
    fixed = p[:B_ALWAYS + B_UNIV].sum()
    rest = slice(B_ALWAYS + B_UNIV, B_ALWAYS + B_UNIV + B_COMMON + B_LOW + B_RARE)
    cur = p[rest].sum()
    if cur > 0:
        p[rest] = np.clip(p[rest] * (target - fixed) / cur, 0.0, 0.95)
    return p.astype(np.float32)


def write_tag_map(path, id2term):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(id2term)))
        for i, term in enumerate(id2term):
            b = term.encode("utf-8")
            f.write(struct.pack("<I", len(b)))
            f.write(b)
            f.write(struct.pack("<i", i))


def write_dataset(path, doc_num, p, rng, chunk, log):
    stride = (TAG_NUM + 63) // 64
    pad = stride * 64 - TAG_NUM
    with open(path, "wb") as f:
        f.write(b"HYDSET2\0")
        f.write(struct.pack("<IIQIIII", 2, 0, doc_num, DIM, TAG_NUM, 0, 0))
        # --- vectors region: L2 ~30, per-doc lognormal scale for the right skew ---
        done = 0
        while done < doc_num:
            n = min(chunk, doc_num - done)
            scale = np.exp(rng.normal(0.0, 0.5, (n, 1))).astype(np.float32)
            vecs = (rng.standard_normal((n, DIM), dtype=np.float32) * np.float32(3.4)) * scale
            f.write(vecs.tobytes())
            done += n
            if done % (chunk * 50) == 0 or done == doc_num:
                log(f"  vectors {done}/{doc_num}")
        # --- bitmaps region: per-doc Bernoulli, ~10% "rich" docs give the tag tail ---
        done = 0
        while done < doc_num:
            n = min(chunk, doc_num - done)
            fdoc = np.ones((n, 1), dtype=np.float32)
            rich = rng.random(n) < 0.05          # ~5% tag-rich docs give the long tail
            if rich.any():
                fdoc[rich, 0] = rng.uniform(2.5, 9.0, int(rich.sum())).astype(np.float32)
            peff = np.minimum(p[None, :] * fdoc, np.float32(1.0))
            hits = (rng.random((n, TAG_NUM), dtype=np.float32) < peff).astype(np.uint8)
            if pad:
                hits = np.pad(hits, ((0, 0), (0, pad)))
            f.write(np.packbits(hits, axis=1, bitorder="little").tobytes())
            done += n
            if done % (chunk * 50) == 0 or done == doc_num:
                log(f"  bitmaps {done}/{doc_num}")


def _terms_group(rng, sec_span, sec_weights, sec_names, n_sec, refs_budget):
    """One `terms` node: pick n_sec sections, each an OR-list of value indices
    biased toward LOW indices (= frequent tags). Returns (node, refs_used)."""
    group = {"join_type": "or", "inner_section_join_type": "or"}
    picks = rng.choice(len(sec_names), size=min(n_sec, len(sec_names)),
                       replace=False, p=sec_weights)
    used = 0
    for si in picks:
        name = sec_names[si]
        start, size = sec_span[name]
        k = 1 + int(rng.random() ** 3 * min(size, 200))  # skew: many small, few large
        k = max(1, min(k, size, max(1, refs_budget - used)))
        # bias to low value-index (frequent). geometric-ish over [0,size)
        idx = np.unique((rng.random(k * 2) ** 2 * size).astype(int))[:k]
        vals = [str(int(v)) for v in idx]
        if rng.random() < 0.5:
            vals.append("delivery_all")  # catch value (often absent from tag_map)
        group[name] = vals
        used += len(idx)
        if used >= refs_budget:
            break
    return {"terms": group}, used


def build_filter(rng, sec_span):
    """syntax_filter ~ AND of ~8 terms-groups (~28 sections, ~700 refs), with a
    couple wrapped in OR and one negated — matching the real node profile."""
    sec_names = [n for n, _ in SECTIONS]
    w = np.array([sz for _, sz in SECTIONS], dtype=np.float64)
    w = w / w.sum()  # weight section choice by size -> hits big/frequent sections
    total = rng.integers(550, 860)  # tag refs/query
    groups = []
    used = 0
    n_groups = 8
    for g in range(n_groups):
        budget = max(20, (total - used) // max(1, (n_groups - g)))
        node, u = _terms_group(rng, sec_span, w, sec_names, rng.integers(2, 5), budget)
        groups.append(node)
        used += u
    # assemble: AND[ t0, t1, t2, OR[t3,t4], OR[t5,t6], NOT[t7] ]
    children = groups[:3]
    children.append({"or": groups[3:5]})
    children.append({"or": groups[5:7]})
    children.append({"not": groups[7]})
    return {"and": children}


def write_queries(path, q, sec_span, rng, log):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(q):
            vec = (rng.standard_normal(DIM) * 0.525).round(6).tolist()  # L2 ~4.2
            inner = {"vector": {
                "relevance_learning2rank": vec,
                "record_name": "relevance_learning2rank",
                "cluster_num": 64, "max_cluster_num": 128, "result_num": 3500,
                "max_distance_score_doc_num": 3500, "parallel_score": True,
                "syntax_filter": build_filter(rng, sec_span),
            }}
            rec = {"corpus": "synthetic", "result_num": 3500,
                   "main_tier": {"json_query": json.dumps(inner, separators=(",", ":")),
                                 "retrieve_num": 3500},
                   "request_id": f"syn-{i}"}
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            if (i + 1) % 2000 == 0:
                log(f"  queries {i + 1}/{q}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--docs", type=int, default=10_000_000)
    ap.add_argument("--queries", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=20260706)
    ap.add_argument("--chunk", type=int, default=2000, help="docs per generation chunk")
    ap.add_argument("--skip-dataset", action="store_true")
    a = ap.parse_args()

    def log(m):
        sys.stderr.write(m + "\n")
        sys.stderr.flush()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    id2term, sec_span = build_tag_map(SECTIONS)
    p = build_probs(rng)
    log(f"tag_num={TAG_NUM} sections={len(SECTIONS)} "
        f"E[tags/doc]={float(p.sum()):.1f} density={100*float(p.sum())/TAG_NUM:.3f}% "
        f"bitset-tags(>=5%)={int((p>=0.05).sum())}")

    log(f"writing {out/'tag_map.bin'}")
    write_tag_map(out / "tag_map.bin", id2term)

    log(f"writing {out/'QueryData.txt'} ({a.queries} queries)")
    write_queries(out / "QueryData.txt", a.queries, sec_span, rng, log)

    if not a.skip_dataset:
        stride = (TAG_NUM + 63) // 64
        gb = (40 + a.docs * (DIM * 4 + stride * 8)) / 1e9
        log(f"writing {out/'dataset.bin'} ({a.docs} docs, ~{gb:.1f} GB)")
        write_dataset(out / "dataset.bin", a.docs, p, rng, a.chunk, log)

    log("done. Verify with: python3 tools/analyze_querydata.py "
        f"--query {out/'QueryData.txt'} --tag-map {out/'tag_map.bin'} "
        f"--dataset {out/'dataset.bin'} --top 50")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
