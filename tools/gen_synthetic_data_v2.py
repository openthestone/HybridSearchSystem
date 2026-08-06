#!/usr/bin/env python3
"""
Synthetic data generator v2 — closer to the Huawei internal distribution than
gen_synthetic_data.py (v1). Writes dataset.bin (HYDSET2) + tag_map.bin +
QueryData.txt. v1 is kept; this does NOT overwrite it (different filename;
default --out-dir differs too).

Calibrated so the estimated filter selectivity matches the real internal data
(docs/query-analysis.md, computed by analyze_query.py's eval_prob): match fraction
p50~0.05 / p90~0.10 / p99~0.145 / max~0.20, NEVER empty, NEVER >=90%.

The selectivity comes from ORing ~200 RARE values of the heavily-referenced
SELECTIVE_SECTIONS (mostly 3000/3001) into ONE union (~5%); CARRIER_SECTIONS add
an always-on value (permissive, P~1) and one mild NOT trims ~1%. The critical
bug this fixes: earlier versions (a) assigned dataset frequencies by global tag
index so the referenced sections were all absent -> 93% empty; and (b) split the
selective values across AND-joined terms nodes, destroying the OR-union. Both are
fixed here: present tags live in the referenced sections, and the selective
values stay OR'd. Other real traits kept: tag refs/query ~normal(780,290),
~40% resolution (60% of values are absent huge-ids/delivery_all -> skipped),
value pairing (3000<->3001 etc.), depth-2 tree, doc vectors L2~30.

Needs numpy. 10M dataset.bin ~47GB (run on the server, e.g. under dtach).
Test small first: --docs 20000 --queries 200.

Usage:
  python3 gen_synthetic_data_v2.py --out-dir /root/xihe_v1/npur_syn_v2 \
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

# 134 real tag_map section sizes (name, #tags), sum == 35672.
SECTIONS = [("99072",5001),("99078",4146),("99036",4020),("99041",4020),("99009",2996),("99052",2971),("3000",2945),("2000",2133),("99097",1652),("99031",1064),("3001",770),("99011",370),("99013",369),("2002",368),("10000",366),("99004",342),("1006",337),("99088",222),("10015",205),("99033",187),("99038",184),("99003",184),("99095",143),("99043",142),("99051",117),("10003",80),("99002",56),("99059",24),("99034",24),("99017",20),("99058",15),("99070",14),("99076",10),("99090",10),("99092",8),("317",7),("307",7),("305",6),("316",6),("300",6),("99069",6),("99057",5),("99074",4),("309",3),("1",3),("303",3),("99098",2),("99091",2),("99093",2),("99085",2),("310",2),("306",2),("8000",2),("99068",2),("308",2),("8006",2),("10014",2),("99065",2),("99055",2),("304",1),("99049",1),("99039",1),("99075",1),("10004",1),("1008",1),("99056",1),("99077",1),("99050",1),("1002",1),("10008",1),("1009",1),("99001",1),("301",1),("99042",1),("3002",1),("99066",1),("99064",1),("99014",1),("99062",1),("99054",1),("99045",1),("2004",1),("312",1),("8003",1),("99048",1),("99012",1),("99063",1),("99046",1),("10010",1),("2",1),("99010",1),("99005",1),("313",1),("99037",1),("315",1),("99053",1),("10012",1),("314",1),("99035",1),("99000",1),("1004",1),("99071",1),("8004",1),("10007",1),("99060",1),("1001",1),("10006",1),("99040",1),("110002",1),("10011",1),("99087",1),("2003",1),("110001",1),("1000",1),("99073",1),("10005",1),("10009",1),("10013",1),("99067",1),("1003",1),("99047",1),("311",1),("99061",1),("8010",1),("1007",1),("10002",1),("8005",1),("99032",1),("99018",1),("99028",1),("99044",1),("2001",1),("110000",1),("10001",1)]

# real per-section share of a query's tag refs (from "sections referenced").
SECTION_REF_WEIGHTS = {"3000":2708023,"3001":2698400,"99002":860316,"10015":290402,"99046":282884,"99047":273403,"99051":232666,"99052":223108,"99036":126848,"305":88326,"307":69628,"99011":42013,"10000":41343,"99017":40720,"2002":36867,"99005":36585,"2000":32696,"99012":31125,"99018":30536,"10004":30455,"99073":28295,"2003":27353,"2001":21808,"10011":21776,"304":21776,"1006":21776,"99000":21776,"99043":20609,"99070":18419,"99001":10888,"10012":10888,"99041":10888,"99033":3029,"10003":654,"99068":486,"99072":146,"99055":88,"99009":52}

# tags referenced in EVERY query (count == #queries in the real data).
UNIVERSAL_TAGS = ["99073#delivery_all","10011#140737489362418","10011#delivery_all","2000#delivery_all","304#0","304#delivery_all","307#delivery_all","1006#delivery_all","99000#delivery_all","99002#delivery_all","10012#140737489362418","99011#delivery_all","10000#delivery_all"]

# sections whose value lists are mirrored (referenced with identical values).
PAIRS = {"3000": "3001", "99046": "99047", "99051": "99052"}

# small "anchor" sections whose value-0 tag is forced ALWAYS-ON in the dataset and
# referenced by EVERY query (resolves) — reproduces the real freq~1.0 spike among
# referenced tags (so filters aren't all-rare -> match fraction > 0, ~15% common).
ANCHOR_SECTIONS = ["304", "305", "307", "99017", "99018", "99000", "99001",
                   "1006", "2001", "2003", "10004", "10011", "99012", "99043", "99005"]

TAG_NUM = sum(sz for _, sz in SECTIONS)  # 35672
DIM = 64
B_ALWAYS, B_UNIV, B_COMMON, B_LOW, B_RARE = 79, 28, 894, 2080, 15601
B_ABSENT = TAG_NUM - (B_ALWAYS + B_UNIV + B_COMMON + B_LOW + B_RARE)


# ---- tag_map + per-tag frequency (unchanged model from v1) ------------------
def build_tag_map(sections):
    id2term, sec_span, idx = [], {}, 0
    for name, size in sections:
        sec_span[name] = (idx, size)
        for v in range(size):
            id2term.append(f"{name}#{v}")
        idx += size
    return id2term, sec_span


def build_probs(rng, sec_span):
    # Tag frequencies must line up with how queries reference sections (see
    # SELECTIVE_SECTIONS / CARRIER_SECTIONS). Getting this wrong is what made the
    # first v2 match nothing (present tags landed in unreferenced sections).
    p = np.zeros(TAG_NUM, dtype=np.float64)

    # SELECTIVE sections: values are RARE (freq F_SEL); a small fraction are made
    # 'common' so the OR-union's tail reaches the real ~19% max match fraction.
    for s in SELECTIVE_SECTIONS:
        start, size = sec_span.get(s, (None, 0))
        if start is None:
            continue
        seg = np.full(size, F_SEL, dtype=np.float64)
        n_com = int(size * SEL_COMMON_FRAC)
        if n_com > 0:
            idx = rng.choice(size, n_com, replace=False)
            # cap at 0.12: a query occasionally ORs several common values, and a
            # higher cap lets them saturate (>=90% match), which real never does.
            seg[idx] = np.clip(rng.exponential(SEL_COMMON_FREQ, n_com), 0.01, 0.12)
        p[start:start + size] = seg

    # CARRIER sections: value 0 is always-on (freq 1.0) -> permissive filter clause.
    for s in CARRIER_SECTIONS:
        start, size = sec_span.get(s, (None, 0))
        if start is not None and size > 0:
            p[start] = 1.0

    # Density pool: the always/univ/common/low/rare buckets (v1 model) are placed
    # on NON-selective tags (carrier values>0 + unreferenced sections). They carry
    # the dataset density (~1%) WITHOUT joining the selective OR-union, so they
    # don't inflate the match fraction. Unreferenced/leftover tags stay absent.
    nonsel = []
    sel_set = set(SELECTIVE_SECTIONS)
    for name, size in SECTIONS:
        if name in sel_set:
            continue
        start, _ = sec_span[name]
        nonsel.extend(range(start, start + size))
    nonsel = np.array([gi for gi in nonsel if p[gi] == 0.0], dtype=np.int64)
    rng.shuffle(nonsel)
    buckets = [
        (B_ALWAYS, lambda n: np.ones(n)),
        (B_UNIV, lambda n: rng.uniform(0.90, 0.999, n)),
        (B_COMMON, lambda n: np.clip(rng.normal(0.24, 0.16, n), 0.05, 0.90)),
        (B_LOW, lambda n: rng.uniform(0.01, 0.05, n)),
        (B_RARE, lambda n: np.clip(rng.exponential(0.0016, n), 5e-5, 0.0099)),
    ]
    i = 0
    for cnt, gen in buckets:
        cnt = min(cnt, max(0, len(nonsel) - i))
        if cnt > 0:
            p[nonsel[i:i + cnt]] = gen(cnt)
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
    log(f"  generating dataset with {jobs} processes ({(total) / 1e9:.1f} GB)...")
    with mp.Pool(jobs) as pool:
        done = 0
        for got in pool.imap_unordered(_gen_range_worker, args):
            done += got
            log(f"  dataset: {done}/{doc_num} docs done")


# ---- section roles (from real query-analysis.md) ---------------------------
# The real filters match ~5% of docs (never empty, never >=90%). That selectivity
# comes from ORing ~200 RARE values of the heavily-referenced SELECTIVE sections
# (mostly 3000/3001) into one union; CARRIER sections contribute an always-on
# value (real: 304#0, 305#1 at freq 1) as PERMISSIVE clauses that don't restrict
# the top-level AND. Splitting the selective values across AND-joined nodes (the
# old bug) destroyed the union -> 93% empty; keeping them OR'd reproduces ~5%.
SELECTIVE_SECTIONS = ["3000", "3001", "99002", "99046", "99047", "99051", "99052", "10015", "99036"]
CARRIER_SECTIONS = ["304", "305", "307", "99017", "99018", "99000", "99001", "1006", "2000", "10000",
                    "10004", "99012", "2001", "2003", "99043", "10011", "10012", "99073", "99005", "99011"]
# calibrated (sim vs query-analysis.md): p50~0.048 mean~0.053 p90~0.087 p99~0.153
# max~0.22, 0% empty, 0% >=90%, density ~1.2%.
F_SEL = 0.00018          # base per-value freq of a SELECTIVE tag (rare)
SEL_COMMON_FRAC = 0.004  # fraction of SELECTIVE values made 'common' -> fat match tail
SEL_COMMON_FREQ = 0.028  # mean freq of those common values


def _sel_value(rng, size):
    """A referenced value: ~40% an in-range index (resolves to a rare SELECTIVE
    tag), else absent (huge id / 'delivery_all') -> skipped at conversion,
    reproducing the real ~40% tag resolution. Absent values still count as refs."""
    if rng.random() < 0.40:
        return str(int(rng.random() ** 2 * size))          # in [0,size): resolves, biased low
    if rng.random() < 0.6:
        return str(int(rng.integers(10 ** 12, 10 ** 16)))   # huge id: absent (skipped)
    return "delivery_all"                                   # catch value: absent (skipped)


def build_filter_v2(rng, sec_span):
    # Selectivity = ONE OR-union of ~200 rare SELECTIVE values (mostly 3000/3001).
    # Top-level AND also holds permissive CARRIER clauses (P~1) + one mild NOT.
    # depth 2; node mix ~ terms 7-8 / or 2-3 / and 1 / not 1 (like the real data).
    total = int(np.clip(rng.normal(780, 290), 67, 2046))
    sel_frac = rng.uniform(0.5, 0.9)
    selw = np.array([SECTION_REF_WEIGHTS.get(s, 1.0) for s in SELECTIVE_SECTIONS], dtype=np.float64)
    selw /= selw.sum()

    bysec: dict = {}
    for si in rng.choice(len(SELECTIVE_SECTIONS), size=int(total * sel_frac), p=selw):
        s = SELECTIVE_SECTIONS[si]
        _, size = sec_span.get(s, (0, 1))
        v = _sel_value(rng, size)
        bysec.setdefault(s, []).append(v)
        if s in PAIRS:                              # mirror value onto the paired section
            bysec.setdefault(PAIRS[s], []).append(v)

    # selective OR-cluster: split sections into <=3 OR-joined terms nodes, all
    # wrapped in ONE {"or":...} so their matches UNION (never AND-split -> empty).
    items = list(bysec.items())
    rng.shuffle(items)
    n_nodes = min(3, len(items)) or 1

    def or_terms(chunk):
        node = {"join_type": "or", "inner_section_join_type": "or"}
        for sec, vals in chunk:
            node[sec] = list(dict.fromkeys(vals))
        return {"terms": node}

    sel_nodes = [or_terms(items[gi::n_nodes]) for gi in range(n_nodes)]

    # permissive carrier clauses: value "0" resolves to an always-on tag -> P~1.
    perm = [{"terms": {"join_type": "or", "inner_section_join_type": "or", s: ["0"]}}
            for s in rng.choice(CARRIER_SECTIONS, size=min(4, len(CARRIER_SECTIONS)), replace=False)]

    # mild NOT: a handful of rare SELECTIVE values -> excludes ~1% of docs.
    ns = str(rng.choice(SELECTIVE_SECTIONS))
    _, nsz = sec_span.get(ns, (0, 1))
    nv = [x for x in (_sel_value(rng, nsz) for _ in range(20)) if x is not None]
    not_node = {"terms": {"join_type": "or", "inner_section_join_type": "or", ns: nv or ["0"]}}

    return {"and": [{"or": sel_nodes}] + perm + [{"not": not_node}]}


def write_queries_v2(path, q, sec_span, rng, log):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(q):
            vec = (rng.standard_normal(DIM) * 0.525).round(6).tolist()   # L2 ~4.2
            inner = {"vector": {
                "relevance_learning2rank": vec,
                "record_name": "relevance_learning2rank",
                "cluster_num": 64, "max_cluster_num": 128, "result_num": 3500,
                "max_distance_score_doc_num": 3500, "parallel_score": True,
                "syntax_filter": build_filter_v2(rng, sec_span),
            }}
            rec = {"corpus": "synthetic-v2", "result_num": 3500,
                   "main_tier": {"json_query": json.dumps(inner, separators=(",", ":")),
                                 "retrieve_num": 3500},
                   "request_id": f"synv2-{i}"}
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            if (i + 1) % 2000 == 0:
                log(f"  queries {i + 1}/{q}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--docs", type=int, default=10_000_000)
    ap.add_argument("--queries", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=20260707)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=16, help="parallel processes for dataset.bin (1 = serial)")
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
    log(f"tag_num={TAG_NUM} sections={len(SECTIONS)} E[tags/doc]={float(p.sum()):.1f} "
        f"density={100 * float(p.sum()) / TAG_NUM:.3f}%")

    log(f"writing {out/'tag_map.bin'}")
    write_tag_map(out / "tag_map.bin", id2term)
    log(f"writing {out/'QueryData.txt'} ({a.queries} queries)")
    write_queries_v2(out / "QueryData.txt", a.queries, sec_span, rng, log)

    if not a.skip_dataset:
        stride = (TAG_NUM + 63) // 64
        gb = (40 + a.docs * (DIM * 4 + stride * 8)) / 1e9
        log(f"writing {out/'dataset.bin'} ({a.docs} docs, ~{gb:.1f} GB, jobs={a.jobs})")
        write_dataset(out / "dataset.bin", a.docs, p, a.seed, a.chunk, a.jobs, log)

    log("done. Verify: python3 tools/analyze_query.py "
        f"--query {out/'QueryData.txt'} --tag-map {out/'tag_map.bin'} "
        f"--dataset {out/'dataset.bin'} --top 50")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
