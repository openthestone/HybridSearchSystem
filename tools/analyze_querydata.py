#!/usr/bin/env python3
"""
Analyze the ORIGINAL Huawei project data (QueryData_*.txt + tag_map.bin +
dataset.bin/HYDSET2) and print a compact distribution report as plain text.

Formats (mirrors include/utils/DataReader.h + convert_querydata.py):
  QueryData_*.txt : JSON lines (or a JSON array). Each record:
      { "result_num": <int top_k>,
        "main_tier": { "json_query": "<escaped inner json>" } }
    inner json: { "vector": {"relevance_learning2rank": [floats...],
                             "syntax_filter": <boolean tree>} }   # filter INSIDE vector
    syntax_filter nodes: term{section:val} | terms{join_type,section:[vals]}
                         | and[..] | or[..] | not{..}   (tag = "section#val")
  tag_map.bin     : [size_t map_size][ repeat: [u32 len][utf8 term][i32 tag_id] ]
  dataset.bin     : HYDSET2, 40B header then FP32 vectors + u64 bitmaps.

The report is bounded well under 50 KB (histograms + top-N lists only), so it
can be pasted back verbatim. Nothing here mutates the inputs.

Usage:
  python3 analyze_querydata.py [--query QueryData_10000.txt]
                               [--tag-map tag_map.bin]
                               [--dataset dataset.bin]
                               [--top 30] [--out report.txt]
"""
from __future__ import annotations

import argparse
import json
import math
import mmap
import random
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---- output buffer (hard-capped so the report stays < 50 KB) ----------------
_LINES: List[str] = []
_MAX_BYTES = 48_000


def emit(line: str = "") -> None:
    _LINES.append(line)


def _render() -> str:
    out = "\n".join(_LINES) + "\n"
    if len(out.encode("utf-8", "replace")) > _MAX_BYTES:
        out = out.encode("utf-8", "replace")[:_MAX_BYTES].decode("utf-8", "ignore")
        out += "\n... [truncated to stay under 50 KB] ...\n"
    return out


# ---- small stats helpers ----------------------------------------------------
def stat_line(name: str, xs: List[float]) -> str:
    if not xs:
        return f"  {name}: (none)"
    xs2 = sorted(xs)
    n = len(xs2)

    def pct(p: float) -> float:
        return xs2[min(n - 1, int(p * n))]

    mean = sum(xs2) / n
    return (f"  {name}: n={n} min={xs2[0]:.4g} p50={pct(0.50):.4g} "
            f"mean={mean:.4g} p90={pct(0.90):.4g} p99={pct(0.99):.4g} max={xs2[-1]:.4g}")


def histogram(name: str, xs: List[float], nbuckets: int = 12) -> None:
    emit(stat_line(name, xs))
    if not xs:
        return
    lo, hi = min(xs), max(xs)
    if hi <= lo:
        emit(f"    all == {lo:g}  (count {len(xs)})")
        return
    width = (hi - lo) / nbuckets
    counts = [0] * nbuckets
    for v in xs:
        b = min(nbuckets - 1, int((v - lo) / width))
        counts[b] += 1
    peak = max(counts) or 1
    for i, c in enumerate(counts):
        a = lo + i * width
        b = a + width
        bar = "#" * int(round(40 * c / peak))
        emit(f"    [{a:>10.4g},{b:>10.4g})  {c:>8d} {bar}")


def top_counter(name: str, ctr: Counter, top: int) -> None:
    emit(f"  {name}: {len(ctr)} distinct")
    for key, cnt in ctr.most_common(top):
        emit(f"    {cnt:>10d}  {str(key)[:80]}")


# ---- tag_map.bin ------------------------------------------------------------
def load_tag_map(path: Path) -> Dict[str, int]:
    data = path.read_bytes()
    for size_t in (8, 4):
        try:
            m, end = _parse_tag_map(data, size_t)
            if end == len(data):
                return m
        except (struct.error, UnicodeDecodeError, ValueError):
            continue
    raise ValueError(f"failed to parse tag_map: {path}")


def _parse_tag_map(data: bytes, size_t: int) -> Tuple[Dict[str, int], int]:
    off = 0
    n = struct.unpack_from("<Q" if size_t == 8 else "<I", data, off)[0]
    off += size_t
    if n <= 0 or n > 50_000_000:
        raise ValueError("bad map size")
    out: Dict[str, int] = {}
    for _ in range(n):
        (length,) = struct.unpack_from("<I", data, off)
        off += 4
        if length <= 0 or off + length + 4 > len(data):
            raise ValueError("truncated entry")
        term = data[off:off + length].decode("utf-8")
        off += length
        (tag_id,) = struct.unpack_from("<i", data, off)
        off += 4
        out[term] = tag_id
    return out, off


def report_tag_map(tag_map: Dict[str, int], top: int) -> None:
    emit(f"\n### TAG_MAP  ({len(tag_map)} terms)")
    ids = list(tag_map.values())
    dup = len(ids) - len(set(ids))
    emit(f"  tag_id range: {min(ids)} .. {max(ids)}   distinct ids: {len(set(ids))}"
         + (f"   !! {dup} duplicate ids" if dup else ""))
    sec = Counter(t.split("#", 1)[0] for t in tag_map)
    malformed = sum(1 for t in tag_map if "#" not in t)
    if malformed:
        emit(f"  {malformed} terms without a '#'-separated section")
    top_counter("sections (prefix before '#'), by #tags", sec, top)


# ---- dataset.bin (HYDSET2): header + sampled density/tag-freq/vector ---------
# Layout (mirrors inspect_hw.py): [40B header][doc_num * dim FP32 vectors]
# [doc_num * stride u64 bitmaps], stride=ceil(tag_num/64). Tag t lives in
# word t//64, bit t%64 (LSB-first). Vectors and bitmaps are SEPARATE regions.
def report_dataset(path: Path, top: int, sample: int, seed: int) -> Optional[Dict[str, int]]:
    fsize = path.stat().st_size
    with path.open("rb") as f:
        head = f.read(40)
    if len(head) < 40 or head[:7] != b"HYDSET2":
        emit(f"  {path}: not HYDSET2 (magic={head[:7]!r})")
        return None
    version, doc_num, dim, tag_num = struct.unpack_from("<I4xQII", head, 8)
    stride = (tag_num + 63) // 64
    vec_off = 40
    bmp_off = 40 + doc_num * dim * 4
    expect = bmp_off + doc_num * stride * 8

    emit(f"\n### DATASET (HYDSET2)  {path}")
    emit(f"  version={version} doc_num={doc_num} vector_dim={dim} tag_num={tag_num}")
    emit(f"  stride(u64)={stride}  bytes/doc={dim * 4 + stride * 8}  "
         f"file={fsize / 1e9:.2f}GB expected={expect / 1e9:.2f}GB "
         f"match={'OK' if fsize == expect else 'MISMATCH'}")
    if fsize != expect:
        emit("  !! size mismatch — layout assumptions may be wrong; stats below suspect")

    N = doc_num
    S = min(sample, N)
    rnd = random.Random(seed)
    idxs = sorted(rnd.sample(range(N), S)) if S < N else list(range(N))
    try:
        import numpy as np  # noqa: N813
        have_np = True
    except ImportError:
        have_np = False
    emit(f"  sampling {S} of {N} docs (seed={seed}, numpy={'yes' if have_np else 'no'})")

    tag_counts: List[int] = []
    vnorms: List[float] = []
    vmin, vmax = math.inf, -math.inf
    csum = csq = 0.0   # component sum / sum-of-squares for mean/std
    ccnt = 0
    tag_freq: Dict[int, int] = {}

    with path.open("rb") as f:
        mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        try:
            if have_np:
                hits = np.zeros(tag_num, dtype=np.int64)
                for d in idxs:
                    vo = vec_off + d * dim * 4
                    # slice mm into a bytes copy first so no numpy view keeps the
                    # mmap "exported" (which would make mm.close() raise).
                    vec = np.frombuffer(mm[vo:vo + dim * 4], dtype="<f4")
                    ss = float(vec @ vec)
                    vnorms.append(math.sqrt(ss))
                    csum += float(vec.sum())
                    csq += ss
                    ccnt += dim
                    vmin = min(vmin, float(vec.min()))
                    vmax = max(vmax, float(vec.max()))
                    bo = bmp_off + d * stride * 8
                    raw = np.frombuffer(mm[bo:bo + stride * 8], dtype=np.uint8)
                    bits = np.unpackbits(raw, bitorder="little")[:tag_num]
                    tag_counts.append(int(bits.sum()))
                    hits += bits
                tag_freq = {t: int(c) for t, c in enumerate(hits.tolist()) if c}
            else:
                for d in idxs:
                    vb = struct.unpack_from(f"<{dim}f", mm, vec_off + d * dim * 4)
                    ss = sum(x * x for x in vb)
                    vnorms.append(math.sqrt(ss))
                    csum += sum(vb)
                    csq += ss
                    ccnt += dim
                    vmin = min(vmin, min(vb))
                    vmax = max(vmax, max(vb))
                    words = struct.unpack_from(f"<{stride}Q", mm, bmp_off + d * stride * 8)
                    cnt = 0
                    for wi, w in enumerate(words):
                        while w:
                            b = (w & -w).bit_length() - 1
                            tag = wi * 64 + b
                            if tag < tag_num:
                                cnt += 1
                                tag_freq[tag] = tag_freq.get(tag, 0) + 1
                            w &= w - 1
                    tag_counts.append(cnt)
        finally:
            mm.close()

    n = len(tag_counts)
    avg = sum(tag_counts) / n if n else 0.0
    emit("")
    emit("-- per-doc tag count (how many tags each doc carries) --")
    histogram("tags/doc", [float(c) for c in tag_counts])
    emit(f"  density = avg/tag_num = {100 * avg / tag_num:.3f}%   "
         f"(avg {avg:.1f} of {tag_num} tags set per doc)")

    emit("")
    emit("-- tag-frequency structure (fraction of sampled docs a tag appears in) --")
    seen = len(tag_freq)

    def fr(c: int) -> float:
        return c / n if n else 0.0

    always = sum(1 for c in tag_freq.values() if fr(c) >= 0.999)
    univ = sum(1 for c in tag_freq.values() if 0.90 <= fr(c) < 0.999)
    common = sum(1 for c in tag_freq.values() if 0.05 <= fr(c) < 0.90)
    low = sum(1 for c in tag_freq.values() if 0.01 <= fr(c) < 0.05)
    rare = sum(1 for c in tag_freq.values() if fr(c) < 0.01)
    emit(f"  distinct tags seen in sample: {seen} of {tag_num}  "
         f"(never seen: {tag_num - seen} => absent/very-rare)")
    emit(f"  always-on (>=99.9%) : {always}")
    emit(f"  universal (90-99.9%): {univ}")
    emit(f"  common    (5-90%)   : {common}   <- dense bitset per segment; dominate filter cost")
    emit(f"  low       (1-5%)    : {low}")
    emit(f"  rare      (<1%)     : {rare}   (+ {tag_num - seen} never-seen)")
    emit(f"  overall bitmap fill = set_bits/(docs*tags) = {100 * avg / tag_num:.3f}%")

    total_bits = sum(tag_counts)
    fs_desc = sorted(tag_freq.values(), reverse=True)

    def cum(k: int) -> float:
        return 100 * sum(fs_desc[:k]) / total_bits if total_bits else 0.0

    emit(f"  set-bit concentration: top-20 tags carry {cum(20):.1f}% of all set bits, "
         f"top-100={cum(100):.1f}%, top-1000={cum(1000):.1f}%")

    by_freq = sorted(tag_freq.items(), key=lambda kv: kv[1])  # ascending
    emit(f"  top-{top} most-common tags (tag_id: fraction of docs, est. doc count):")
    for t, c in reversed(by_freq[-top:]):
        emit(f"    tag {t:>6d}: {100 * fr(c):6.2f}%  (~{int(fr(c) * doc_num):,} docs)")

    emit(f"  rarest-{top} SEEN tags (lowest non-zero frequency in the sample):")
    for t, c in by_freq[:top]:
        emit(f"    tag {t:>6d}: {100 * fr(c):7.4f}%  ({c}/{n} sampled docs)")
    emit(f"  (+ {tag_num - seen} tags NEVER seen in the sample -> effectively absent)")

    # index storage implication (why full-recall postings are huge / HBM-bound):
    # a tag with global freq f has per-segment density ~f; when f >= threshold it
    # is stored as a dense bitset (seg/8 bytes) and, being frequent, appears in
    # essentially every segment. So common tags dominate posting storage.
    emit("")
    emit("-- index storage implication (doc_num_per_segment=65536, density_threshold=0.05) --")
    seg, thr = 65536, 0.05
    nseg = math.ceil(doc_num / seg)
    bitset_tags = sum(1 for c in tag_freq.values() if fr(c) >= thr)
    bitset_bytes = bitset_tags * nseg * (seg // 8)
    emit(f"  segments={nseg}  bitset-tags(freq>=5%)={bitset_tags}  bitlist/absent-tags={tag_num - bitset_tags}")
    emit(f"  ~bitset posting storage = bitset_tags * segments * (seg/8) = {bitset_bytes / 1e9:.1f} GB "
         f"(dense common tags, present in ~all segments -> dominant term)")

    emit("")
    emit("-- document vectors --")
    if ccnt:
        cmean = csum / ccnt
        cstd = math.sqrt(max(0.0, csq / ccnt - cmean * cmean))
        mean_norm = sum(vnorms) / len(vnorms)
        emit(f"  dim={dim}  component mean={cmean:.4g} std={cstd:.4g}  "
             f"mean L2 norm={mean_norm:.4g}  (normalized={'~yes' if abs(mean_norm - 1) < 0.1 else 'no'})")
    histogram("L2 norm", vnorms)
    if vnorms:
        emit(f"  component value range (sampled): [{vmin:.4g}, {vmax:.4g}]")

    frac = {t: c / n for t, c in tag_freq.items()} if n else {}
    return {"doc_num": doc_num, "dim": dim, "tag_num": tag_num, "frac": frac}


# ---- syntax_filter tree walk ------------------------------------------------
class FilterStats:
    def __init__(self) -> None:
        self.node_types: Counter = Counter()
        self.sections: Counter = Counter()       # section -> #tag references
        self.tags: Counter = Counter()           # "section#value" -> refs
        self.join_types: Counter = Counter()


def walk_filter(node: Any, st: FilterStats, depth: int) -> Tuple[int, int]:
    """Return (max_depth, tag_ref_count) for this subtree; updates st in place."""
    if not isinstance(node, dict) or len(node) != 1:
        return depth, 0
    key = next(iter(node))
    st.node_types[key] += 1
    if key == "term":
        tn = node["term"]
        refs = 0
        if isinstance(tn, dict):
            for sec, val in tn.items():
                st.sections[sec] += 1
                st.tags[f"{sec}#{_vstr(val)}"] += 1
                refs += 1
        return depth, refs
    if key == "terms":
        tn = node["terms"]
        refs = 0
        if isinstance(tn, dict):
            st.join_types[str(tn.get("join_type", "or")).lower()] += 1
            for sec, vals in tn.items():
                if sec in ("join_type", "inner_section_join_type"):
                    continue
                vals = vals if isinstance(vals, list) else [vals]
                for v in vals:
                    st.sections[sec] += 1
                    st.tags[f"{sec}#{_vstr(v)}"] += 1
                    refs += 1
        return depth, refs
    if key in ("and", "or"):
        st.join_types[key] += 1
        md, refs = depth, 0
        for ch in (node[key] if isinstance(node[key], list) else []):
            d, r = walk_filter(ch, st, depth + 1)
            md = max(md, d)
            refs += r
        return md, refs
    if key == "not":
        d, r = walk_filter(node["not"], st, depth + 1)
        return d, r
    return depth, 0


def _vstr(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return format(v, ".6g")
    return str(v)


def _combine_prob(ps: List[float], op: str) -> float:
    if not ps:
        return 1.0 if op == "and" else 0.0
    if op == "and":
        r = 1.0
        for p in ps:
            r *= p
        return r
    r = 1.0  # or: 1 - prod(1-p)
    for p in ps:
        r *= (1.0 - p)
    return 1.0 - r


def eval_prob(node: Any, tag_map: Dict[str, int], frac: Dict[int, float]) -> float:
    """Estimated P(a doc satisfies this filter), assuming tag independence.
    Uses dataset tag frequencies; a tag absent from tag_map/dataset -> P=0."""
    if not isinstance(node, dict) or len(node) != 1:
        return 1.0
    key = next(iter(node))

    def p_of(sec: str, val: Any) -> float:
        tid = tag_map.get(f"{sec}#{_vstr(val)}")
        return frac.get(tid, 0.0) if tid is not None else 0.0

    if key == "term":
        tn = node["term"]
        ps = [p_of(s, v) for s, v in tn.items()] if isinstance(tn, dict) else []
        return _combine_prob(ps, "and")
    if key == "terms":
        tn = node["terms"]
        if not isinstance(tn, dict):
            return 1.0
        ij = str(tn.get("inner_section_join_type", "or")).lower()
        jt = str(tn.get("join_type", "or")).lower()
        sec_ps = []
        for sec, vals in tn.items():
            if sec in ("join_type", "inner_section_join_type"):
                continue
            vals = vals if isinstance(vals, list) else [vals]
            sec_ps.append(_combine_prob([p_of(sec, v) for v in vals], ij))
        return _combine_prob(sec_ps, jt)
    if key in ("and", "or"):
        ch = node[key] if isinstance(node[key], list) else []
        return _combine_prob([eval_prob(c, tag_map, frac) for c in ch], key)
    if key == "not":
        return 1.0 - eval_prob(node["not"], tag_map, frac)
    return 1.0


# ---- QueryData_*.txt --------------------------------------------------------
def load_records(path: Path) -> List[Dict[str, Any]]:
    txt = path.read_text(encoding="utf-8", errors="replace").strip()
    if not txt:
        return []
    try:
        parsed = json.loads(txt)
        if isinstance(parsed, list):
            return [x for x in parsed if isinstance(x, dict)]
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass
    recs = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
            if isinstance(item, dict):
                recs.append(item)
        except json.JSONDecodeError:
            pass
    return recs


def report_query(path: Path, tag_map: Optional[Dict[str, int]], top: int,
                 frac: Optional[Dict[int, float]] = None) -> None:
    recs = load_records(path)
    emit(f"\n### QUERYDATA  {path}   ({len(recs)} records)")
    if not recs:
        emit("  (no records parsed)")
        return
    sel: List[float] = []          # estimated per-query match fraction
    ref_fracs: List[float] = []    # dataset frequency of each referenced tag

    result_nums: List[float] = []
    dims: Counter = Counter()
    vnorms: List[float] = []
    vminmax: List[float] = []
    have_filter = 0
    const_filter = 0
    depths: List[float] = []
    refs_per_q: List[float] = []
    distinct_tags_per_q: List[float] = []
    fs = FilterStats()
    resolved = missing = 0
    bad = 0

    for r in recs:
        rn = r.get("result_num")
        if isinstance(rn, int) and not isinstance(rn, bool) and rn > 0:
            result_nums.append(rn)
        mt = r.get("main_tier")
        jq = mt.get("json_query") if isinstance(mt, dict) else None
        if not isinstance(jq, str):
            bad += 1
            continue
        try:
            inner = json.loads(jq)
        except json.JSONDecodeError:
            bad += 1
            continue
        vnode = inner.get("vector")
        if isinstance(vnode, dict):
            vec = vnode.get("relevance_learning2rank")
            if isinstance(vec, list) and vec:
                dims[len(vec)] += 1
                fv = [float(x) for x in vec if isinstance(x, (int, float))]
                if fv:
                    vnorms.append(math.sqrt(sum(x * x for x in fv)))
                    vminmax.append(min(fv))
                    vminmax.append(max(fv))
        # syntax_filter lives INSIDE the vector node (mirrors convert_querydata.py /
        # DataReader.h: `if syntax_filter_key in vector_node`), not at inner level.
        sf = vnode.get("syntax_filter") if isinstance(vnode, dict) else None
        if sf is None:  # tolerate the alternate placement just in case
            sf = inner.get("syntax_filter")
        if sf is None:
            const_filter += 1
        else:
            have_filter += 1
            st_q = FilterStats()
            d, refs = walk_filter(sf, st_q, 0)
            depths.append(d)
            refs_per_q.append(refs)
            distinct_tags_per_q.append(len(st_q.tags))
            fs.node_types.update(st_q.node_types)
            fs.sections.update(st_q.sections)
            fs.tags.update(st_q.tags)
            fs.join_types.update(st_q.join_types)
            if tag_map is not None:
                for tg, c in st_q.tags.items():
                    if tg in tag_map:
                        resolved += c
                    else:
                        missing += c
                    if frac is not None:
                        tid = tag_map.get(tg)
                        if tid is not None and tid in frac:
                            ref_fracs.extend([frac[tid]] * c)
                if frac is not None:
                    sel.append(eval_prob(sf, tag_map, frac))

    emit(f"  parsed OK: {len(recs) - bad}   unparseable json_query: {bad}")
    emit(f"  with syntax_filter: {have_filter}   match-all (no filter): {const_filter}")
    emit("")
    emit("-- result_num (top_k) --")
    histogram("result_num", result_nums)
    emit("")
    emit("-- query vector --")
    emit(f"  dim(s): {dict(dims)}")
    histogram("L2 norm", vnorms)
    if vminmax:
        emit(f"  component value range: [{min(vminmax):.4g}, {max(vminmax):.4g}]")
    emit("")
    emit("-- filter tree shape (over queries that have a filter) --")
    histogram("tree depth", depths, 10)
    histogram("tag refs / query", refs_per_q)
    histogram("distinct tags / query", distinct_tags_per_q)
    emit("  node types: " + ", ".join(f"{k}={v}" for k, v in fs.node_types.most_common()))
    emit("  join types: " + ", ".join(f"{k}={v}" for k, v in fs.join_types.most_common()))
    emit("")
    top_counter("sections referenced (by #tag refs)", fs.sections, top)
    emit("")
    top_counter("most-referenced tags (section#value)", fs.tags, top)
    if tag_map is not None:
        tot = resolved + missing
        pct = (100.0 * resolved / tot) if tot else 0.0
        emit("")
        emit(f"-- tag resolution vs tag_map: {resolved}/{tot} resolved ({pct:.2f}%), "
             f"{missing} missing --")

    if ref_fracs:
        emit("")
        emit("-- commonness of referenced tags (their dataset doc-frequency) --")
        histogram("referenced-tag freq", ref_fracs)
        cr = sum(1 for f in ref_fracs if f >= 0.05)
        emit(f"  referenced tags that are 'common' (freq>=5% => stored bitset): "
             f"{cr}/{len(ref_fracs)} ({100 * cr / len(ref_fracs):.1f}%) "
             f"-- these set the filter's per-query cost")

    if sel:
        emit("")
        emit("-- estimated filter selectivity (P a doc matches; tag-independence) --")
        histogram("match fraction", sel, 10)
        near1 = sum(1 for p in sel if p >= 0.9)
        emit(f"  filters matching >=90% of docs: {near1}/{len(sel)} "
             f"-- a permissive filter barely prunes yet the engine still scans every "
             f"referenced posting (full-recall cost is paid regardless of selectivity)")


# ---- main -------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", default="QueryData_10000.txt")
    ap.add_argument("--tag-map", default="tag_map.bin")
    ap.add_argument("--dataset", default="dataset.bin")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--dataset-sample", type=int, default=20000,
                    help="docs to sample for dataset density/tag-freq (0 = all; full scan reads the whole file)")
    ap.add_argument("--seed", type=int, default=1234, help="sampling seed")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    emit("HUAWEI ORIGINAL-DATA DISTRIBUTION REPORT")
    emit("")

    tag_map: Optional[Dict[str, int]] = None
    tmp = Path(a.tag_map)
    if tmp.is_file():
        try:
            tag_map = load_tag_map(tmp)
            report_tag_map(tag_map, a.top)
        except Exception as e:  # noqa: BLE001
            emit(f"[tag_map] error: {e}")
    else:
        emit(f"[tag_map] not found: {tmp}")
    emit("")

    frac: Optional[Dict[int, float]] = None
    dsp = Path(a.dataset)
    if dsp.is_file():
        try:
            samp = a.dataset_sample if a.dataset_sample > 0 else (1 << 62)
            info = report_dataset(dsp, a.top, samp, a.seed)
            if info:
                frac = info.get("frac")
        except Exception as e:  # noqa: BLE001
            emit(f"[dataset] error: {e}")
    else:
        emit(f"[dataset] not found: {dsp}")

    qp = Path(a.query)
    if qp.is_file():
        try:
            report_query(qp, tag_map, a.top, frac)
        except Exception as e:  # noqa: BLE001
            emit(f"[querydata] error: {e}")
    else:
        emit(f"[querydata] not found: {qp}")

    report = _render()
    if a.out:
        Path(a.out).write_text(report, encoding="utf-8")
        sys.stderr.write(f"wrote {len(report.encode('utf-8'))} bytes to {a.out}\n")
    else:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
