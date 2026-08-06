#!/usr/bin/env python3
"""
Analyze ONLY the QueryData_*.txt distribution (query vectors, top_k, and the
syntax_filter tree) and print a compact plain-text report. Extracted from
analyze_querydata.py; the tag_map / dataset distribution reports are dropped.

QueryData_*.txt : JSON lines (or a JSON array). Each record:
    { "result_num": <int top_k>,
      "main_tier": { "json_query": "<escaped inner json>" } }
  inner json: { "vector": {"relevance_learning2rank": [floats...],
                           "syntax_filter": <boolean tree>} }   # filter INSIDE vector
  syntax_filter nodes: term{section:val} | terms{join_type,section:[vals]}
                       | and[..] | or[..] | not{..}   (tag = "section#val")

--tag-map (optional) adds referenced-tag RESOLUTION; --dataset (optional, HYDSET2)
adds referenced-tag COMMONNESS + estimated filter SELECTIVITY. Output is bounded
< 50 KB (histograms + top-N). Read-only.

Usage:
  python3 analyze_query.py --query QueryData_10000.txt
                           [--tag-map tag_map.bin] [--dataset dataset.bin]
                           [--top 50] [--out report.txt]
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


def stat_line(name: str, xs: List[float]) -> str:
    if not xs:
        return f"  {name}: (none)"
    xs2 = sorted(xs)
    n = len(xs2)

    def pct(p: float) -> float:
        return xs2[min(n - 1, int(p * n))]

    return (f"  {name}: n={n} min={xs2[0]:.4g} p50={pct(0.50):.4g} "
            f"mean={sum(xs2) / n:.4g} p90={pct(0.90):.4g} p99={pct(0.99):.4g} max={xs2[-1]:.4g}")


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
        counts[min(nbuckets - 1, int((v - lo) / width))] += 1
    peak = max(counts) or 1
    for i, c in enumerate(counts):
        a = lo + i * width
        emit(f"    [{a:>10.4g},{a + width:>10.4g})  {c:>8d} {'#' * int(round(40 * c / peak))}")


def top_counter(name: str, ctr: Counter, top: int) -> None:
    emit(f"  {name}: {len(ctr)} distinct")
    for key, cnt in ctr.most_common(top):
        emit(f"    {cnt:>10d}  {str(key)[:80]}")


# ---- tag_map.bin (optional; for resolution) --------------------------------
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


# ---- dataset.bin (optional; per-tag frequency for selectivity/commonness) ---
def sample_tag_freq(path: Path, sample: int, seed: int) -> Optional[Dict[int, float]]:
    with path.open("rb") as f:
        head = f.read(40)
    if len(head) < 40 or head[:7] != b"HYDSET2":
        return None
    _, doc_num, dim, tag_num = struct.unpack_from("<I4xQII", head, 8)
    stride = (tag_num + 63) // 64
    bmp_off = 40 + doc_num * dim * 4
    S = min(sample, doc_num)
    idxs = sorted(random.Random(seed).sample(range(doc_num), S)) if S < doc_num else list(range(doc_num))
    try:
        import numpy as np  # noqa: N813
    except ImportError:
        np = None
    hits = [0] * tag_num
    with path.open("rb") as f:
        mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        try:
            if np is not None:
                acc = np.zeros(tag_num, dtype=np.int64)
                for d in idxs:
                    bo = bmp_off + d * stride * 8
                    bits = np.unpackbits(np.frombuffer(mm[bo:bo + stride * 8], dtype=np.uint8),
                                         bitorder="little")[:tag_num]
                    acc += bits
                hits = acc.tolist()
            else:
                for d in idxs:
                    words = struct.unpack_from(f"<{stride}Q", mm, bmp_off + d * stride * 8)
                    for wi, w in enumerate(words):
                        while w:
                            b = (w & -w).bit_length() - 1
                            t = wi * 64 + b
                            if t < tag_num:
                                hits[t] += 1
                            w &= w - 1
        finally:
            mm.close()
    return {t: c / S for t, c in enumerate(hits) if c}


# ---- syntax_filter tree ----------------------------------------------------
class FilterStats:
    def __init__(self) -> None:
        self.node_types: Counter = Counter()
        self.sections: Counter = Counter()
        self.tags: Counter = Counter()
        self.join_types: Counter = Counter()


def _vstr(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return format(v, ".6g")
    return str(v)


def _as_int(v: Any) -> Optional[int]:
    """Positive int (not bool) or None — for the top_k / result_num-like fields."""
    return v if isinstance(v, int) and not isinstance(v, bool) and v > 0 else None


def walk_filter(node: Any, st: FilterStats, depth: int) -> Tuple[int, int]:
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
                for v in (vals if isinstance(vals, list) else [vals]):
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
        return walk_filter(node["not"], st, depth + 1)
    return depth, 0


def _combine_prob(ps: List[float], op: str) -> float:
    if not ps:
        return 1.0 if op == "and" else 0.0
    if op == "and":
        r = 1.0
        for p in ps:
            r *= p
        return r
    r = 1.0
    for p in ps:
        r *= (1.0 - p)
    return 1.0 - r


def eval_prob(node: Any, tag_map: Dict[str, int], frac: Dict[int, float]) -> float:
    if not isinstance(node, dict) or len(node) != 1:
        return 1.0
    key = next(iter(node))

    def p_of(sec: str, val: Any) -> float:
        tid = tag_map.get(f"{sec}#{_vstr(val)}")
        return frac.get(tid, 0.0) if tid is not None else 0.0

    if key == "term":
        tn = node["term"]
        return _combine_prob([p_of(s, v) for s, v in tn.items()] if isinstance(tn, dict) else [], "and")
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


# ---- QueryData_*.txt -------------------------------------------------------
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


def report_query(path: Path, tag_map: Optional[Dict[str, int]], frac: Optional[Dict[int, float]],
                 top: int) -> None:
    recs = load_records(path)
    emit(f"### QUERYDATA  {path}   ({len(recs)} records)")
    if not recs:
        emit("  (no records parsed)")
        return

    result_nums: List[float] = []            # top-level result_num
    retrieve_nums: List[float] = []          # main_tier.retrieve_num
    vec_result_nums: List[float] = []        # vector.result_num
    vec_maxdist: List[float] = []            # vector.max_distance_score_doc_num
    topk_all_equal = topk_multi = 0          # cross-field agreement per record
    dims: Counter = Counter()
    vnorms: List[float] = []
    vminmax: List[float] = []
    have_filter = const_filter = bad = 0
    depths: List[float] = []
    refs_per_q: List[float] = []
    distinct_tags_per_q: List[float] = []
    fs = FilterStats()
    resolved = missing = 0
    sel: List[float] = []
    ref_fracs: List[float] = []

    for r in recs:
        rn = _as_int(r.get("result_num"))
        if rn is not None:
            result_nums.append(rn)
        mt = r.get("main_tier")
        rtn = _as_int(mt.get("retrieve_num")) if isinstance(mt, dict) else None
        if rtn is not None:
            retrieve_nums.append(rtn)
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
        vrn = vmd = None
        if isinstance(vnode, dict):
            vrn = _as_int(vnode.get("result_num"))
            vmd = _as_int(vnode.get("max_distance_score_doc_num"))
            if vrn is not None:
                vec_result_nums.append(vrn)
            if vmd is not None:
                vec_maxdist.append(vmd)
        present = [x for x in (rn, rtn, vrn, vmd) if x is not None]
        if len(present) >= 2:
            topk_multi += 1
            if len(set(present)) == 1:
                topk_all_equal += 1
        if isinstance(vnode, dict):
            vec = vnode.get("relevance_learning2rank")
            if isinstance(vec, list) and vec:
                dims[len(vec)] += 1
                fv = [float(x) for x in vec if isinstance(x, (int, float))]
                if fv:
                    vnorms.append(math.sqrt(sum(x * x for x in fv)))
                    vminmax.append(min(fv))
                    vminmax.append(max(fv))
        # filter lives inside the vector node (per convert_querydata.py / DataReader.h)
        sf = vnode.get("syntax_filter") if isinstance(vnode, dict) else None
        if sf is None:
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
    emit("-- top_k / result_num-like fields (which one the engine uses matters) --")
    histogram("result_num (top-level)", result_nums)
    histogram("main_tier.retrieve_num", retrieve_nums)
    histogram("vector.result_num", vec_result_nums)
    histogram("vector.max_distance_score_doc_num", vec_maxdist)
    if topk_multi:
        emit(f"  cross-field agreement: {topk_all_equal}/{topk_multi} records have all present "
             f"fields equal ({100 * topk_all_equal / topk_multi:.1f}%); "
             f"{topk_multi - topk_all_equal} differ")
    emit("  (our pipeline uses TOP-LEVEL result_num -> topk.txt -> --topk_file)")
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
        emit(f"-- tag resolution vs tag_map: {resolved}/{tot} resolved ({pct:.2f}%), {missing} missing --")
    if ref_fracs:
        emit("")
        emit("-- commonness of referenced tags (their dataset doc-frequency) --")
        histogram("referenced-tag freq", ref_fracs)
        cr = sum(1 for f in ref_fracs if f >= 0.05)
        emit(f"  referenced tags that are 'common' (freq>=5% => stored bitset): "
             f"{cr}/{len(ref_fracs)} ({100 * cr / len(ref_fracs):.1f}%)")
    if sel:
        emit("")
        emit("-- estimated filter selectivity (P a doc matches; tag-independence) --")
        histogram("match fraction", sel, 10)
        near1 = sum(1 for p in sel if p >= 0.9)
        emit(f"  filters matching >=90% of docs: {near1}/{len(sel)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", required=True)
    ap.add_argument("--tag-map", default="", help="tag_map.bin (optional): referenced-tag resolution")
    ap.add_argument("--dataset", default="", help="dataset.bin HYDSET2 (optional): selectivity/commonness")
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--dataset-sample", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    emit("QUERYDATA DISTRIBUTION REPORT")
    emit("")

    tag_map: Optional[Dict[str, int]] = None
    if a.tag_map:
        tmp = Path(a.tag_map)
        if tmp.is_file():
            try:
                tag_map = load_tag_map(tmp)
            except Exception as e:  # noqa: BLE001
                emit(f"[tag_map] error: {e}")
        else:
            emit(f"[tag_map] not found: {tmp}")

    frac: Optional[Dict[int, float]] = None
    if a.dataset and tag_map is not None:
        dsp = Path(a.dataset)
        if dsp.is_file():
            try:
                frac = sample_tag_freq(dsp, a.dataset_sample if a.dataset_sample > 0 else (1 << 62), a.seed)
            except Exception as e:  # noqa: BLE001
                emit(f"[dataset] error: {e}")
        else:
            emit(f"[dataset] not found: {dsp}")

    qp = Path(a.query)
    if not qp.is_file():
        emit(f"[querydata] not found: {qp}")
    else:
        try:
            report_query(qp, tag_map, frac, a.top)
        except Exception as e:  # noqa: BLE001
            emit(f"[querydata] error: {e}")

    report = _render()
    if a.out:
        Path(a.out).write_text(report, encoding="utf-8")
        sys.stderr.write(f"wrote {len(report.encode('utf-8'))} bytes to {a.out}\n")
    else:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
