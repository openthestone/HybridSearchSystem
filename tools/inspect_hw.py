#!/usr/bin/env python3
"""Inspect a dataset_HW.bin (HYDSET2) file: header, per-doc tag density, and
vector sanity. Reads only a sample of docs, so it's fast even on 45G files.

Layout (see old/test/full_npu/common/dataset_hw.h):
  40B header: magic[8] "HYDSET2\0", u32 version, u32 pad, u64 doc_num,
              u32 vector_dim, u32 tag_num, u32 reserved, u32 pad
  vectors: doc_num * vector_dim * f32
  bitmaps: doc_num * stride * u64,  stride = ceil(tag_num/64)

Usage:
  python3 inspect_hw.py /root/.../dataset_HW.bin [--sample 2000]
"""
import argparse, mmap, struct, sys, random, math

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--sample", type=int, default=2000, help="docs to sample")
    ap.add_argument("--docs", type=int, default=0, help="only consider first N docs (0=all); use the same N you built the index with")
    ap.add_argument("--filter", default="", help="filter_expr_600.txt to cross-check tag ids against this data")
    a = ap.parse_args()

    f = open(a.path, "rb")
    hdr = f.read(40)
    magic = hdr[:8].split(b"\0")[0].decode("ascii", "replace")
    version, _pad0, doc_num, vdim, tag_num, reserved, _pad1 = struct.unpack("<IIQIIII", hdr[8:40])
    stride = (tag_num + 63) // 64
    vec_off = 40
    bmp_off = 40 + doc_num * vdim * 4
    import os
    fsize = os.path.getsize(a.path)
    expect = bmp_off + doc_num * stride * 8

    print(f"=== header ===")
    print(f"magic={magic!r} version={version} doc_num={doc_num} vector_dim={vdim} tag_num={tag_num}")
    print(f"stride(u64 words)={stride}  file_size={fsize/1e9:.2f}GB  expected={expect/1e9:.2f}GB  match={'OK' if fsize==expect else 'MISMATCH!'}")
    if magic != "HYDSET2":
        print("!! bad magic — not a HYDSET2 file"); return

    mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
    N = doc_num if a.docs == 0 else min(a.docs, doc_num)
    sample = min(a.sample, N)
    idxs = random.sample(range(N), sample) if sample < N else list(range(N))

    # ---- tag density ----
    tag_counts = []
    tag_freq = {}   # tag -> how many sampled docs have it (to spot degenerate tags)
    for d in idxs:
        base = bmp_off + d * stride * 8
        words = struct.unpack_from(f"<{stride}Q", mm, base)
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
    tag_counts.sort()
    n = len(tag_counts)
    def pct(p): return tag_counts[min(n-1, int(p*n))]
    avg = sum(tag_counts)/n
    print(f"\n=== per-doc tag count (sampled {n} of {N} docs) ===")
    print(f"avg={avg:.1f}  min={tag_counts[0]}  p50={pct(.5)}  p90={pct(.9)}  p99={pct(.99)}  max={tag_counts[-1]}")
    print(f"density = avg/tag_num = {avg/tag_num*100:.1f}%  (of {tag_num} possible tags)")
    # most-common tags in the sample (universal tags inflate filter cost)
    top = sorted(tag_freq.items(), key=lambda kv: -kv[1])[:10]
    print("top-10 most-common tags (tag: fraction of sampled docs):")
    for t, c in top:
        print(f"  tag {t}: {c/n*100:.1f}%")

    # ---- tag-frequency structure (is it power-law: universal/common/rare?) ----
    # buckets by per-tag frequency (fraction of sampled docs that have the tag).
    b_univ = sum(1 for c in tag_freq.values() if c/n >= 0.90)
    b_common = sum(1 for c in tag_freq.values() if 0.05 <= c/n < 0.90)
    b_low = sum(1 for c in tag_freq.values() if 0.01 <= c/n < 0.05)
    b_rareseen = sum(1 for c in tag_freq.values() if c/n < 0.01)
    seen = len(tag_freq)
    print(f"\n=== tag-frequency structure (distinct tags seen in sample: {seen} of {tag_num}) ===")
    print(f"universal (>=90%): {b_univ}")
    print(f"common   (5-90%): {b_common}")
    print(f"low      (1-5%) : {b_low}")
    print(f"rare     (<1%)  : {b_rareseen} seen  (+ {tag_num-seen} never seen in sample => very rare/absent)")
    print("(realistic power-law ~ a few universal + tens of thousands common + long rare tail;")
    print(" uniform Bernoulli(0.5) would put ALL ~{} tags near 50%)".format(tag_num))

    # ---- optional: cross-check a filter file against the data ----
    if a.filter:
        import re
        exprs = [ln.strip() for ln in open(a.filter) if ln.strip()]
        all_tags = []
        for e in exprs:
            all_tags += [int(x) for x in re.findall(r"\d+", e)]
        distinct = sorted(set(all_tags))
        oor = [t for t in distinct if t >= tag_num]
        # frequency of referenced tags (from sample); missing => rare/absent
        def bucket(t):
            c = tag_freq.get(t, 0) / n
            if t >= tag_num: return "out-of-range"
            if c >= 0.90: return "universal"
            if c >= 0.05: return "common"
            if c >= 0.01: return "low"
            return "rare/absent"
        from collections import Counter
        comp = Counter(bucket(t) for t in distinct)
        print(f"\n=== filter cross-check: {a.filter} ===")
        print(f"exprs={len(exprs)}  total tag refs={len(all_tags)}  distinct tags={len(distinct)}  "
              f"tag id range=[{distinct[0]}, {distinct[-1]}]")
        print(f"out-of-range (>= tag_num {tag_num}): {len(oor)}"
              + (f"  e.g. {oor[:8]}" if oor else "  (all in range, good)"))
        print(f"referenced-tag composition (drives filter cost — 'common/universal' => big bitset postings):")
        for k in ("universal", "common", "low", "rare/absent", "out-of-range"):
            if comp.get(k): print(f"  {k}: {comp[k]}")

    # ---- vector sanity ----
    norms, zeros = [], 0
    vmin, vmax = math.inf, -math.inf
    for d in idxs[:min(1000, len(idxs))]:
        base = vec_off + d * vdim * 4
        v = struct.unpack_from(f"<{vdim}f", mm, base)
        s = sum(x*x for x in v)
        norms.append(math.sqrt(s))
        if s == 0: zeros += 1
        vmin = min(vmin, min(v)); vmax = max(vmax, max(v))
    norms.sort()
    m = len(norms)
    print(f"\n=== vector sanity (sampled {m} docs) ===")
    print(f"L2 norm: min={norms[0]:.3f} p50={norms[m//2]:.3f} max={norms[-1]:.3f}  zero-vectors={zeros}")
    print(f"value range: [{vmin:.3f}, {vmax:.3f}]")
    print(f"(normalized vectors would have norm≈1; raw embeddings vary)")

if __name__ == "__main__":
    main()
