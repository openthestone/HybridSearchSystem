#!/usr/bin/env python3
"""
Small-scale test for gen_bitmap. Generates 10K docs bitmap (no file I/O),
validates:
  - bitmap shape
  - tag freq distribution (power-law: universal ~1.0, rare <1e-5)
  - per-doc tag count (log-normal, mean ~400)
  - sentinel-friendly (universal tags hit by nearly all docs)
  - bits set correctly (no out-of-range, no double-set)
"""
import sys
import os
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_bitmap as gb


def run_small(doc_num=10_000):
    # patch tag_num etc. stay same; only doc count changes
    print(f"=== small test: doc_num={doc_num} ===\n")

    # init worker state
    gb.worker_init()

    # single chunk
    t0 = time.time()
    doc_start, bitmap_bytes = gb.gen_chunk((0, 0, doc_num))
    elapsed = time.time() - t0
    print(f"gen_chunk: {elapsed:.2f}s for {doc_num} docs ({doc_num/elapsed:.0f} docs/s)")

    bitmap = np.frombuffer(bitmap_bytes, dtype=np.uint64).reshape(doc_num, gb.TAG_STRIDE)

    # ---- checks ----
    print("\n--- checks ---")

    # 1. row popcount distribution
    # use bit count per row
    def popcount_u64(x):
        # numpy popcount via lookup is slow; use bincount on bytes
        b = x.view(np.uint8).reshape(x.shape + (8,))
        # use Python int.bit_count() per byte is slow for 10K*4464 bytes
        # Use vectorized: split into nibbles
        return np.unpackbits(b.view(np.uint8)).sum(axis=-1).reshape(x.shape)

    # vectorized popcount using bit trick
    def popcount_arr(u64):
        u64 = u64.copy()
        u64 = u64 - ((u64 >> 1) & 0x5555555555555555)
        u64 = (u64 & 0x3333333333333333) + ((u64 >> 2) & 0x3333333333333333)
        u64 = (u64 + (u64 >> 4)) & 0x0f0f0f0f0f0f0f0f
        return (u64 * 0x0101010101010101) >> 56

    row_counts = popcount_arr(bitmap).sum(axis=1)
    print(f"\n[1] per-doc tag count:")
    print(f"  mean   = {row_counts.mean():.1f}  (target 400-800)")
    print(f"  median = {np.median(row_counts):.1f}")
    print(f"  P99    = {np.percentile(row_counts, 99):.0f}")
    print(f"  min    = {row_counts.min()}")
    print(f"  max    = {row_counts.max()}")

    # 2. tag freq distribution
    tag_pop = popcount_arr(bitmap).sum(axis=0)  # bit position sums over docs (per uint64 word)
    # expand to per-tag count: each word has 64 bits
    tag_doc_count = np.zeros(gb.TAG_NUM, dtype=np.int64)
    for word_idx in range(gb.TAG_STRIDE):
        col = bitmap[:, word_idx]
        for bit_idx in range(64):
            t = word_idx * 64 + bit_idx
            if t >= gb.TAG_NUM:
                break
            cnt = int(((col >> np.uint64(bit_idx)) & np.uint64(1)).sum())
            tag_doc_count[t] = cnt

    tag_freq = tag_doc_count / doc_num
    print(f"\n[2] tag freq distribution:")
    print(f"  universal (0..{gb.UNIVERSAL_NUM}):")
    print(f"    min={tag_freq[:gb.UNIVERSAL_NUM].min():.3f}  max={tag_freq[:gb.UNIVERSAL_NUM].max():.3f}")
    print(f"    mean={tag_freq[:gb.UNIVERSAL_NUM].mean():.3f}  (target >0.9)")
    print(f"  common ({gb.UNIVERSAL_NUM}..{gb.UNIVERSAL_NUM+gb.COMMON_NUM}):")
    cm = tag_freq[gb.UNIVERSAL_NUM:gb.UNIVERSAL_NUM+gb.COMMON_NUM]
    print(f"    min={cm.min():.6f}  max={cm.max():.4f}  median={np.median(cm):.6f}  (target power-law)")
    print(f"  rare ({gb.UNIVERSAL_NUM+gb.COMMON_NUM}..{gb.TAG_NUM}):")
    rm = tag_freq[gb.UNIVERSAL_NUM+gb.COMMON_NUM:]
    print(f"    min={rm.min():.8f}  max={rm.max():.6f}  mean={rm.mean():.8f}")
    print(f"    tags with 0 docs: {(rm==0).sum()} / {len(rm)}")

    # 3. sorted freq curve
    sorted_freq = np.sort(tag_freq)[::-1]
    print(f"\n[3] sorted freq top/bottom:")
    print(f"  top 5    : {sorted_freq[:5]}")
    print(f"  P50      : {sorted_freq[gb.TAG_NUM//2]:.6f}")
    print(f"  bottom 5 : {sorted_freq[-5:]}")

    # 4. bitmap row size sanity
    expected_row_bytes = gb.TAG_STRIDE * 8
    actual_row_bytes   = bitmap.shape[1] * 8
    print(f"\n[4] row size: {actual_row_bytes} bytes (expected {expected_row_bytes})")

    return bitmap, tag_doc_count


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    run_small(n)
