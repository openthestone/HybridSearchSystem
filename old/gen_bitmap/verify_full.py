#!/usr/bin/env python3
"""
Full-dataset verification for dataset_HW_v2.bin.
Streams through bitmap section, computes:
  - header sanity
  - vectors match input (spot-check 10 random docs)
  - per-doc tag count stats (sampled)
  - tag freq stats (full scan)
"""
import os
import struct
import sys
import numpy as np

NEW_PATH = '/root/sks_hw/gen_bitmap/dataset_HW_v2.bin'
OLD_PATH = '/root/sks_hw/dataset_HW.bin'

TAG_NUM = 35672
TAG_STRIDE = (TAG_NUM + 63) // 64  # 558
HEADER_SIZE = 40
DOC_NUM = 10_000_000
VECTOR_DIM = 64
UNIVERSAL_NUM = 20
COMMON_NUM = 3052


def popcount_arr(u64):
    u64 = u64 - ((u64 >> 1) & 0x5555555555555555)
    u64 = (u64 & 0x3333333333333333) + ((u64 >> 2) & 0x3333333333333333)
    u64 = (u64 + (u64 >> 4)) & 0x0f0f0f0f0f0f0f0f
    return (u64 * 0x0101010101010101) >> 56


def main():
    print(f"=== verify {NEW_PATH} ===")
    sz = os.path.getsize(NEW_PATH)
    expected = HEADER_SIZE + DOC_NUM*VECTOR_DIM*4 + DOC_NUM*TAG_STRIDE*8
    print(f"size: {sz} (expected {expected}, match={sz==expected})")
    assert sz == expected

    # 1. header
    with open(NEW_PATH, 'rb') as f:
        hdr = f.read(40)
    magic = hdr[:8]
    version, = struct.unpack('<I', hdr[8:12])
    doc_num, = struct.unpack('<Q', hdr[16:24])
    vector_dim, tag_num, reserved = struct.unpack('<III', hdr[24:36])
    print(f"\nheader: magic={magic} version={version} doc_num={doc_num} "
          f"vector_dim={vector_dim} tag_num={tag_num}")
    assert magic == b'HYDSET2\x00' and version == 2
    assert doc_num == DOC_NUM and vector_dim == 64 and tag_num == TAG_NUM

    # 2. vector spot-check (10 random docs)
    print("\nvector spot-check (10 random docs vs input):")
    rng = np.random.default_rng(42)
    sample_docs = sorted(rng.choice(DOC_NUM, size=10, replace=False))
    vector_size_per_doc = VECTOR_DIM * 4
    vec_section_off = HEADER_SIZE

    with open(OLD_PATH, 'rb') as fo, open(NEW_PATH, 'rb') as fn:
        for d in sample_docs:
            off = vec_section_off + d * vector_size_per_doc
            fo.seek(off); fn.seek(off)
            v_o = np.frombuffer(fo.read(vector_size_per_doc), dtype=np.float32)
            v_n = np.frombuffer(fn.read(vector_size_per_doc), dtype=np.float32)
            diff = np.abs(v_o - v_n).max()
            assert diff == 0.0, f"doc {d} vector mismatch!"
    print(f"  10 docs sampled (ids {sample_docs[:3]}...{sample_docs[-2:]}): all match")

    # 3. bitmap full scan: tag freq + per-doc count stats
    print("\nbitmap scan (sample 1M docs evenly across 10M):")
    bitmap_section_off = HEADER_SIZE + DOC_NUM * vector_size_per_doc
    chunk_docs = 10_000       # smaller chunks = less mem
    sample_stride = 10        # sample 1 in 10 docs (1M total)
    bytes_per_row = TAG_STRIDE * 8
    tag_doc_count = np.zeros(TAG_NUM, dtype=np.int64)
    row_count_samples = []
    bytes_per_row = TAG_STRIDE * 8

    with open(NEW_PATH, 'rb') as f:
        docs_done = 0
        sampled = 0
        while docs_done < DOC_NUM:
            # jump by sample_stride chunks
            f.seek(bitmap_section_off + docs_done * bytes_per_row)
            n = chunk_docs
            if docs_done + n > DOC_NUM:
                n = DOC_NUM - docs_done
            data = f.read(n * bytes_per_row)
            if len(data) < n * bytes_per_row:
                print(f"  SHORT READ at doc {docs_done}: got {len(data)}")
                break
            bm = np.frombuffer(data, dtype=np.uint64).reshape(n, TAG_STRIDE)
            # per-doc popcount (all sampled docs)
            pc = popcount_arr(bm).sum(axis=1)
            row_count_samples.extend(pc.tolist())
            # tag_doc_count vectorized (bitorder='little' for correct bit->tag mapping)
            bm_bytes = bm.view(np.uint8).reshape(n, TAG_STRIDE, 8)
            bits = np.unpackbits(bm_bytes, axis=-1, bitorder='little')
            tag_doc_count_in_chunk = bits.sum(axis=0, dtype=np.int64).reshape(-1)
            tag_doc_count[:TAG_NUM] += tag_doc_count_in_chunk[:TAG_NUM]
            docs_done += sample_stride * chunk_docs  # skip 9 chunks
            sampled += n
            if sampled % 100_000 == 0:
                print(f"  sampled {sampled//1000}K docs (jumped to {docs_done//1_000_000}M)", flush=True)
        print(f"\n  total sampled: {sampled} docs (1 of every {sample_stride})", flush=True)
        # scale up freq estimate
        scale = DOC_NUM / sampled
        tag_freq = tag_doc_count * scale / DOC_NUM
    tag_freq = tag_doc_count / sampled  # use actual sample size

    print(f"\n--- bitmap stats (sampled {len(row_count_samples)} docs, freq est from sample) ---")
    tag_freq = tag_doc_count / sampled
    print(f"\n[tag freq by tier]")
    print(f"  universal (0..{UNIVERSAL_NUM}): "
          f"min={tag_freq[:UNIVERSAL_NUM].min():.4f} "
          f"mean={tag_freq[:UNIVERSAL_NUM].mean():.4f} "
          f"max={tag_freq[:UNIVERSAL_NUM].max():.4f}")
    common = tag_freq[UNIVERSAL_NUM:UNIVERSAL_NUM+COMMON_NUM]
    print(f"  common ({UNIVERSAL_NUM}..{UNIVERSAL_NUM+COMMON_NUM}): "
          f"min={common.min():.6f} "
          f"P25={np.percentile(common, 25):.4f} "
          f"median={np.median(common):.4f} "
          f"P75={np.percentile(common, 75):.4f} "
          f"max={common.max():.4f}")
    rare = tag_freq[UNIVERSAL_NUM+COMMON_NUM:]
    print(f"  rare ({UNIVERSAL_NUM+COMMON_NUM}..{TAG_NUM}): "
          f"min={rare.min():.8f} "
          f"max={rare.max():.6f} "
          f"mean={rare.mean():.8f} "
          f"zero_tags={int((rare==0).sum())}/{len(rare)}")

    # power-law shape: top 20 freq
    sorted_freq = np.sort(tag_freq)[::-1]
    print(f"\n[sorted freq curve]")
    print(f"  top 5    : {sorted_freq[:5]}")
    print(f"  rank 20  : {sorted_freq[19]:.4f}")
    print(f"  rank 100 : {sorted_freq[99]:.4f}")
    print(f"  rank 1000: {sorted_freq[999]:.4f}")
    print(f"  rank 10000: {sorted_freq[9999]:.4f}")
    print(f"  P50      : {sorted_freq[TAG_NUM//2]:.6f}")
    print(f"  rank last 5: {sorted_freq[-5:]}")

    # per-doc tag count
    rc = np.array(row_count_samples)
    print(f"\n[per-doc tag count] (n={len(rc)} samples)")
    print(f"  mean   = {rc.mean():.1f}  (target 400-800)")
    print(f"  median = {np.median(rc):.1f}")
    print(f"  P25    = {np.percentile(rc, 25):.0f}")
    print(f"  P75    = {np.percentile(rc, 75):.0f}")
    print(f"  P99    = {np.percentile(rc, 99):.0f}")
    print(f"  min    = {rc.min()}")
    print(f"  max    = {rc.max()}")

    # sentinel folding check: how many tags are universal (freq>0.9) or zero
    univ_count = int((tag_freq > 0.9).sum())
    zero_count = int((tag_freq == 0).sum())
    print(f"\n[sentinel folding potential]")
    print(f"  tags with freq > 0.9  (kSentOnes): {univ_count}")
    print(f"  tags with freq == 0   (kSentZero): {zero_count}")
    print(f"  total tags with extreme freq: {univ_count + zero_count} / {TAG_NUM} "
          f"({100*(univ_count+zero_count)/TAG_NUM:.1f}%)")


if __name__ == '__main__':
    main()
