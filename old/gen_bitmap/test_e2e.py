#!/usr/bin/env python3
"""
End-to-end small-scale test (10K docs):
  - patches DOC_NUM to 10_000
  - runs full pipeline (header + vector copy + bitmap gen)
  - reads back and verifies
"""
import os
import sys
import struct
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_bitmap as gb

SMALL_DOC_NUM = 10_000
SMALL_OUT = '/root/sks_hw/gen_bitmap/test_small.bin'


def patch_and_run():
    # patch constants
    gb.DOC_NUM = SMALL_DOC_NUM
    gb.VECTOR_SIZE = SMALL_DOC_NUM * gb.VECTOR_DIM * 4
    gb.BITMAP_SIZE = SMALL_DOC_NUM * gb.TAG_STRIDE * 8
    gb.TOTAL_SIZE  = gb.HEADER_SIZE + gb.VECTOR_SIZE + gb.BITMAP_SIZE
    gb.OUTPUT_PATH = SMALL_OUT
    gb.INPUT_PATH  = '/root/sks_hw/dataset_HW.bin'

    # truncate input handling: we read first SMALL_DOC_NUM vectors from input
    # but copy_vectors reads VECTOR_SIZE bytes starting at HEADER_SIZE.
    # Since input has 10M vectors, reading first 10K * 64 * 4 = 2.56MB is fine.

    gb.main()


def verify():
    print(f"\n=== verify {SMALL_OUT} ===")
    sz = os.path.getsize(SMALL_OUT)
    expected = gb.HEADER_SIZE + gb.VECTOR_SIZE + gb.BITMAP_SIZE
    print(f"size: {sz} (expected {expected})")
    assert sz == expected, "size mismatch"

    with open(SMALL_OUT, 'rb') as f:
        hdr = f.read(40)
    magic = hdr[:8]
    version, = struct.unpack('<I', hdr[8:12])
    doc_num, = struct.unpack('<Q', hdr[16:24])
    vector_dim, tag_num, reserved = struct.unpack('<III', hdr[24:36])
    print(f"magic={magic}, version={version}, doc_num={doc_num}, vector_dim={vector_dim}, tag_num={tag_num}")
    assert magic == b'HYDSET2\x00'
    assert version == 2
    assert doc_num == SMALL_DOC_NUM
    assert vector_dim == 64
    assert tag_num == gb.TAG_NUM

    # verify vectors: first 10 vectors should match input
    print("\nverify vectors (first 10 docs):")
    with open('/root/sks_hw/dataset_HW.bin', 'rb') as fin, open(SMALL_OUT, 'rb') as fout:
        fin.seek(40)
        fout.seek(40)
        v_in = np.frombuffer(fin.read(10 * 64 * 4), dtype=np.float32).reshape(10, 64)
        v_out = np.frombuffer(fout.read(10 * 64 * 4), dtype=np.float32).reshape(10, 64)
        diff = np.abs(v_in - v_out).max()
        print(f"  max diff = {diff}")
        assert diff == 0.0, "vector mismatch!"

    # verify bitmap: read all, check stats
    print("\nverify bitmap:")
    with open(SMALL_OUT, 'rb') as f:
        f.seek(gb.HEADER_SIZE + gb.VECTOR_SIZE)
        bm_bytes = f.read(gb.BITMAP_SIZE)
    bitmap = np.frombuffer(bm_bytes, dtype=np.uint64).reshape(SMALL_DOC_NUM, gb.TAG_STRIDE)

    def popcount_arr(u64):
        u64 = u64 - ((u64 >> 1) & 0x5555555555555555)
        u64 = (u64 & 0x3333333333333333) + ((u64 >> 2) & 0x3333333333333333)
        u64 = (u64 + (u64 >> 4)) & 0x0f0f0f0f0f0f0f0f
        return (u64 * 0x0101010101010101) >> 56

    row_counts = popcount_arr(bitmap).sum(axis=1)
    print(f"  per-doc tag count: mean={row_counts.mean():.0f} median={np.median(row_counts):.0f} min={row_counts.min()} max={row_counts.max()}")

    # tag freq
    tag_doc_count = np.zeros(gb.TAG_NUM, dtype=np.int64)
    for word_idx in range(gb.TAG_STRIDE):
        col = bitmap[:, word_idx]
        for bit_idx in range(64):
            t = word_idx * 64 + bit_idx
            if t >= gb.TAG_NUM: break
            tag_doc_count[t] = int(((col >> np.uint64(bit_idx)) & np.uint64(1)).sum())

    tag_freq = tag_doc_count / SMALL_DOC_NUM
    print(f"  universal freq: min={tag_freq[:gb.UNIVERSAL_NUM].min():.3f} mean={tag_freq[:gb.UNIVERSAL_NUM].mean():.3f}")
    print(f"  common freq:    min={tag_freq[gb.UNIVERSAL_NUM:gb.UNIVERSAL_NUM+gb.COMMON_NUM].min():.4f} "
          f"max={tag_freq[gb.UNIVERSAL_NUM:gb.UNIVERSAL_NUM+gb.COMMON_NUM].max():.4f}")
    rare_freq = tag_freq[gb.UNIVERSAL_NUM+gb.COMMON_NUM:]
    print(f"  rare freq:      min={rare_freq.min():.6f} max={rare_freq.max():.6f} "
          f"zero_tags={int((rare_freq==0).sum())}/{len(rare_freq)}")

    # selection rate test: simulate (universal AND common_mid AND common_mid)
    # should give high selection rate
    # simulate (rare AND rare) → low
    univ_t = 0  # universal tag id 0
    common_top = 20  # first common tag
    common_mid = 1500
    rare_t = 5000  # rare tag id
    docs_with_univ = set(np.where(((bitmap[:, univ_t//64] >> np.uint64(univ_t%64)) & np.uint64(1)))[0])
    docs_with_top  = set(np.where(((bitmap[:, common_top//64] >> np.uint64(common_top%64)) & np.uint64(1)))[0])
    docs_with_mid  = set(np.where(((bitmap[:, common_mid//64] >> np.uint64(common_mid%64)) & np.uint64(1)))[0])
    docs_with_rare = set(np.where(((bitmap[:, rare_t//64] >> np.uint64(rare_t%64)) & np.uint64(1)))[0])

    print(f"\n  selection rate simulation:")
    print(f"    univ(0) alone               : {len(docs_with_univ)/SMALL_DOC_NUM:.3f}")
    print(f"    univ AND common_top(20)     : {len(docs_with_univ & docs_with_top)/SMALL_DOC_NUM:.3f}")
    print(f"    common_mid(1500) alone      : {len(docs_with_mid)/SMALL_DOC_NUM:.3f}")
    print(f"    univ AND common_mid(1500)   : {len(docs_with_univ & docs_with_mid)/SMALL_DOC_NUM:.3f}")
    print(f"    rare(5000) alone            : {len(docs_with_rare)/SMALL_DOC_NUM:.4f}")
    print(f"    rare AND common_mid         : {len(docs_with_rare & docs_with_mid)/SMALL_DOC_NUM:.6f}")

    print("\n=== ALL CHECKS PASSED ===")


if __name__ == '__main__':
    patch_and_run()
    verify()
    # cleanup
    if os.path.exists(SMALL_OUT):
        os.remove(SMALL_OUT)
        print(f"\ncleaned up {SMALL_OUT}")
