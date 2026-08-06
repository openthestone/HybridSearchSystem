#!/usr/bin/env python3
"""
dataset_HW_v2.bin bitmap generator.

Generates bitmap section using 3-layer tag model + cluster affinity.
Vector section copied verbatim from input dataset_HW.bin.

See README.md for full design.
"""
import numpy as np
import struct
import os
import sys
import time
import multiprocessing as mp
from collections import defaultdict

# ============================================================
# Config
# ============================================================
DOC_NUM       = 10_000_000
VECTOR_DIM    = 64
TAG_NUM       = 35672        # user-specified

UNIVERSAL_NUM = 20
COMMON_NUM    = 30000
RARE_NUM      = TAG_NUM - UNIVERSAL_NUM - COMMON_NUM   # 5652

CLUSTER_NUM   = 16           # 3052/16 ~= 190 tags/cluster; 3 clusters ~= 570 candidates

LOGNORMAL_MU     = 7.0      # ln(1100) ~= 7.0  (kept for compat, unused in Bernoulli mode)
LOGNORMAL_SIGMA  = 0.5
TAG_COUNT_MIN    = 200
TAG_COUNT_MAX    = 8000

UNIVERSAL_FREQ_MIN = 0.90
UNIVERSAL_FREQ_MAX = 1.00
COMMON_ZIPF_S      = 0.7
COMMON_FREQ_MAX    = 0.7

# Bernoulli mode (target ~500 matches/expr on filter_expr_600.txt)
BERNOULLI_MODE         = True
COMMON_BASE_FREQ_MEDIAN = 0.09    # bumped: sel ~1.4x prior, median ~660 matches/expr
COMMON_BASE_FREQ_SIGMA  = 0.2     # tight spread
RARE_FREQ_MAX      = 1e-5
RARE_FREQ_MIN      = 1e-7
RARE_ATTACH_PROB   = 0.30
RARE_ATTACH_MAX    = 3

CLUSTER_PER_DOC_MIN = 2
CLUSTER_PER_DOC_MAX = 5
UNIVERSAL_MIN_PER_DOC = 3
UNIVERSAL_MAX_PER_DOC = 8

SEED = 42

INPUT_PATH  = '/root/sks_hw/dataset_HW.bin'
OUTPUT_PATH = '/root/sks_hw/gen_bitmap/dataset_HW_v2.bin'

# Derived
TAG_STRIDE  = (TAG_NUM + 63) // 64              # 558
HEADER_SIZE = 40
VECTOR_SIZE = DOC_NUM * VECTOR_DIM * 4          # 2.56 GB
BITMAP_SIZE = DOC_NUM * TAG_STRIDE * 8          # ~44.64 GB
TOTAL_SIZE  = HEADER_SIZE + VECTOR_SIZE + BITMAP_SIZE

# Tier id ranges
UNIVERSAL_IDS = np.arange(0, UNIVERSAL_NUM, dtype=np.int64)
COMMON_IDS    = np.arange(UNIVERSAL_NUM, UNIVERSAL_NUM + COMMON_NUM, dtype=np.int64)
RARE_IDS      = np.arange(UNIVERSAL_NUM + COMMON_NUM, TAG_NUM, dtype=np.int64)


# ============================================================
# Per-worker setup (computed once per process via init)
# ============================================================
_WORKER_STATE = {}

def worker_init():
    rng = np.random.default_rng(SEED)
    universal_freq = rng.uniform(UNIVERSAL_FREQ_MIN, UNIVERSAL_FREQ_MAX, UNIVERSAL_NUM).astype(np.float64)

    # Bernoulli mode: per-tag freq from LogNormal
    if BERNOULLI_MODE:
        common_freq = np.clip(
            rng.lognormal(np.log(COMMON_BASE_FREQ_MEDIAN), COMMON_BASE_FREQ_SIGMA, COMMON_NUM),
            0.001, 0.5,
        ).astype(np.float64)
    else:
        common_freq = np.minimum(
            COMMON_FREQ_MAX,
            0.3 / (np.arange(COMMON_NUM, dtype=np.float64) + 5.0) ** COMMON_ZIPF_S,
        ).astype(np.float64)

    rare_freq = rng.uniform(RARE_FREQ_MIN, RARE_FREQ_MAX, RARE_NUM).astype(np.float64)

    cluster_pop = 1.0 / (np.arange(CLUSTER_NUM, dtype=np.float64) + 1.0) ** 0.8
    cluster_pop /= cluster_pop.sum()

    cluster_of = (np.arange(COMMON_NUM) % CLUSTER_NUM).astype(np.int32)
    tags_in_cluster = [None] * CLUSTER_NUM
    weights_in_cluster = [None] * CLUSTER_NUM
    for c in range(CLUSTER_NUM):
        idx = np.where(cluster_of == c)[0]
        tags = COMMON_IDS[idx]
        w = common_freq[idx]
        tags_in_cluster[c] = tags
        weights_in_cluster[c] = w / w.sum()

    _WORKER_STATE['rng'] = np.random.default_rng(SEED + os.getpid())
    _WORKER_STATE['universal_freq'] = universal_freq
    _WORKER_STATE['common_freq'] = common_freq
    _WORKER_STATE['rare_freq'] = rare_freq
    _WORKER_STATE['cluster_pop'] = cluster_pop
    _WORKER_STATE['tags_in_cluster'] = tags_in_cluster
    _WORKER_STATE['weights_in_cluster'] = weights_in_cluster


# ============================================================
# Per-doc generation (inner loop, must be fast)
# ============================================================
# Precompute bit weights for vectorized packing (constant)
_BIT_WEIGHTS = (np.uint64(1) << np.arange(64, dtype=np.uint64))


def gen_chunk(args):
    chunk_idx, doc_start, doc_count = args
    rng = np.random.default_rng(SEED + chunk_idx * 7919)

    if BERNOULLI_MODE:
        return _gen_chunk_bernoulli(chunk_idx, doc_start, doc_count, rng)
    return _gen_chunk_cluster(chunk_idx, doc_start, doc_count, rng)


def _gen_chunk_bernoulli(chunk_idx, doc_start, doc_count, rng):
    universal_freq = _WORKER_STATE['universal_freq']
    common_freq    = _WORKER_STATE['common_freq']
    rare_freq      = _WORKER_STATE['rare_freq']

    # process in sub-chunks to bound memory
    SUB = 500
    bitmap = np.zeros((doc_count, TAG_STRIDE), dtype=np.uint64)
    bit_weights = _BIT_WEIGHTS

    for sub_start in range(0, doc_count, SUB):
        sub_n = min(SUB, doc_count - sub_start)
        # build bool matrix (sub_n, TAG_NUM)
        bits = np.zeros((sub_n, TAG_NUM), dtype=bool)
        # universal
        bits[:, :UNIVERSAL_NUM] = rng.random((sub_n, UNIVERSAL_NUM)) < universal_freq
        # common
        bits[:, UNIVERSAL_NUM:UNIVERSAL_NUM+COMMON_NUM] = (
            rng.random((sub_n, COMMON_NUM)) < common_freq
        )
        # rare: per-doc Bernoulli attach 0-3 rare tags
        for i in range(sub_n):
            if rng.random() < RARE_ATTACH_PROB:
                rp = int(rng.integers(0, RARE_ATTACH_MAX + 1))
                if rp > 0:
                    rp = min(rp, RARE_NUM)
                    picks = rng.choice(RARE_NUM, size=rp, replace=False)
                    bits[i, UNIVERSAL_NUM+COMMON_NUM+picks] = True

        # pack bits to uint64 words
        # pad to multiple of 64
        padded = np.zeros((sub_n, TAG_STRIDE * 64), dtype=np.uint64)
        padded[:, :TAG_NUM] = bits.astype(np.uint64)
        words = (padded.reshape(sub_n, TAG_STRIDE, 64) * bit_weights).sum(axis=2)
        bitmap[sub_start:sub_start+sub_n] = words

    return doc_start, bitmap.tobytes()


def _gen_chunk_cluster(chunk_idx, doc_start, doc_count, rng):
    universal_freq = _WORKER_STATE['universal_freq']
    cluster_pop    = _WORKER_STATE['cluster_pop']
    tags_in_cluster    = _WORKER_STATE['tags_in_cluster']
    weights_in_cluster = _WORKER_STATE['weights_in_cluster']

    bitmap = np.zeros((doc_count, TAG_STRIDE), dtype=np.uint64)

    for i in range(doc_count):
        k_clusters = int(rng.integers(CLUSTER_PER_DOC_MIN, CLUSTER_PER_DOC_MAX + 1))
        chosen = rng.choice(CLUSTER_NUM, size=k_clusters, replace=False, p=cluster_pop)

        n = int(rng.lognormal(LOGNORMAL_MU, LOGNORMAL_SIGMA))
        if n < TAG_COUNT_MIN: n = TAG_COUNT_MIN
        elif n > TAG_COUNT_MAX: n = TAG_COUNT_MAX

        univ_mask = rng.random(UNIVERSAL_NUM) < universal_freq
        doc_tags = UNIVERSAL_IDS[univ_mask].tolist()
        while len(doc_tags) < UNIVERSAL_MIN_PER_DOC:
            extra = int(rng.integers(0, UNIVERSAL_NUM))
            if extra not in doc_tags:
                doc_tags.append(int(extra))

        m_per_cluster = max(1, n // k_clusters)
        for c in chosen:
            tags = tags_in_cluster[c]
            w    = weights_in_cluster[c]
            pick = min(m_per_cluster, len(tags))
            picked = rng.choice(tags, size=pick, replace=False, p=w)
            doc_tags.extend(picked.tolist())

        if rng.random() < RARE_ATTACH_PROB:
            rp = int(rng.integers(0, RARE_ATTACH_MAX + 1))
            if rp > 0:
                rp = min(rp, RARE_NUM)
                picked = rng.choice(RARE_IDS, size=rp, replace=False)
                doc_tags.extend(picked.tolist())

        tags_arr = np.asarray(doc_tags, dtype=np.int64)
        words = tags_arr // 64
        bits  = tags_arr % 64
        keys = words.astype(np.int64) * 64 + bits
        uniq = np.unique(keys)
        u_words = (uniq // 64).astype(np.int64)
        u_bits  = (uniq % 64).astype(np.int64)
        uniq_words, inverse = np.unique(u_words, return_inverse=True)
        combined = np.zeros(len(uniq_words), dtype=np.uint64)
        np.bitwise_or.at(combined, inverse, np.uint64(1) << u_bits.astype(np.uint64))
        bitmap[i, uniq_words] = combined

    return doc_start, bitmap.tobytes()


# ============================================================
# Main pipeline
# ============================================================
def write_header(fout):
    hdr = bytearray(HEADER_SIZE)
    hdr[0:8] = b'HYDSET2\x00'
    struct.pack_into('<I', hdr, 8, 2)              # version
    struct.pack_into('<I', hdr, 12, 0)             # padding
    struct.pack_into('<Q', hdr, 16, DOC_NUM)
    struct.pack_into('<I', hdr, 24, VECTOR_DIM)
    struct.pack_into('<I', hdr, 28, TAG_NUM)
    struct.pack_into('<I', hdr, 32, 0)             # reserved
    struct.pack_into('<I', hdr, 36, 0)             # tail padding
    fout.seek(0)
    fout.write(bytes(hdr))


def copy_vectors(fin, fout, log_every_gb=1):
    fin.seek(HEADER_SIZE)
    fout.seek(HEADER_SIZE)
    copied = 0
    chunk_bytes = 256 * 1024 * 1024
    last_gb = -1
    t0 = time.time()
    while copied < VECTOR_SIZE:
        to_read = min(chunk_bytes, VECTOR_SIZE - copied)
        data = fin.read(to_read)
        if not data:
            break
        fout.write(data)
        copied += len(data)
        gb = copied // (1024**3)
        if gb != last_gb and gb >= log_every_gb:
            elapsed = time.time() - t0
            rate = copied / elapsed / 1e9 if elapsed > 0 else 0
            print(f"  vector {copied/1e9:.2f}/{VECTOR_SIZE/1e9:.2f} GB ({rate:.2f} GB/s)")
            last_gb = gb
    return copied


def main():
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: input {INPUT_PATH} missing", file=sys.stderr)
        sys.exit(1)

    print(f"=== dataset_HW_v2.bin generation ===")
    print(f"  doc_num     : {DOC_NUM}")
    print(f"  vector_dim  : {VECTOR_DIM}")
    print(f"  tag_num     : {TAG_NUM} (universal={UNIVERSAL_NUM}, common={COMMON_NUM}, rare={RARE_NUM})")
    print(f"  cluster_num : {CLUSTER_NUM}")
    print(f"  tag_stride  : {TAG_STRIDE} uint64/row")
    print(f"  sizes       : hdr={HEADER_SIZE}, vec={VECTOR_SIZE/1e9:.2f}GB, bmp={BITMAP_SIZE/1e9:.2f}GB, total={TOTAL_SIZE/1e9:.2f}GB")

    # 1. pre-allocate output
    if os.path.exists(OUTPUT_PATH):
        print(f"  removing existing {OUTPUT_PATH}")
        os.remove(OUTPUT_PATH)

    print(f"\n[1/3] Pre-allocating {TOTAL_SIZE/1e9:.2f} GB ...")
    with open(OUTPUT_PATH, 'wb') as f:
        f.truncate(TOTAL_SIZE)

    print(f"\n[2/3] Writing header + copying vectors ...")
    with open(INPUT_PATH, 'rb') as fin, open(OUTPUT_PATH, 'r+b') as fout:
        write_header(fout)
        n = copy_vectors(fin, fout)
        print(f"  vectors copied: {n/1e9:.2f} GB")

    # 3. parallel bitmap generation
    print(f"\n[3/3] Generating bitmap (parallel) ...")
    num_workers = min(mp.cpu_count(), 96)
    chunk_size  = 20_000    # docs per chunk -> ~89 MB bitmap per chunk
    chunks = []
    for start in range(0, DOC_NUM, chunk_size):
        n = min(chunk_size, DOC_NUM - start)
        chunks.append((start // chunk_size, start, n))
    print(f"  workers={num_workers}, chunks={len(chunks)}, chunk_size={chunk_size}")

    bitmap_base = HEADER_SIZE + VECTOR_SIZE
    t0 = time.time()
    written = 0
    with open(OUTPUT_PATH, 'r+b') as fout, \
         mp.Pool(num_workers, initializer=worker_init) as pool:
        done = 0
        for doc_start, bitmap_bytes in pool.imap_unordered(gen_chunk, chunks):
            offset = bitmap_base + doc_start * TAG_STRIDE * 8
            fout.seek(offset)
            fout.write(bitmap_bytes)
            written += len(bitmap_bytes)
            done += 1
            if done % 50 == 0 or done == len(chunks):
                elapsed = time.time() - t0
                docs_done = done * chunk_size
                rate = docs_done / elapsed if elapsed > 0 else 0
                eta = (DOC_NUM - docs_done) / rate if rate > 0 else 0
                print(f"  [{done}/{len(chunks)}] docs={docs_done}/{DOC_NUM} "
                      f"({rate/1e3:.1f}K docs/s, ETA {eta:.0f}s)")
    elapsed = time.time() - t0
    print(f"\n  bitmap done in {elapsed:.1f}s ({written/1e9:.2f} GB written)")

    actual = os.path.getsize(OUTPUT_PATH)
    print(f"\n=== DONE ===")
    print(f"  output   : {OUTPUT_PATH}")
    print(f"  size     : {actual} bytes (expected {TOTAL_SIZE}, match={actual==TOTAL_SIZE})")
    print(f"  total    : {time.time()-t0:.1f}s bitmap phase")


if __name__ == '__main__':
    main()
