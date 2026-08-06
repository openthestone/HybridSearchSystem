#!/usr/bin/env python3
"""
Diagnostic: for each expr, show the top-3 AND leaves' individual tag freq
and the implied joint probability. Explains why sel_rate = 0 on synthetic data.
"""
import os, sys, struct
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calc_sel_rate import parse_expr, tokenize, Parser, Node, collect_tags

DATASET = '/root/sks_hw/gen_bitmap/dataset_HW_v2.bin'
EXPR_FILE = '/root/sks_hw/filter_expr_600.txt'

HEADER_SIZE = 40
TAG_NUM = 35672
TAG_STRIDE = (TAG_NUM + 63) // 64
DOC_NUM = 10_000_000
VECTOR_DIM = 64
UNIVERSAL_NUM = 20
COMMON_NUM = 30000


def get_top_and_leaves(ast):
    """Return the list of direct children of the root AND chain."""
    if ast.kind == 'leaf':
        return [ast.tag]
    if ast.op == 'AND':
        # flatten: collect direct leaves and OR/NOT groups
        leaves = []
        for c in ast.children:
            if c.kind == 'leaf':
                leaves.append(c.tag)
        return leaves
    return []


def main():
    # 1. parse exprs
    with open(EXPR_FILE) as f:
        lines = [l.strip() for l in f if l.strip()]
    asts = [parse_expr(l) for l in lines]

    # 2. collect all referenced tags across all exprs
    all_tags = set()
    per_expr_tags = []
    for ast in asts:
        s = set()
        collect_tags(ast, s)
        per_expr_tags.append(s)
        all_tags |= s
    all_tags_arr = sorted(all_tags)
    print(f"total unique tags across all exprs: {len(all_tags_arr)}")

    # 3. compute tag freq for all referenced tags (sample 1M)
    bitmap_section_off = HEADER_SIZE + DOC_NUM * VECTOR_DIM * 4
    bytes_per_row = TAG_STRIDE * 8
    chunk_docs = 10_000
    sample_stride = 10

    tag_count = {t: 0 for t in all_tags_arr}
    total = 0
    t0 = time.time()
    with open(DATASET, 'rb') as f:
        docs_done = 0
        while docs_done < DOC_NUM:
            f.seek(bitmap_section_off + docs_done * bytes_per_row)
            n = chunk_docs
            if docs_done + n > DOC_NUM:
                n = DOC_NUM - docs_done
            data = f.read(n * bytes_per_row)
            if len(data) < n * bytes_per_row:
                break
            bm = np.frombuffer(data, dtype=np.uint64).reshape(n, TAG_STRIDE)
            # vectorized: for each tag, count
            for t in all_tags_arr:
                word = t // 64
                bit = t % 64
                col = bm[:, word]
                cnt = int(((col >> np.uint64(bit)) & np.uint64(1)).sum())
                tag_count[t] += cnt
            total += n
            docs_done += sample_stride * chunk_docs
    print(f"sampled {total} docs in {time.time()-t0:.1f}s")

    tag_freq = {t: tag_count[t] / total for t in all_tags_arr}

    # 4. for each expr: show top-3 AND leaves' freq + implied joint
    print(f"\n=== per-expr diagnostic ===")
    print(f"{'idx':>4} {'first_AND_leaves':>40} {'individual_freq':>40} {'implied_joint':>15} {'tier':>10}")
    for i, ast in enumerate(asts):
        leaves = get_top_and_leaves(ast)
        first3 = leaves[:3]
        freqs = [tag_freq.get(t, 0) for t in first3]
        joint = 1.0
        for fr in freqs:
            joint *= max(fr, 1e-9)
        # tier classification
        tiers = []
        for t in first3:
            if t < UNIVERSAL_NUM: tiers.append('U')
            elif t < UNIVERSAL_NUM + COMMON_NUM: tiers.append('C')
            else: tiers.append('R')
        tier_str = '/'.join(tiers)
        print(f"{i:>4} {str(first3):>40} {str([f'{f:.4f}' for f in freqs]):>40} {joint:>15.2e} {tier_str:>10}")

    # 5. also: how many expr leaves fall in each tier
    print(f"\n=== expr tag tier breakdown ===")
    print(f"{'idx':>4} {'total_leaves':>13} {'universal':>10} {'common':>8} {'rare':>6} {'rare_pct':>10}")
    for i, tags in enumerate(per_expr_tags):
        u = sum(1 for t in tags if t < UNIVERSAL_NUM)
        c = sum(1 for t in tags if UNIVERSAL_NUM <= t < UNIVERSAL_NUM + COMMON_NUM)
        r = sum(1 for t in tags if t >= UNIVERSAL_NUM + COMMON_NUM)
        pct = 100 * r / len(tags) if tags else 0
        print(f"{i:>4} {len(tags):>13} {u:>10} {c:>8} {r:>6} {pct:>9.1f}%")


if __name__ == '__main__':
    main()
