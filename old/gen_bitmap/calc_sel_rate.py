#!/usr/bin/env python3
"""
Compute selection rate of each filter_expr_600.txt expression on dataset_HW_v2.bin.

Approach:
  - Parse each expr into AST (recursive descent)
  - Collect unique tag IDs per expr
  - Stream bitmap in chunks (multiprocessing); for each chunk build a (N, T) bool matrix per expr
  - Evaluate AST with vectorized numpy bitwise ops
  - Sum Trues across chunks -> match count -> selection rate
"""
import os
import sys
import struct
import numpy as np
import time
import re
import multiprocessing as mp

DATASET = '/root/sks_hw/gen_bitmap/dataset_HW_v2.bin'
EXPR_FILE = '/root/sks_hw/filter_expr_600.txt'

HEADER_SIZE = 40
TAG_NUM = 35672
TAG_STRIDE = (TAG_NUM + 63) // 64  # 558
DOC_NUM = 10_000_000
VECTOR_DIM = 64
CHUNK_DOCS = 50_000


# ============================================================
# Parser
# ============================================================
class Node:
    __slots__ = ('kind', 'op', 'tag', 'children')
    def __init__(self, kind, op=None, tag=None, children=None):
        self.kind = kind        # 'leaf' or 'op'
        self.op = op            # 'AND', 'OR', 'NOT'
        self.tag = tag          # int for leaf
        self.children = children or []

    def __repr__(self):
        if self.kind == 'leaf':
            return f"T{self.tag}"
        if self.op == 'NOT':
            return f"NOT({self.children[0]})"
        return f"({f' {self.op} '.join(map(str, self.children))})"


def tokenize(s):
    # tokens: int, AND, OR, NOT, (, )
    tokens = []
    i = 0
    while i < len(s):
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c == '(':
            tokens.append(('LP', '('))
            i += 1
        elif c == ')':
            tokens.append(('RP', ')'))
            i += 1
        elif c.isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            tokens.append(('NUM', int(s[i:j])))
            i = j
        else:
            # word: AND/OR/NOT
            j = i
            while j < len(s) and s[j].isalpha():
                j += 1
            word = s[i:j]
            if word == 'AND':
                tokens.append(('AND', word))
            elif word == 'OR':
                tokens.append(('OR', word))
            elif word == 'NOT':
                tokens.append(('NOT', word))
            else:
                raise ValueError(f"unknown token {word!r} at pos {i}")
            i = j
    return tokens


class Parser:
    """Recursive descent: expr := or_expr; or_expr := and_expr (OR and_expr)*;
       and_expr := not_expr (AND not_expr)*; not_expr := NOT not_expr | atom;
       atom := NUM | LP expr RP"""
    def __init__(self, tokens):
        self.toks = tokens
        self.pos = 0

    def peek(self):
        return self.toks[self.pos] if self.pos < len(self.toks) else (None, None)

    def advance(self):
        t = self.toks[self.pos]
        self.pos += 1
        return t

    def parse_expr(self):
        node = self.parse_or()
        return node

    def parse_or(self):
        left = self.parse_and()
        while self.peek()[0] == 'OR':
            self.advance()
            right = self.parse_and()
            if left.kind == 'op' and left.op == 'OR':
                left.children.append(right)
            else:
                left = Node('op', op='OR', children=[left, right])
        return left

    def parse_and(self):
        left = self.parse_not()
        while self.peek()[0] == 'AND':
            self.advance()
            right = self.parse_not()
            if left.kind == 'op' and left.op == 'AND':
                left.children.append(right)
            else:
                left = Node('op', op='AND', children=[left, right])
        return left

    def parse_not(self):
        if self.peek()[0] == 'NOT':
            self.advance()
            child = self.parse_not()
            return Node('op', op='NOT', children=[child])
        return self.parse_atom()

    def parse_atom(self):
        t = self.peek()
        if t[0] == 'NUM':
            self.advance()
            return Node('leaf', tag=t[1])
        elif t[0] == 'LP':
            self.advance()
            node = self.parse_expr()
            assert self.peek()[0] == 'RP', f"expected ) got {self.peek()}"
            self.advance()
            return node
        raise ValueError(f"unexpected token {t}")


def parse_expr(s):
    toks = tokenize(s)
    p = Parser(toks)
    ast = p.parse_expr()
    assert p.pos == len(toks), "extra tokens"
    return ast


def collect_tags(node, acc):
    if node.kind == 'leaf':
        acc.add(node.tag)
    else:
        for c in node.children:
            collect_tags(c, acc)


def count_leaves(node):
    if node.kind == 'leaf':
        return 1
    return sum(count_leaves(c) for c in node.children)


# ============================================================
# Vectorized evaluator
# ============================================================
def eval_ast(node, tag_bits):
    """tag_bits: dict {tag_id: bool_array shape (N,)}"""
    if node.kind == 'leaf':
        return tag_bits[node.tag]
    if node.op == 'NOT':
        return ~eval_ast(node.children[0], tag_bits)
    if node.op == 'AND':
        result = eval_ast(node.children[0], tag_bits)
        for c in node.children[1:]:
            result &= eval_ast(c, tag_bits)
        return result
    if node.op == 'OR':
        result = eval_ast(node.children[0], tag_bits)
        for c in node.children[1:]:
            result |= eval_ast(c, tag_bits)
        return result
    raise ValueError(f"bad node {node}")


# ============================================================
# Main
# ============================================================
_WORKER = {}

def worker_init(asts, per_expr_tags):
    # store ASTs in module-level for fork inheritance
    _WORKER['asts'] = asts
    _WORKER['tags'] = per_expr_tags


def eval_chunk(args):
    doc_start, doc_count = args
    asts = _WORKER['asts']
    per_expr_tags = _WORKER['tags']

    bitmap_section_off = HEADER_SIZE + DOC_NUM * VECTOR_DIM * 4
    bytes_per_row = TAG_STRIDE * 8

    match_counts = [0] * len(asts)
    with open(DATASET, 'rb') as f:
        f.seek(bitmap_section_off + doc_start * bytes_per_row)
        data = f.read(doc_count * bytes_per_row)
    if len(data) < doc_count * bytes_per_row:
        return doc_start, match_counts, 0
    bm = np.frombuffer(data, dtype=np.uint64).reshape(doc_count, TAG_STRIDE)

    for i, (ast, tags) in enumerate(zip(asts, per_expr_tags)):
        if ast is None:
            continue
        tag_bits = {}
        for t in tags:
            word = t // 64
            bit = t % 64
            col = bm[:, word]
            tag_bits[t] = ((col >> np.uint64(bit)) & np.uint64(1)).astype(bool)
        result = eval_ast(ast, tag_bits)
        match_counts[i] += int(result.sum())

    return doc_start, match_counts, doc_count


def main():
    print(f"=== selection rate calc ===")
    print(f"dataset: {DATASET}")
    print(f"exprs  : {EXPR_FILE}")

    with open(EXPR_FILE) as f:
        lines = [l.strip() for l in f if l.strip()]
    print(f"loaded {len(lines)} expressions")

    asts = []
    for i, l in enumerate(lines):
        try:
            asts.append(parse_expr(l))
        except Exception as e:
            print(f"  parse fail expr {i}: {e}")
            asts.append(None)

    per_expr_tags = []
    for ast in asts:
        if ast is None:
            per_expr_tags.append(set())
            continue
        s = set()
        collect_tags(ast, s)
        per_expr_tags.append(s)

    print(f"\nexpr stats:")
    for i, (ast, tags) in enumerate(zip(asts, per_expr_tags)):
        if ast is None:
            print(f"  [{i}] PARSE FAIL")
            continue
        leaves = count_leaves(ast)
        print(f"  [{i}] leaves={leaves:4d}  unique_tags={len(tags):4d}")

    chunks = []
    for start in range(0, DOC_NUM, CHUNK_DOCS):
        n = min(CHUNK_DOCS, DOC_NUM - start)
        chunks.append((start, n))

    print(f"\nparallel scan: {len(chunks)} chunks x {CHUNK_DOCS} docs")
    num_workers = min(mp.cpu_count(), 64)
    t0 = time.time()
    match_counts = [0] * len(asts)
    total_docs = 0

    with mp.Pool(num_workers, initializer=worker_init, initargs=(asts, per_expr_tags)) as pool:
        done = 0
        for doc_start, mc, n in pool.imap_unordered(eval_chunk, chunks):
            for i in range(len(asts)):
                match_counts[i] += mc[i]
            total_docs += n
            done += 1
            if done % 20 == 0 or done == len(chunks):
                elapsed = time.time() - t0
                rate = total_docs / elapsed if elapsed > 0 else 0
                eta = (DOC_NUM - total_docs) / rate if rate > 0 else 0
                print(f"  [{done}/{len(chunks)}] docs={total_docs//1_000_000}M/{DOC_NUM//1_000_000}M "
                      f"({rate/1e3:.1f}K docs/s, ETA {eta:.0f}s)", flush=True)

    print(f"\ntotal scanned: {total_docs} docs in {time.time()-t0:.1f}s")

    print(f"\n=== selection rates (n={total_docs}) ===")
    print(f"{'idx':>4} {'match':>10} {'sel_rate':>14} {'leaves':>7} {'uniq_tags':>10}")
    for i, (ast, tags, mc) in enumerate(zip(asts, per_expr_tags, match_counts)):
        if ast is None:
            print(f"{i:>4}  PARSE FAIL")
            continue
        leaves = count_leaves(ast)
        rate = mc / total_docs
        print(f"{i:>4} {mc:>10} {rate:>14.8f} {leaves:>7} {len(tags):>10}")


if __name__ == '__main__':
    main()
