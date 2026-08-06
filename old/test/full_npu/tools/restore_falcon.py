#!/usr/bin/env python3
"""Restore hw_queries.fvecs + filter_expr_600.txt → PaaS falcon JSON.

Inputs:
  - hw_queries.fvecs: uint32 n_queries, uint32 dim, float[n_queries*dim]
  - filter_expr_600.txt: 10 boolean expr lines
      Grammar:
        expr   := or_expr
        or_expr := and_expr (OR and_expr)*
        and_expr := unary (AND unary)*
        unary  := NOT unary | atom
        atom   := '(' expr ')' | INTEGER

Output (QueryData_5.txt style, JSON-per-line):
  Each line is one retrieval request:
    corpus, result_num, main_tier.json_query (with vector.relevance_learning2rank +
    syntax_filter), score_plugin/score_query, parallel_retrieve, aggregate_*,
    timeout, request_id.

sks_hw expands 10 exprs to N queries by `query_idx % 10`.
"""
import argparse
import json
import re
import struct
import sys
from pathlib import Path


class ExprParser:
    """Recursive-descent parser → JSON IR tree.

    IR node forms (match QueryData_5.txt syntax_filter schema):
      {"and": [c1, c2, ...]}
      {"or":  [c1, c2, ...]}
      {"not": c}
      {"terms": {str(tag_id): ["delivery_all"]}}
    """
    TOKEN_RE = re.compile(r'\(|\)|AND|OR|NOT|\d+')

    def __init__(self, text):
        self.tokens = self.TOKEN_RE.findall(text)
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def next(self):
        t = self.tokens[self.pos]
        self.pos += 1
        return t

    def parse(self):
        node = self.parse_or()
        if self.pos != len(self.tokens):
            raise ValueError(f'trailing tokens at {self.pos}: {self.tokens[self.pos:]}')
        return node

    def parse_or(self):
        children = [self.parse_and()]
        while self.peek() == 'OR':
            self.next()
            children.append(self.parse_and())
        return children[0] if len(children) == 1 else {'or': children}

    def parse_and(self):
        children = [self.parse_unary()]
        while self.peek() == 'AND':
            self.next()
            children.append(self.parse_unary())
        return children[0] if len(children) == 1 else {'and': children}

    def parse_unary(self):
        if self.peek() == 'NOT':
            self.next()
            return {'not': self.parse_unary()}
        return self.parse_atom()

    def parse_atom(self):
        t = self.next()
        if t == '(':
            node = self.parse_or()
            if self.next() != ')':
                raise ValueError('expected )')
            return node
        if t.isdigit():
            # term name on dataset side is "tag_<id>#1"; here section = "tag_<id>", value = "1".
            return {'terms': {f'tag_{t}': ['1']}}
        raise ValueError(f'unexpected token: {t}')


def load_queries(fvecs_path):
    with open(fvecs_path, 'rb') as f:
        data = f.read()
    n, dim = struct.unpack('<II', data[:8])
    expected = 8 + n * dim * 4
    if len(data) != expected:
        raise ValueError(f'{fvecs_path}: size {len(data)} != expected {expected}')
    floats = struct.unpack(f'<{n * dim}f', data[8:])
    return n, dim, floats


def load_exprs(expr_path):
    exprs = []
    with open(expr_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            exprs.append(ExprParser(line).parse())
    return exprs


def build_request(query_idx, vector_floats, dim, expr_tree, result_num=3500):
    vec = vector_floats[query_idx * dim:(query_idx + 1) * dim]
    # Inner json_query string (will be re-parsed by HybridSearchSystem).
    # Real PaaS layout puts ALL query params (vector value, cluster_num,
    # syntax_filter, ...) inside the "vector" node — see QueryData_5.txt.
    inner = {
        'vector': {
            'relevance_learning2rank': list(vec),
            'record_name': str(query_idx),
            'cluster_num': 64,
            'max_cluster_num': 256,
            'result_num': result_num,
            'max_distance_score_doc_num': result_num,
            'parallel_score': True,
            'syntax_filter': expr_tree,
        },
    }
    req = {
        'corpus': f'relevance_learning2rank-restored',
        'result_num': result_num,
        'main_tier': {'json_query': json.dumps(inner, separators=(',', ':'), ensure_ascii=False)},
        'score_plugin': 'ScorePlugin.SoftLTR',
        'score_query': '{}',
        'parallel_retrieve': True,
        'aggregate_plugin': 'AggregatePlugin.TopK',
        'aggregate_query': json.dumps({'top_k': result_num}, separators=(',', ':')),
        'timeout': 1000,
        'request_id': f'restored-{query_idx:06d}',
    }
    return req


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queries', default='/root/sks_hw/datasets/hw_queries.fvecs')
    ap.add_argument('--exprs', default='/root/sks_hw/filter_expr_600.txt')
    ap.add_argument('--out', default='/root/sks_hw/test/restored/QueryData_restored.txt')
    ap.add_argument('--n_queries', type=int, default=0,
                    help='0 = all in fvecs file')
    ap.add_argument('--result_num', type=int, default=3500)
    args = ap.parse_args()

    n, dim, vecs = load_queries(args.queries)
    exprs = load_exprs(args.exprs)
    if not exprs:
        sys.exit('no exprs parsed')
    print(f'queries={n} dim={dim} exprs={len(exprs)}')

    M = n if args.n_queries == 0 else min(args.n_queries, n)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        for q in range(M):
            expr = exprs[q % len(exprs)]
            req = build_request(q, vecs, dim, expr, args.result_num)
            f.write(json.dumps(req, separators=(',', ':'), ensure_ascii=False))
            f.write('\n')
            if (q + 1) % 1000 == 0 or q + 1 == M:
                print(f'  progress: {q+1}/{M}', end='\r')
    print()
    print(f'wrote {args.out} ({Path(args.out).stat().st_size} bytes)')


if __name__ == '__main__':
    main()
