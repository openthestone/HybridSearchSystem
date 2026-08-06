#!/usr/bin/env python3
"""Round-trip verify restore_section.py output.

Reads:
  - test/restored/relevance_softLtrAfmConditionv1/section.relevance_softLtrAfmConditionv1.0
  - test/restored/inverted_union/section.inverted_union.0
  - original /root/sks_hw/dataset_HW.bin (subset M)

Mirrors HybridSearchSystem DataReader.h:2162 BuildDatasetAndTagMapCachesFromRaw:
  pass A: read vector file → gid_to_lid, vectors[FP32]
  pass B (1st attr): scan inverted_union files → tag_map (term str → tag_id)
  pass B (2nd attr): re-scan → fill bitmap[lid][tag_id bit]

Then compare to original:
  - header (doc_num, dim, tag_num)
  - vector bytes bit-exact
  - bitmap bytes bit-exact (after canonical tag_id assignment)
"""
import argparse
import struct
import sys
from pathlib import Path

import doc_pb2


def read_records(path):
    """Yield (gid, Section) per record. Mirrors ReadRawRecordFile."""
    with open(path, 'rb') as f:
        count = struct.unpack('<I', f.read(4))[0]
        for i in range(count):
            rec_len = struct.unpack('<I', f.read(4))[0]
            gid = struct.unpack('<Q', f.read(8))[0]
            payload_len = rec_len - 8
            payload = f.read(payload_len)
            if len(payload) != payload_len:
                raise ValueError(f'short payload at record {i}')
            sec = doc_pb2.Section()
            sec.ParseFromString(payload)
            yield gid, sec


def restore_dataset(vec_file, inv_file, expected_dim):
    """Returns (vectors_flat, bitmaps_packed, tag_map).
    bitmaps_packed: list of bytearrays, each ceil(tag_num/8) bytes."""
    vectors = []
    gid_to_lid = {}
    dim = None

    # Pass A: vectors
    n_vec = 0
    for gid, sec in read_records(vec_file):
        if not sec.float_embedding:
            raise ValueError(f'gid {gid} has no float_embedding')
        emb = list(sec.float_embedding[0].embedding)
        if dim is None:
            dim = len(emb)
            if dim != expected_dim:
                raise ValueError(f'dim mismatch: {dim} vs expected {expected_dim}')
        if len(emb) != dim:
            raise ValueError(f'gid {gid} dim {len(emb)} != {dim}')
        lid = len(vectors) // dim
        if gid != lid:
            print(f'WARN: gid {gid} != lid {lid} (filled by position)')
        gid_to_lid[gid] = lid
        vectors.extend(emb)
        n_vec += 1
    print(f'vector pass: docs={n_vec} dim={dim}')

    # Pass B1: collect tag map (sorted tag name → tag_id, deterministic)
    tag_set = set()
    n_inv = 0
    for gid, sec in read_records(inv_file):
        for ti in sec.termInfos:
            tag_set.add(ti.term)
        n_inv += 1
    print(f'inverted pass1: records={n_inv} unique tags={len(tag_set)}')

    tag_map = {}
    # Sort by numeric value extracted from "tag_<n>#1" so canonical id == original id.
    def sort_key(name):
        prefix = 'tag_'
        suffix = '#1'
        if name.startswith(prefix) and name.endswith(suffix):
            mid = name[len(prefix):-len(suffix)]
            if mid.isdigit():
                return (0, int(mid))
        return (1, name)
    for tag_id, name in enumerate(sorted(tag_set, key=sort_key)):
        tag_map[name] = tag_id

    # Pass B2: fill bitmaps
    stride = (len(tag_map) + 7) // 8
    bitmaps = [bytearray(stride) for _ in range(n_vec)]

    n_inv2 = 0
    for gid, sec in read_records(inv_file):
        lid = gid_to_lid.get(gid)
        if lid is None:
            print(f'WARN: gid {gid} in inverted but not vector')
            continue
        for ti in sec.termInfos:
            tag_id = tag_map[ti.term]
            byte_off = tag_id >> 3
            bit = 1 << (tag_id & 7)
            bitmaps[lid][byte_off] |= bit
        n_inv2 += 1
    print(f'inverted pass2: records={n_inv2}')

    return vectors, bitmaps, tag_map


def parse_header(buf):
    magic = buf[0:8]
    ver = struct.unpack('<I', buf[8:12])[0]
    doc_num = struct.unpack('<Q', buf[16:24])[0]
    dim = struct.unpack('<I', buf[24:28])[0]
    tag_num = struct.unpack('<I', buf[28:32])[0]
    return ver, doc_num, dim, tag_num


def compare(args):
    # Original subset
    with open(args.dataset, 'rb') as f:
        header = f.read(40)
        ver, doc_total, dim, tag_num = parse_header(header)
        M = doc_total if args.doc_subset == 0 else min(args.doc_subset, doc_total)
        stride_u64 = (tag_num + 63) // 64
        vec_bytes_orig = f.read(M * dim * 4)
        bmp_bytes_orig = f.read(M * stride_u64 * 8)
    print(f'original: M={M} dim={dim} tag_num={tag_num} stride_u64={stride_u64}')
    print(f'  orig vector bytes={len(vec_bytes_orig)} bitmap bytes={len(bmp_bytes_orig)}')

    # Restored
    vectors, bitmaps, tag_map = restore_dataset(args.vec_file, args.inv_file, dim)

    if len(vectors) // dim != M:
        sys.exit(f'doc count mismatch: orig {M} vs restored {len(vectors)//dim}')

    # --- vector compare (bit-exact expected) ---
    vec_mism = 0
    for i in range(M * dim):
        o = struct.unpack_from('<f', vec_bytes_orig, i * 4)[0]
        r = vectors[i]
        if o != r:
            vec_mism += 1
            if vec_mism <= 3:
                print(f'  vec diff at i={i}: orig={o} restored={r}')
    print(f'vector diffs: {vec_mism} / {M*dim}')

    # --- bitmap compare (canonical tag_id mapping) ---
    # Original bitmap ordering is bit position = tag_id (0..tag_num-1).
    # Restored tag_id is sorted-by-name. Since restore_section named tags
    # "tag_<orig_id>", sort by name == sort by numeric orig_id == identity.
    # So restored tag_id should equal original tag_id.
    bmp_mism_docs = 0
    bmp_mism_bits = 0
    for d in range(M):
        # Original bitmap: u64 stride little-endian → bytes little-endian
        orig_u64 = struct.unpack_from(f'<{stride_u64}Q', bmp_bytes_orig, d * stride_u64 * 8)
        orig_bytes = bytearray()
        for w in orig_u64:
            orig_bytes += struct.pack('<Q', w)
        # Restored bitmap: stride = ceil(tag_num/8) bytes, bit i = tag i
        rest_bytes = bitmaps[d]
        # Compare up to min length
        n = min(len(orig_bytes), len(rest_bytes))
        for b in range(n):
            if orig_bytes[b] != rest_bytes[b]:
                bmp_mism_docs += 1
                x = orig_bytes[b] ^ rest_bytes[b]
                bmp_mism_bits += bin(x).count('1')
        # tail bytes: count if any extra set in longer side
        if len(orig_bytes) != len(rest_bytes):
            tail = orig_bytes[n:] + rest_bytes[n:]
            for b in tail:
                if b != 0:
                    bmp_mism_bits += bin(b).count('1')
                    bmp_mism_docs += 1
    print(f'bitmap diffs: {bmp_mism_docs} docs / {M}, total bit mismatches={bmp_mism_bits}')

    return vec_mism == 0 and bmp_mism_docs == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='/root/sks_hw/dataset_HW.bin')
    ap.add_argument('--restored_dir', default='/root/sks_hw/test/restored')
    ap.add_argument('--doc_subset', type=int, default=1024)
    ap.add_argument('--query_file', default='/root/sks_hw/test/restored/QueryData_restored.txt')
    args = ap.parse_args()

    args.vec_file = str(Path(args.restored_dir) /
                        'relevance_softLtrAfmConditionv1/section.relevance_softLtrAfmConditionv1.0')
    args.inv_file = str(Path(args.restored_dir) /
                        'inverted_union/section.inverted_union.0')

    ok = compare(args)
    print()
    print('RESULT:', 'PASS — bit-exact round-trip' if ok else 'FAIL — see diffs above')

    # Phase 2: simulate HybridSearchSystem query→tag_map lookup.
    if ok and Path(args.query_file).exists():
        # Rebuild tag_map from restored inverted_union (same path as compare()).
        _, _, tag_map = restore_dataset(args.vec_file, args.inv_file, 64)
        nq, hits, misses = simulate_query_lookup(tag_map, args.query_file)
        print(f'\nQuery lookup simulation (HybridSearchSystem ParseSyntaxFilterToExpr):')
        print(f'  queries       : {nq}')
        print(f'  term hits     : {hits}')
        print(f'  term misses   : {misses}')
        q_ok = misses == 0 and hits > 0
        print('QUERY RESULT:', 'PASS — all term lookups hit tag_map' if q_ok
              else 'FAIL — see misses above')
        ok = ok and q_ok

    sys.exit(0 if ok else 1)


def simulate_query_lookup(tag_map, query_file):
    """Mirror HybridSearchSystem ParseSyntaxFilterToExpr tag lookup.

    For each query line, walk the syntax_filter tree. Each leaf is
    {"terms": {"<section>": [vals]}}. Lookup key = "<sec>#<val>".
    Count hits/misses against tag_map.
    """
    import json
    hits = 0
    misses = 0
    n_queries = 0

    def walk(node):
        nonlocal hits, misses
        if isinstance(node, dict):
            for k, v in node.items():
                if k == 'terms' and isinstance(v, dict):
                    for sec, vals in v.items():
                        if not isinstance(vals, list):
                            vals = [vals]
                        for val in vals:
                            key = f'{sec}#{val}'
                            if key in tag_map:
                                hits += 1
                            else:
                                misses += 1
                else:
                    walk(v)
        elif isinstance(node, list):
            for c in node:
                walk(c)

    with open(query_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            n_queries += 1
            mt = obj.get('main_tier', {})
            jq = mt.get('json_query', '')
            if isinstance(jq, str):
                jq = json.loads(jq)
            vec_node = jq.get('vector', {})
            sf = vec_node.get('syntax_filter', {})
            walk(sf)
    return n_queries, hits, misses


if __name__ == '__main__':
    main()


if __name__ == '__main__':
    main()
