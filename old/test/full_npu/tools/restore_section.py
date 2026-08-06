#!/usr/bin/env python3
"""Restore dataset_HW.bin → PaaS-style section.* raw files.

Inputs:
  - dataset_HW.bin (HYDSET2: 40B header + FP32 vectors + uint64 bitmaps)

Outputs (under --out_dir):
  - relevance_softLtrAfmConditionv1/section.relevance_softLtrAfmConditionv1.0
      Per doc: record[float_embedding only]
  - inverted_union/section.inverted_union.0
      Per doc: record[termInfos only, terms synthesized as "tag_<id>"]

Container format (matches HybridSearchSystem DataReader.h:2056 ReadRawRecordFile):
  uint32 count
  for each doc:
    uint32 len          # 8 + pb_payload_len
    uint64 gid          # = local id (sks_hw has no global gid)
    uint8  payload[len-8]  # serialized Section protobuf

Section schema (Falcon.IndexFactory.Section, see doc.proto):
  termInfos=1, int32_embedding=2, float_embedding=3, ranges=4

Lossy: original tag→term string mapping lost (sks_hw bitmap is random).
Synthesize "tag_<id>" for each set bit.
"""
import argparse
import struct
import sys
from pathlib import Path

import doc_pb2


HYDSET2_HEADER = 40  # bytes


def parse_header(buf):
    magic = buf[0:8]
    if magic != b'HYDSET2\x00':
        raise ValueError(f'bad magic: {magic!r}')
    ver = struct.unpack('<I', buf[8:12])[0]
    doc_num = struct.unpack('<Q', buf[16:24])[0]
    dim = struct.unpack('<I', buf[24:28])[0]
    tag_num = struct.unpack('<I', buf[28:32])[0]
    return ver, doc_num, dim, tag_num


def write_record(f, gid, payload_bytes):
    """Emit one raw record: uint32 len, uint64 gid, payload[len-8]."""
    rec_len = 8 + len(payload_bytes)
    f.write(struct.pack('<I', rec_len))
    f.write(struct.pack('<Q', gid))
    f.write(payload_bytes)


def restore(dataset_path, out_dir, doc_subset):
    out_dir = Path(out_dir)
    vec_dir = out_dir / 'relevance_softLtrAfmConditionv1'
    inv_dir = out_dir / 'inverted_union'
    vec_dir.mkdir(parents=True, exist_ok=True)
    inv_dir.mkdir(parents=True, exist_ok=True)

    with open(dataset_path, 'rb') as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(0)
        header = f.read(HYDSET2_HEADER)
        ver, doc_num_total, dim, tag_num = parse_header(header)
        M = doc_num_total if doc_subset == 0 else min(doc_subset, doc_num_total)
        stride = (tag_num + 63) // 64  # u64 words per doc bitmap
        vec_bytes = M * dim * 4
        bmp_bytes = M * stride * 8
        print(f'header: ver={ver} doc_total={doc_num_total} dim={dim} tag_num={tag_num} stride_u64={stride}')
        print(f'subset M={M}, vector bytes={vec_bytes}, bitmap bytes={bmp_bytes}')
        vectors = f.read(vec_bytes)
        bitmaps = f.read(bmp_bytes)

    if len(vectors) != vec_bytes or len(bitmaps) != bmp_bytes:
        raise RuntimeError(f'short read: vec={len(vectors)}/{vec_bytes} bmp={len(bitmaps)}/{bmp_bytes}')

    vec_path = vec_dir / 'section.relevance_softLtrAfmConditionv1.0'
    inv_path = inv_dir / 'section.inverted_union.0'

    written_terms = 0
    with open(vec_path, 'wb') as fvec, open(inv_path, 'wb') as finv:
        # record count header
        fvec.write(struct.pack('<I', M))
        finv.write(struct.pack('<I', M))

        for d in range(M):
            # --- vector record ---
            vsec = doc_pb2.Section()
            v = vsec.float_embedding.add()
            v.embedding.extend(struct.unpack_from(f'<{dim}f', vectors, d * dim * 4))
            v_payload = vsec.SerializeToString()

            write_record(fvec, d, v_payload)

            # --- inverted_union record (terms only) ---
            isec = doc_pb2.Section()
            bmp_off = d * stride * 8
            for w in range(stride):
                word = struct.unpack_from('<Q', bitmaps, bmp_off + w * 8)[0]
                if word == 0:
                    continue
                base_tag = w * 64
                while word:
                    b = (word & -word).bit_length() - 1
                    tag_id = base_tag + b
                    if tag_id >= tag_num:
                        break
                    ti = isec.termInfos.add()
                    ti.term = f'tag_{tag_id}#1'
                    written_terms += 1
                    word &= word - 1
            i_payload = isec.SerializeToString()
            write_record(finv, d, i_payload)

            if (d + 1) % 10000 == 0 or d + 1 == M:
                print(f'  progress: {d+1}/{M} docs, terms so far={written_terms}', end='\r')
    print()
    print(f'wrote {vec_path} ({vec_path.stat().st_size} bytes)')
    print(f'wrote {inv_path} ({inv_path.stat().st_size} bytes)')
    print(f'total terms emitted: {written_terms}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='/root/sks_hw/dataset_HW.bin')
    ap.add_argument('--out_dir', default='/root/sks_hw/test/restored')
    ap.add_argument('--doc_subset', type=int, default=1024,
                    help='0 = all; default 1024 for quick test')
    args = ap.parse_args()
    restore(args.dataset, args.out_dir, args.doc_subset)


if __name__ == '__main__':
    main()
