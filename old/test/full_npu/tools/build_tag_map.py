#!/usr/bin/env python3
"""Synthesize HS3 tag_map.bin matching dataset_HW.bin bitmap bit layout.

dataset_HW.bin bitmap bit i = doc has tag i (no original term strings).
We need tag_map such that HS3's query parser lookup `<sec>#<val>` hits.

For each tag_id i in [0, tag_num):
  term string = "tag_<i>#1"   (matches restore_falcon.py leaf format)
  id          = i              (matches bitmap bit position)

Format (DataReader.h:2007 SaveTagMapCache):
  size_t map_size
  per entry:
    uint32_t len
    char term[len]
    int id
"""
import argparse
import struct
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/root/dqj/HybridSearchSystem/tag_map.bin')
    ap.add_argument('--tag_num', type=int, default=35840)
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'wb') as f:
        f.write(struct.pack('<Q', args.tag_num))
        for i in range(args.tag_num):
            term = f'tag_{i}#1'.encode('ascii')
            f.write(struct.pack('<I', len(term)))
            f.write(term)
            f.write(struct.pack('<i', i))
    print(f'wrote {args.out} ({Path(args.out).stat().st_size} bytes, {args.tag_num} entries)')


if __name__ == '__main__':
    main()
