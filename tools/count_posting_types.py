#!/usr/bin/env python3
"""
Count BIT_SET vs BIT_LIST postings in a built index.

A built inverted section file is:
  <index>/<field>/poissonengine_<version>_<segId>_<field>.posting
laid out as:
  header : version(u8) + reserved(5 bytes) + extLen(u16) + ext(extLen bytes)
  tokenNum(u32)
  token dict : tokenNum x (tokenId u64, postingOffset u32, tokenOffset u32)   # offsets absolute from file start
  postings   : each token's posting starts at its postingOffset with a u32 head
               head = (type << 28) | (cardinality & 0x0FFFFFFF)  -> type = head >> 28
PostingType: 0=BIT_SET 1=BIT_LIST 2=BYTE_SET 3=INT_LIST

Usage:
  count_posting_types.py <index_dir> [field=content]
"""
import sys, os, glob, struct
from collections import Counter

TYPE_NAMES = {0: "BIT_SET", 1: "BIT_LIST", 2: "BYTE_SET", 3: "INT_LIST", 4: "UNKNOWN"}


def parse_file(path):
    with open(path, "rb") as f:
        data = f.read()
    off = 0
    off += 1  # version u8
    off += 5  # reserved 5 bytes (WriteHeader)
    ext_len = struct.unpack_from("<H", data, off)[0]; off += 2
    off += ext_len
    token_num = struct.unpack_from("<I", data, off)[0]; off += 4
    counts = Counter()
    for _ in range(token_num):
        _tok, posting_off, _tok_off = struct.unpack_from("<QII", data, off); off += 16
        head = struct.unpack_from("<I", data, posting_off)[0]
        counts[head >> 28] += 1
    return counts, token_num


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    idx = sys.argv[1]
    field = sys.argv[2] if len(sys.argv) > 2 else "content"
    pat = os.path.join(idx, field, "poissonengine_*_*_%s.posting" % field)
    files = sorted(glob.glob(pat))
    if not files:
        print("no posting files at", pat); sys.exit(1)
    total = Counter()
    for fp in files:
        c, _ = parse_file(fp)
        total += c
    grand = sum(total.values())
    print("index:", idx)
    print("segments (posting files):", len(files))
    print("total (tag,segment) postings:", grand)
    for t in sorted(total):
        print("  %-9s %14d  %6.2f%%" % (TYPE_NAMES.get(t, t), total[t], 100.0 * total[t] / grand))
    bs, bl = total.get(0, 0), total.get(1, 0)
    if bs + bl:
        print("bitset:bitlist ratio = %.1f : 1" % (bs / bl if bl else float("inf")))


if __name__ == "__main__":
    main()
