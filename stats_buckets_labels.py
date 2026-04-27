#!/usr/bin/env python3
import argparse
import os
import re
import struct
from typing import Dict, List, Set, Tuple


BUCKET_FILE_RE = re.compile(r"^bucket(\d+)_doc_ids\.bin$")


def read_bucket_doc_ids_bin(path: str) -> Tuple[int, List[int]]:
    with open(path, "rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise ValueError(f"文件头长度不足: {path}")
        (n_doc_ids,) = struct.unpack("<q", header)
        if n_doc_ids < 0:
            raise ValueError(f"非法文档数: {n_doc_ids}, file={path}")

        body = f.read()
        expected_size = n_doc_ids * 4
        if len(body) != expected_size:
            raise ValueError(
                f"文件体长度不匹配: file={path}, 期望 {expected_size} 字节, 实际 {len(body)} 字节"
            )

        doc_ids = list(struct.unpack(f"<{n_doc_ids}i", body)) if n_doc_ids > 0 else []
    return n_doc_ids, doc_ids


def read_buckets_dir(path: str) -> Tuple[Dict[int, int], int, Set[int]]:
    bucket_counts: Dict[int, int] = {}
    total_assignments = 0
    unique_docs: Set[int] = set()

    for name in os.listdir(path):
        m = BUCKET_FILE_RE.match(name)
        if not m:
            continue

        bucket_id = int(m.group(1))
        file_path = os.path.join(path, name)
        n_doc_ids, doc_ids = read_bucket_doc_ids_bin(file_path)
        bucket_counts[bucket_id] = n_doc_ids
        total_assignments += n_doc_ids
        unique_docs.update(doc_ids)

    return bucket_counts, total_assignments, unique_docs


def main():
    parser = argparse.ArgumentParser(
        description="读取 buckets/bucket*_doc_ids.bin 并统计每个桶文档数"
    )
    parser.add_argument(
        "--input",
        default="buckets",
        help="输入目录路径（默认: buckets）",
    )
    parser.add_argument(
        "--sort",
        choices=["bucket", "count_desc"],
        default="bucket",
        help="输出排序方式：按桶号或按文档数降序（默认: bucket）",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"未找到目录: {args.input}")
    if not os.path.isdir(args.input):
        raise NotADirectoryError(f"输入不是目录: {args.input}")

    counter, total_assignments, unique_docs = read_buckets_dir(args.input)

    if args.sort == "bucket":
        items = sorted(counter.items(), key=lambda x: x[0])
    else:
        items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    for bucket_id, count in items:
        print(f"bucket {bucket_id}: {count}")

    total_buckets = len(counter)
    total_docs = total_assignments
    unique_doc_count = len(unique_docs)
    min_docs = min(counter.values()) if counter else 0
    max_docs = max(counter.values()) if counter else 0

    print("---")
    print(f"总桶数量: {total_buckets}")
    print(f"总文档数量: {total_docs}")
    print(f"唯一文档数量: {unique_doc_count}")
    print(f"最小桶文档数: {min_docs}")
    print(f"最大桶文档数: {max_docs}")


if __name__ == "__main__":
    main()
