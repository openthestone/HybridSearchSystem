#!/usr/bin/env python3
"""
Convert the original project's QueryData_10000.txt into files consumed by
hx_npu/port's fr_search harness:

  queries.fvecs  - headered [int32 rows][int32 dim][rows * dim float32]
  filters.txt    - one boolean filter expression per query line
  topk.txt       - one top_k value per query line
  manifest.json  - conversion metadata

The parser intentionally mirrors include/utils/DataReader.h so the generated
filter expressions use the same tag ids as the original project.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class ExprResult:
    is_valid: bool = True
    is_const: bool = False
    value: bool = False
    expr: str = ""

    @staticmethod
    def const(value: bool) -> "ExprResult":
        return ExprResult(True, True, value, "")

    @staticmethod
    def expression(expr: str) -> "ExprResult":
        return ExprResult(True, False, False, expr)

    @staticmethod
    def invalid() -> "ExprResult":
        return ExprResult(False, False, False, "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert QueryData_10000.txt to fr_search query/filter/topk files."
    )
    parser.add_argument("--query", default="QueryData_10000.txt", help="Input QueryData text file")
    parser.add_argument("--tag-map", default="tag_map.bin", help="Original project's tag_map.bin")
    parser.add_argument("--dataset", default="dataset.bin", help="Optional HYDSET2 dataset.bin for dim/tag validation")
    parser.add_argument("--out-dir", default="converted_queries", help="Output directory")
    parser.add_argument("--limit", type=int, default=0, help="Convert at most N valid queries; 0 means all")
    parser.add_argument("--result-num-key", default="result_num")
    parser.add_argument("--main-tier-key", default="main_tier")
    parser.add_argument("--json-query-key", default="json_query")
    parser.add_argument("--vector-node-key", default="vector")
    parser.add_argument("--vector-value-key", default="relevance_learning2rank")
    parser.add_argument("--syntax-filter-key", default="syntax_filter")
    parser.add_argument("--strict", action="store_true", help="Fail on the first invalid query instead of skipping it")
    return parser.parse_args()


def read_exact_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return -1


def load_tag_map(path: Path) -> Dict[str, int]:
    data = path.read_bytes()
    for size_t_size in (8, 4):
        try:
            result, end = _load_tag_map_with_size_t(data, size_t_size)
            if end == len(data):
                return result
        except (struct.error, UnicodeDecodeError, ValueError):
            continue
    raise ValueError(f"failed to parse tag map cache: {path}")


def _load_tag_map_with_size_t(data: bytes, size_t_size: int) -> Tuple[Dict[str, int], int]:
    if len(data) < size_t_size:
        raise ValueError("tag map file too small")
    offset = 0
    if size_t_size == 8:
        map_size = struct.unpack_from("<Q", data, offset)[0]
    else:
        map_size = struct.unpack_from("<I", data, offset)[0]
    offset += size_t_size
    if map_size <= 0 or map_size > 10_000_000:
        raise ValueError("invalid map size")

    tag_map: Dict[str, int] = {}
    for _ in range(map_size):
        if offset + 4 > len(data):
            raise ValueError("truncated tag length")
        length = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        if length <= 0 or offset + length + 4 > len(data):
            raise ValueError("invalid tag entry")
        term = data[offset : offset + length].decode("utf-8")
        offset += length
        tag_id = struct.unpack_from("<i", data, offset)[0]
        offset += 4
        tag_map[term] = tag_id
    return tag_map, offset


def read_dataset_header(path: Path) -> Optional[Dict[str, int]]:
    if not path.exists():
        return None
    with path.open("rb") as f:
        header = f.read(40)
    if len(header) < 40 or header[:7] != b"HYDSET2":
        raise ValueError(f"{path} is not a HYDSET2 dataset cache")
    version, doc_num, dim, tag_num = struct.unpack_from("<I4xQII", header, 8)
    return {
        "version": int(version),
        "doc_num": int(doc_num),
        "vector_dim": int(dim),
        "tag_num": int(tag_num),
    }


def load_query_records(path: Path) -> List[Dict[str, Any]]:
    content = path.read_text(encoding="utf-8")
    stripped = content.strip()
    if not stripped:
        return []
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    records: List[Dict[str, Any]] = []
    for line_no, line in enumerate(content.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_no}: invalid JSON: {exc}") from exc
        if isinstance(item, dict):
            records.append(item)
    return records


def json_value_to_string(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format(value, ".6g")
    return ""


def is_logical_op(op: str) -> bool:
    return op.lower() in ("and", "or")


def combine_exprs(exprs: Iterable[ExprResult], op_type: str) -> ExprResult:
    low = op_type.lower()
    if low not in ("and", "or"):
        return ExprResult.invalid()
    is_and = low == "and"
    valid_exprs: List[str] = []

    for expr in exprs:
        if not expr.is_valid:
            return ExprResult.invalid()
        if expr.is_const:
            if is_and and not expr.value:
                return ExprResult.const(False)
            if not is_and and expr.value:
                return ExprResult.const(True)
            continue
        valid_exprs.append(expr.expr)

    if not valid_exprs:
        return ExprResult.const(True if is_and else False)
    if len(valid_exprs) == 1:
        return ExprResult.expression(valid_exprs[0])

    joiner = " AND " if is_and else " OR "
    return ExprResult.expression("(" + joiner.join(valid_exprs) + ")")


def parse_syntax_filter_to_expr(node: Any, tag_map: Dict[str, int]) -> ExprResult:
    if not isinstance(node, dict) or len(node) != 1:
        return ExprResult.invalid()

    key = next(iter(node.keys()))

    if key == "term":
        term_node = node["term"]
        if not isinstance(term_node, dict) or not term_node:
            return ExprResult.invalid()
        parts: List[ExprResult] = []
        for section, value in term_node.items():
            tag = f"{section}#{json_value_to_string(value)}"
            if tag.endswith("#"):
                return ExprResult.invalid()
            tag_id = tag_map.get(tag)
            parts.append(ExprResult.const(False) if tag_id is None else ExprResult.expression(str(tag_id)))
        return combine_exprs(parts, "and")

    if key == "terms":
        terms_node = node["terms"]
        if not isinstance(terms_node, dict):
            return ExprResult.invalid()

        join_type = terms_node.get("join_type", "or")
        inner_join = terms_node.get("inner_section_join_type", "or")
        if not isinstance(join_type, str) or not isinstance(inner_join, str):
            return ExprResult.invalid()
        if not is_logical_op(join_type) or not is_logical_op(inner_join):
            return ExprResult.invalid()

        section_exprs: List[ExprResult] = []
        for section, values in terms_node.items():
            if section in ("join_type", "inner_section_join_type"):
                continue
            inner_exprs: List[ExprResult] = []
            iter_values = values if isinstance(values, list) else [values]
            for value in iter_values:
                value_str = json_value_to_string(value)
                if not value_str:
                    return ExprResult.invalid()
                tag_id = tag_map.get(f"{section}#{value_str}")
                inner_exprs.append(ExprResult.const(False) if tag_id is None else ExprResult.expression(str(tag_id)))
            if not inner_exprs:
                return ExprResult.invalid()
            section_expr = combine_exprs(inner_exprs, inner_join)
            if not section_expr.is_valid:
                return ExprResult.invalid()
            section_exprs.append(section_expr)
        if not section_exprs:
            return ExprResult.invalid()
        return combine_exprs(section_exprs, join_type)

    if key in ("and", "or"):
        children = node[key]
        if not isinstance(children, list):
            return ExprResult.invalid()
        return combine_exprs((parse_syntax_filter_to_expr(child, tag_map) for child in children), key)

    if key == "not":
        child = parse_syntax_filter_to_expr(node["not"], tag_map)
        if not child.is_valid:
            return ExprResult.invalid()
        if child.is_const:
            return ExprResult.const(not child.value)
        return ExprResult.expression("(NOT " + child.expr + ")")

    return ExprResult.invalid()


def finalize_filter_expr(result: ExprResult) -> str:
    if result.is_const:
        return "" if result.value else "0 AND (NOT 0)"
    return result.expr


def parse_one_query(
    root: Dict[str, Any],
    line_no: int,
    tag_map: Dict[str, int],
    args: argparse.Namespace,
    expected_dim: Optional[int],
) -> Tuple[List[float], str, int]:
    result_num = root.get(args.result_num_key)
    if not isinstance(result_num, int) or isinstance(result_num, bool) or result_num <= 0:
        raise ValueError(f"line {line_no}: missing or invalid {args.result_num_key}")

    main_tier = root.get(args.main_tier_key)
    if not isinstance(main_tier, dict):
        raise ValueError(f"line {line_no}: missing {args.main_tier_key} object")

    json_query = main_tier.get(args.json_query_key)
    if not isinstance(json_query, str):
        raise ValueError(f"line {line_no}: missing {args.main_tier_key}.{args.json_query_key}")
    try:
        inner = json.loads(json_query)
    except json.JSONDecodeError as exc:
        raise ValueError(f"line {line_no}: invalid inner json_query: {exc}") from exc

    vector_node = inner.get(args.vector_node_key)
    if not isinstance(vector_node, dict):
        raise ValueError(f"line {line_no}: missing vector node")
    vec_json = vector_node.get(args.vector_value_key)
    if not isinstance(vec_json, list):
        raise ValueError(f"line {line_no}: missing vector values")
    try:
        vector = [float(v) for v in vec_json]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"line {line_no}: vector contains non-numeric values") from exc

    if expected_dim is not None and len(vector) != expected_dim:
        raise ValueError(f"line {line_no}: vector dim {len(vector)} != expected dim {expected_dim}")

    filter_expr = ""
    if args.syntax_filter_key in vector_node:
        parsed_filter = parse_syntax_filter_to_expr(vector_node[args.syntax_filter_key], tag_map)
        if not parsed_filter.is_valid:
            raise ValueError(f"line {line_no}: invalid syntax_filter")
        filter_expr = finalize_filter_expr(parsed_filter)

    return vector, filter_expr, int(result_num)


def write_headered_fvecs(path: Path, vectors: List[List[float]]) -> None:
    if not vectors:
        raise ValueError("no vectors to write")
    dim = len(vectors[0])
    with path.open("wb") as out:
        out.write(struct.pack("<ii", len(vectors), dim))
        for vector in vectors:
            out.write(struct.pack("<" + "f" * dim, *vector))


def write_lines(path: Path, lines: Iterable[Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as out:
        for item in lines:
            out.write(str(item))
            out.write("\n")


def main() -> int:
    args = parse_args()
    query_path = Path(args.query)
    tag_map_path = Path(args.tag_map)
    dataset_path = Path(args.dataset)
    out_dir = Path(args.out_dir)

    if not query_path.exists():
        print(f"[convert_querydata] missing query file: {query_path}", file=sys.stderr)
        return 2
    if not tag_map_path.exists():
        print(f"[convert_querydata] missing tag map: {tag_map_path}", file=sys.stderr)
        return 2

    print(f"[convert_querydata] loading tag map: {tag_map_path}")
    tag_map = load_tag_map(tag_map_path)
    dataset_header = read_dataset_header(dataset_path) if dataset_path.exists() else None
    expected_dim = dataset_header["vector_dim"] if dataset_header else None

    print(f"[convert_querydata] loading queries: {query_path}")
    records = load_query_records(query_path)
    vectors: List[List[float]] = []
    filters: List[str] = []
    topks: List[int] = []
    warnings: List[str] = []
    inferred_dim: Optional[int] = expected_dim

    for line_no, record in enumerate(records, 1):
        if args.limit > 0 and len(vectors) >= args.limit:
            break
        try:
            vector, filter_expr, top_k = parse_one_query(record, line_no, tag_map, args, inferred_dim)
            if inferred_dim is None:
                inferred_dim = len(vector)
            vectors.append(vector)
            filters.append(filter_expr)
            topks.append(top_k)
        except ValueError as exc:
            if args.strict:
                raise
            warnings.append(str(exc))

    if not vectors:
        print("[convert_querydata] no valid queries converted", file=sys.stderr)
        for warning in warnings[:20]:
            print(f"[convert_querydata] skipped: {warning}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    query_out = out_dir / "queries.fvecs"
    filter_out = out_dir / "filters.txt"
    topk_out = out_dir / "topk.txt"
    manifest_out = out_dir / "manifest.json"

    write_headered_fvecs(query_out, vectors)
    write_lines(filter_out, filters)
    write_lines(topk_out, topks)

    manifest = {
        "source_query": str(query_path),
        "source_tag_map": str(tag_map_path),
        "source_dataset": str(dataset_path) if dataset_path.exists() else None,
        "query_count": len(vectors),
        "vector_dim": len(vectors[0]),
        "tag_map_size": len(tag_map),
        "skipped_query_count": len(warnings),
        "topk_min": min(topks),
        "topk_max": max(topks),
        "outputs": {
            "queries_fvecs": str(query_out),
            "filters_txt": str(filter_out),
            "topk_txt": str(topk_out),
        },
    }
    if dataset_header:
        manifest["dataset_header"] = dataset_header
    manifest_out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        "[convert_querydata] done: "
        f"queries={len(vectors)}, dim={len(vectors[0])}, "
        f"topk=[{min(topks)}, {max(topks)}], skipped={len(warnings)}"
    )
    if warnings:
        warn_path = out_dir / "skipped_warnings.txt"
        write_lines(warn_path, warnings)
        print(f"[convert_querydata] skipped warnings written to {warn_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # keep shell output concise but actionable
        print(f"[convert_querydata] error: {exc}", file=sys.stderr)
        raise SystemExit(1)
