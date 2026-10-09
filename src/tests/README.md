# port/tests — host-side unit tests

Pure C++17 tests for the correctness-critical, NPU-independent code the port
layer adds. No CANN / protobuf / NPU needed, so they run on the local dev box
**and** the server.

```bash
bash port/tests/run_tests.sh          # builds + runs all; exits non-zero on failure
```

## What's covered

| test | target | what it pins |
|---|---|---|
| `test_filter_expr` | [filter_expr.h](../harness/query/filter_expr.h) | parser precedence (NOT>AND>OR), parens, n-ary AND/OR, `NOT` of groups, empty=match-all, malformed→parse-error; and that the tag-set evaluator and the bitmap evaluator **always agree** (incl. boundary tags 0 / 63 / 64 / 35839). |
| `test_query_io` | [query_io.h](../harness/query/query_io.h) | `.fvecs` loader: headered `[i32 rows][i32 dim][floats]` (hw_queries.fvecs) vs per-vector standard fvecs; dim=64; missing/truncated/too-small files don't crash. |
| `test_record_io` | [record_io.h](../common/record_io.h) | the builder-input record envelope round-trips through new/'s **actual** `TianjiEngine::ReadAndDoTask` (compiled from `new/build/file/`): gdocid (incl. > 2³²) and payload bytes (incl. empty + binary/NUL) come back exactly. This locks the `fr_converter` → `fr_builder` on-disk contract to the real reader. |

## Notes / limits

- These test the **host logic** (parsing, file formats, envelope). The NPU path
  (MMad scoring, bitmap filter, TopK, aggregator) and the protobuf `Section`
  encoding are validated on the server via the end-to-end run + the CPU
  brute-force recall self-check in `fr_search` (see [../README.md](../README.md)).
- The filter **semantics caveat** (new/ drops a term whose token is absent from
  the index; the CPU reference treats an absent tag as false — they agree only on
  a full-corpus index) is documented in `filter_expr.h` and the top-level TODO.
