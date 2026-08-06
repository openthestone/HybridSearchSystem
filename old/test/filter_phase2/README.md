# filter_phase2 — Phase 2 AICore forward-bitmap filter testbed

Skeleton. Filled during Phase 2 §5.2 implementation
(see `纯NPU路径优化草案.md §5.2`).

## Why separate from `filter_cmp`

`filter_cmp` tests Tianji-inverted kernel correctness at synthetic shape
(100K docs × 1024 tags × 5% density). Real sks_hw data is fundamentally
different — absolute numbers from `filter_cmp` do **not** extrapolate
(see `filter_cmp/README.md` disclaimer table for the 7-axis mismatch).

This testbed validates the **original** Phase 2 AICore forward-bitmap kernel
design (草案 §5.2.2, 32 docs/block, GatherMask compact, BucketPlan UB layout)
against production data shape.

## Real sks_hw data shape (target)

From `真实数据测试结果-L2桶信息.md` and `纯NPU路径优化草案.md §3.4`:

| Dimension | Value |
|---|---|
| Tag count | 35672 (stride = ceil(35672/64) = **558 uint64 word**) |
| Docs per L2 bucket | ≤1024 (multiple of 16 for NPU alignment) |
| Buckets per query | P50 ~96, P99+ mean **1454** (max 1601) |
| Density (filter selectivity) | P50 ~0.38, P99+ mean **0.027** (~0.005%-0.05% raw bit density) |
| Expr leaf count | business ≤1000 tags |
| RPN depth | ≤3000 entries (post NOT expansion) |
| BucketPlan node count | ≤2500 (节点数 > 2500 host fallback AI CPU) |

## Planned paths (TODO — none implemented yet)

| Path | Style | Status |
|------|-------|--------|
| `cpu_forward` | NEON forward bitmap, sks_hw `Query.h` BucketPlan style, **locked 16 thread** (NOT `hardware_concurrency`) | TODO |
| `aicore_forward` | Original Phase 2 §5.2.2 kernel (32 doc/block, leaf tag word gather, AND/OR short-circuit, GatherMask compact) | TODO |
| `aicpu_mask_filter` | Reuse current sks_hw `aicpu_mask_filter_kernel.cpp` as baseline | TODO |

## Sweep dimensions (when implemented)

- `bucket_count`: 1 / 100 / 800 / 1500 (cover P50 → P99+ query load)
- `expr_leaves`: 10 / 100 / 500 / 1000
- `density`: 0.005% / 0.05% / 0.5% / 5% (cover raw bit density range)
- `rpn_depth`: 1× / 3× leaf count (matches `FilterExpCompiler` NOT expansion)
- `cpu_threads`: **locked 16** (avoid 192-thread oversubscription regression)
- `npu_block_dim`: kernel-internal, multi-block per bucket

## Output format (when implemented)

Per-stage latency breakdown, NOT single total microsecond:

```
=== bucket_count=800 leaves=500 density=0.05% ===
path            H2D_bitmap_ms   kernel_exec_ms   D2H_ms   total_ms   matches
cpu_forward     n/a             n/a              n/a      12.34      12345
aicore_forward  3.21            1.87             0.45     5.53       12345
aicpu_mask      0.12            8.92             0.34     9.38       12345
```

Per-stage breakdown needed because Phase 2 §5.2.7 risk #1 (H2D bandwidth)
and risk #3 (cache hit rate) depend on stages, not totals.

## Build

```bash
cd /root/sks_hw/test/filter_phase2
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Skeleton builds `filter_phase2` binary that parses args and reports TODO.
No real kernels until Phase 2 §5.2 implementation fills stubs.

## Files

```
README.md       this file
CMakeLists.txt  build setup
main.cpp        arg parser + sweep loop skeleton, all paths stub to TODO
common/         (TODO) forward bitmap gen, real-shape expr gen, BucketPlan compiler
npu_common/     (TODO) ACL RAII, score buffer setup matching d_result_ws layout
cpu_forward/    (TODO) NEON baseline
aicore_forward/ (TODO) Phase 2 §5.2.2 original kernel
aicpu_mask/     (TODO) reuse aicpu_mask_filter_kernel.cpp
```
