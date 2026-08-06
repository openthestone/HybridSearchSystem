# filter_cmp — Tianji AIV kernel correctness regression

**Scope (changed)**: Tianji AIV filter kernel correctness regression only.
NOT a Phase 1/Phase 2 performance predictor. See `纯NPU路径优化草案.md §4.5`
for why historical 100K/1024/5% numbers don't extrapolate to real data
(10M docs × 35672 tags, density < 0.05%, forward doc→tag bitmap).

For Phase 2 AICore forward-bitmap kernel testbed, see `test/filter_phase2/`.

Standalone. No deps on sks_hw or Tianji. Compares three filter paths:

| Path | Style | Bitmap layout | Where compute runs |
|------|-------|---------------|--------------------|
| `cpu`     | sks_hw forward     | doc → uint64[num_tags/64] tag bitmap    | CPU (NEON optional) |
| `simple`  | simplified NPU     | tag → uint64[docs/64] doc bitmap (flat) | AIV kernel |
| `tianji`  | Tianji port        | tag → uint16[seg_docs/16] per-segment posting + BIT_LIST sparse | AIV kernel (PostingBitListToSet + BitmapTextFilter) |

## Scale
- 100,000 docs × 1,024 tags
- Synthetic raw data: each (doc, tag) bit set with prob `tag_density` (default 0.05)
- One boolean filter expr, depth 4–8 leaves, AND/OR/NOT

## Build
```bash
cd /root/sks_hw/test/filter_cmp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Run
```bash
export LD_LIBRARY_PATH="build/lib:\
/usr/local/Ascend/cann-8.5.0/aarch64-linux/lib64:\
/usr/local/Ascend/driver/lib64:\
/usr/local/Ascend/driver/lib64/common:\
/usr/local/Ascend/driver/lib64/driver"

./build/filter_cmp \
    --doc_num=100000 \
    --tag_num=1024 \
    --segments_num=10 \
    --tag_density=0.05 \
    --expr_leaves=6 \
    --warmup=50 \
    --rounds=500
```

Output: cold + warm p50/p99 per-query latency (us) for each of three paths.

## Sample results — regression baseline, do NOT extrapolate

100K docs × 1024 tags, 5% density, 6-leaf expr, 16 CPU threads, 8 NPU blocks:
```
  cpu         cold=  1077.5 us   warm p50=   924.4 us   p99=  1262.9 us
  npu_simple  cold=   551.8 us   warm p50=   490.0 us   p99=   529.8 us
  npu_tianji  cold=   685.2 us   warm p50=   658.5 us   p99=   702.1 us
```
Both NPU kernels produce identical results to CPU reference (0 mismatches).

### Why this number does not predict Phase 1/Phase 2 behavior

Real sks_hw production data shape (`真实数据测试结果-L2桶信息.md`):

| Dimension | This testbed | Production | Mismatch |
|---|---|---|---|
| Data model | inverted BIT_SET (Tianji port) | forward doc→tag bitmap | kernel input structure incompatible (草案 §4 阻断点 #1) |
| Tag count | 1024 | **35672** (stride 558 word) | UB/HBM estimate off by ~35× |
| Docs per query | 100K (segment-flat) | **~1.5M** (P99+ 1454 bucket × 1024 doc) | launch count / pipeline overlap differ |
| Density | 5% (5000/100K) | **< 0.05%** (P99+ sel mean 0.027) | mask kernel branch behavior differs ~200× |
| Expr depth | 6 leaves | **≤1000 tags → RPN ≤3000** | `kMaxStackDepth=64` overflows (草案 §5.2.7 风险 #4) |
| CPU thread | `hardware_concurrency` (~192, oversubscribed) | optimal 16 | CPU baseline is the regression case, not a reference |
| Score coupling | synthetic | NPU FP16 `d_result_ws` direct read | mask kernel perf depends on score buffer layout |

**Do not cite** "NPU 1.4–1.9× faster than CPU" or "5–9× at 1.5M docs" as Phase 1/2 predictions.
The 5–9× / +74-110% leaf-depth / 192→16 thread numbers were historical conversational
claims that were never landed in this testbed — they need reproduction at real data
shape before being used as baseline.

## Files
```
common/         expr, raw data, timer, CPU filter
npu_common/     ACL RAII, posting encoder
npu_simple/     minimal AIV kernel + host launcher
npu_tianji/     Tianji AIV port + host launcher
main.cpp        harness + measurement
```
