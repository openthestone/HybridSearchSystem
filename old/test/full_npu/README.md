# full_npu — minimal full-corpus NPU pipeline testbed

Phase A skeleton. Goal: validate Tianji-style full-NPU path
(score + filter + aggregator + topk) on `dataset_HW.bin` at small scale,
compare against sks_hw IVF baseline.

See `真实数据推测-Tianji对比.md` for analytical extrapolation this testbed
aims to validate.

## Path B (re-implement minimal), not Path A (Tianji direct)

Reuses sks_hw infrastructure where possible:
- Score kernel: based on `kernel_vector_mmad_op.cpp` (sks_hw AIC Cube FP16)
- Filter kernel: based on `test/filter_cmp/npu_tianji/`
- Aggregator + TopK: port from Tianji `kernel_aggregator.h` / `kernel_topk.h`
- Data: direct `dataset_HW.bin` read (no Tianji 4-file conversion)

Trade-off: ~70-85% fidelity vs Tianji direct, but ~3× faster bring-up.

## Dataset format (`dataset_HW.bin`)

```
40B header: HYDSET2 magic + version(2) + doc_num + vector_dim + tag_num + reserved
Vectors: doc_num × vector_dim × 4B float32, row-major
Bitmaps: doc_num × stride × 8B uint64, row-major
         stride = ceil(tag_num / 64)
```

Production values:
- doc_num = 10,000,000
- vector_dim = 64
- tag_num = 35840
- stride = 560 word
- Total = 47.36 GB

## Phase A roadmap

### A.1 — Harness + CPU baseline

- [x] Arg parser, dataset loader (subset to first N docs)
- [x] Random Boolean expr generator (filter_expr compatible)
- [x] CPU brute-force reference: full-corpus score + filter + top-K
- [x] Per-stage latency breakdown
- [x] Verify on 1K → 1M doc subset

Run:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/full_npu --dataset=/root/sks_hw/dataset_HW.bin --doc_subset=10000 \
                 --leaves=100 --topk=100 --queries=20
```

### A.2 — NPU kernels (in progress)

- [x] `npu/kernel_score.cpp` — AIC Cube FP16 MMad, multi-block tiling (256 docs/block)
- [x] `npu/session.h` — ACL RAII wrapper (aclInit + stream)
- [x] `npu/score_launcher.h` — host launcher: FP32→FP16, H2D, ACLRT_LAUNCH_KERNEL, D2H
- [x] Verify vs CPU dot product, max rel diff < 0.1% on real data
- [x] Sweep 1K → 100K doc, measure NPU vs CPU latency
- [x] `npu/kernel_filter.cpp` — AIV forward bitmap RPN (16 docs/block, chunked 32-block launches)
- [x] `npu/filter_launcher.h` — host launcher with chunked launch + sync between chunks
- [x] Verify vs CPU EvalExpr, accuracy = 1.000000 on 1K → 64K doc
- [x] `npu/kernel_aggregator.cpp` — AIV scalar compaction (16 docs/block, chunked 32-block launches)
- [x] `npu/kernel_topk.cpp` — AIV scalar min-heap top-K (single-block serial, K ≤ 128)

Score kernel run:
```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
LD_LIBRARY_PATH=build/lib:$LD_LIBRARY_PATH ./build/full_npu \
    --dataset=/root/sks_hw/dataset_HW.bin \
    --doc_subset=100000 --queries=1 \
    --npu_score --npu_verify_count=512 --npu_score_repeat=10
```

Phase A.2 score sweep results (single NPU, 910B3, k=64, FP32 in / FP16 compute):

| M docs | NPU p50 ms | CPU proj ms (FP32) | speedup | max rel diff |
|---|---|---|---|---|
| 1,024   | 0.11  | 0.63 | 1.33× | 0.05% |
| 4,096   | 0.23  | 1.22 | 2.15× | 0.08% |
| 16,384  | 1.34  | 4.58 | 0.74× (H2D-bound) | 0.08% |
| 65,536  | 3.75  | 17.4 | 3.52× | 0.08% |
| 100,000 | 7.56  | 17.9 | 1.80× | 12.5% (small-mag) |

Note: max rel diff is dominated by docs with very small score magnitude
(abs diff <0.03). At p50 latency scales sublinearly with M — H2D is the
primary bottleneck below ~50K docs, MMad cube compute dominates beyond.

Filter kernel run:
```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
LD_LIBRARY_PATH=build/lib:$LD_LIBRARY_PATH ./build/full_npu \
    --dataset=/root/sks_hw/dataset_HW.bin \
    --doc_subset=16384 --leaves=100 --queries=1 \
    --npu_filter --npu_filter_repeat=2
```

Phase A.2 filter sweep results (single NPU, 910B3, leaves=100):

| M docs | NPU p50 ms | accuracy | npu=cpu matches |
|---|---|---|---|
| 1,024   | 6.3   | 1.000000 | 829/829 |
| 4,096   | 18.4  | 1.000000 | 3207/3207 |
| 16,384  | 97.7  | 1.000000 | 12820/12820 |
| 65,536  | 249.5 | 1.000000 | 51251/51251 |

Critical implementation detail: 910B3 AIV silently drops results when
launching >32 blocks at once. Chunked launch (32 blocks, sync, repeat)
with explicit block_offset_blocks parameter is required for correctness.

Latency dominated by bitmap H2D (M × 560 words × 8 bytes per query):
- 16K docs → 71 MB transfer → ~80% of total time
- Tianji's inverted-index design avoids this; our forward-bitmap path
  inherits the sks_hw bitmap layout for fair comparison.

Aggregator kernel run:
```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
LD_LIBRARY_PATH=build/lib:$LD_LIBRARY_PATH ./build/full_npu \
    --dataset=/root/sks_hw/dataset_HW.bin \
    --doc_subset=65536 --queries=1 \
    --npu_aggregator --npu_agg_repeat=5
```

Phase A.2 aggregator sweep results (single NPU, 910B3, scores+filter from CPU):

| M docs | NPU p50 ms | count match | id set match | score max abs diff |
|---|---|---|---|---|
| 1,024   | 0.15  | yes | yes | 0.000000 |
| 4,096   | 0.35  | yes | yes | 0.000000 |
| 16,384  | 1.22  | yes | yes | 0.000000 |
| 65,536  | 4.89  | yes | yes | 0.000000 |

Implementation: scalar compaction kernel (Tianji uses GatherMask vector
instruction for higher throughput; deferred). Each AIV block scans 16 docs,
writes compacted entries to its own output slot, writes per-block count
to a stride-16 padded slot.

Critical 910B3 AIV bug discovered: concurrent blocks writing to consecutive
u32 slots in the same GM region silently drop most block outputs
(non-deterministic). Cache-line racing on the count array. Fix: pad count
slots to one-per-cache-line (stride 16 = 64B). Without this, only ~10% of
blocks complete writes. With this, all blocks complete reliably.

Score path is currently CPU FP32 (Phase A.2 isolated test). Phase A.3 will
chain NPU score → NPU filter → NPU aggregator → NPU topk end-to-end.

TopK kernel run:
```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
LD_LIBRARY_PATH=build/lib:$LD_LIBRARY_PATH ./build/full_npu \
    --dataset=/root/sks_hw/dataset_HW.bin \
    --doc_subset=65536 --queries=1 --topk=100 \
    --npu_topk --npu_topk_repeat=5
```

Phase A.2 topk sweep results (single NPU, 910B3, K=100, compacted input from CPU):

| M docs | matched N | NPU p50 ms | position match | id set match | max score diff |
|---|---|---|---|---|---|
| 1,024   | 829    | 0.21  | 100/100 | yes | 0.000000 |
| 4,096   | 3,207  | 0.29  | 100/100 | yes | 0.000000 |
| 16,384  | 12,820 | 0.39  | 100/100 | yes | 0.000000 |
| 65,536  | 51,251 | 1.15  | 100/100 | yes | 0.000000 |

Implementation: scalar min-heap of size K, single-block serial scan over all
matched scores. Heap-sort produces descending output directly. Tianji uses
multi-block sample-sort (kernel_topk.h, 396 lines) for higher throughput;
deferred. K capped at 128 for static array sizing. Latency sublinear in N
(single-block serial scan dominates beyond ~10K matched docs).

### A.3 — End-to-end pipeline

End-to-end run (5 queries, all 4 kernels chained via host buffers):
```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
LD_LIBRARY_PATH=build/lib:$LD_LIBRARY_PATH ./build/full_npu \
    --dataset=/root/sks_hw/dataset_HW.bin \
    --doc_subset=65536 --leaves=100 --queries=5 --topk=100 \
    --npu_full
```

Phase A.3 sweep results (single NPU, 910B3, 5 queries, top-100):

| M docs | score ms | filter ms | agg ms | topk ms | TOTAL p50 ms | recall@100 |
|---|---|---|---|---|---|---|
| 1,024   | 0.09  | 4.4   | 0.17 | 0.21 | 7.5   | 1.000000 |
| 4,096   | 0.45  | 14.5  | 0.39 | 0.36 | 30    | 1.000000 |
| 16,384  | 1.0   | 128   | 1.5  | 0.76 | 175   | 1.000000 |
| 65,536  | 13.3  | 291   | 5.9  | 1.8  | 380   | 1.000000 |

Correctness: 100% recall@100 vs CPU brute force at all scales. Top-3 NPU
scores within 0.05 of CPU FP32 scores (FP16 quantization noise).

Latency dominated by filter stage (76-92% of total) due to bitmap H2D
(M × 560 words × 8 bytes per query). Score path now chunked-launch fixed
(see Critical fix below). Topk single-block serial scan dominates beyond
10K matched docs.

Critical fix during A.3: discovered score kernel (AIC cube) ALSO suffers
silent block-drop beyond 32 blocks per launch. Hidden by `--npu_verify_count`
defaulting to 256 (first 256 docs verified clean, rest silently zero).
Fix: chunked launch (≤32 blocks) + `block_offset_blocks` kernel param,
same pattern as filter and aggregator. Without this, 65K docs returned
score=0 for ~10% of docs in the tail, recall dropped to ~10%.

### Next steps (Phase A.3+)

- [x] Device-side fused pipeline (`--npu_fused`): chains 4 kernels via shared
      device buffers, skips score+filter D2H/H2D roundtrips.
- [x] Agg→topk direct device handoff: `kernel_topk_fused` reads aggregator's
      per-block layout directly, skips agg D2H + reupload roundtrip. Required
      own .cpp file (co-locating with `kernel_topk` broke AIV binary parse
      with "kernel_topk_fused_3 get kernel type failed").
- [x] Inverted-index filter (`--npu_fused_inv`): Tianji-style tag-major postings.
      Preprocess once (`tools/build_inverted_index`), then per query H2D only
      expr's tags' segment bitmaps (~100 tags × segs × 2B = 800KB at 65K docs,
      vs 280MB forward bitmap).
- [ ] Optimize: topk multi-block sample-sort (Tianji kernel_topk.h port).
- [ ] Scale test to 1M docs.

Fused pipeline sweep (single NPU, 910B3, 5 queries, top-100, agg→topk device handoff):

| M docs | npu_full p50 ms | npu_fused p50 ms | speedup | recall (fused) |
|---|---|---|---|---|
| 4,096   | 29    | 12.9  | 2.25× | 1.000000 |
| 16,384  | 111   | 97    | 1.14× | 1.000000 |
| 65,536  | 399   | 329   | 1.21× | 1.000000 |

Inverted-index filter sweep (single NPU, 910B3, 5 queries, top-100, docs cached):

| M docs | npu_full p50 ms | npu_fused_inv p50 ms | speedup | recall |
|---|---|---|---|---|
| 1,024   | 7.5   | 5.8   | 1.29×  | 1.000000 |
| 4,096   | 29    | 6.9   | 4.2×   | 1.000000 |
| 16,384  | 111   | 9.7   | 11.4×  | 1.000000 |
| 65,536  | 399   | 23    | 17.4×  | 1.000000 |
| 1,000,000 | —   | 313   | —      | 1.000000 |

Per-stage at 1M docs (steady state, p50):
```
h2d:    12.6 ms  (postings compact copy)
score:   4.5 ms
filter: 248   ms  (62500 segs / 32 = 1953 chunked launches, sync overhead dominant)
agg:    49    ms
topk:    1.4 ms  (multi-block B=32 partial heaps + 1 merge)
d2h:     0.1 ms
TOTAL: 313   ms
```

Per-stage breakdown at 65K docs (steady state, p50):
```
h2d:    0.44 ms  (was 280 ms — docs cached, postings 800KB)
score:  0.32 ms
filter: 16   ms  (was 290 ms — inverted postings, 800KB vs 280MB H2D)
agg:    3.2  ms
topk:   2.0  ms
TOTAL:  23   ms  (was 399 ms — 17× faster)
```

Implementation:
- `tools/build_inverted_index.cpp` — transposes dataset_HW.bin forward bitmap
  into tag-major postings (tag × segments × uint16). 280MB at 65K docs.
- `npu/kernel_filter_inverted.cpp` — Tianji-style AIV kernel. 1 block per
  16-doc segment. Pre-fetches all n_tags segment u16s into UB cache, then
  16 doc RPN evals read from cache (avoids 16× GM re-read amplification).
- `npu/inverted_index_loader.h` — loads inv index, builds compact per-query
  postings array with only expr's tags, launches inverted kernel.
- `npu/fused_pipeline.h` — `SetInverted(&idx)` switches to inverted path.
  Also caches FP16 docs across queries (avoids 80ms re-conversion).

Build inverted index:
```bash
./build/build_inverted_index --dataset=/root/sks_hw/dataset_HW.bin \
                              --out=dataset_HW_inv.bin --doc_subset=65536
```

Filter is now the bottleneck again at 16ms (was 290ms). Remaining time is
inverted postings GM reads scattered across tag-bitmap regions. Tianji's
 GatherMask vector instruction would parallelize this further; deferred.

### A.3 — Scale test

- 1K → 10K → 100K → 1M doc sweep
- Per-stage latency at each scale
- Compare with sks_hw serial on same subset (real data `真实数据测试结果-L2桶信息.md`)

## Sweep dimensions

- `doc_subset`: 1K / 10K / 100K / 1M (eventually 10M)
- `leaves`: 10 / 100 / 500 / 1000
- `density`: controlled by expr generator (random tag selection)
- `topk`: 10 / 100 / 1000
- `queries`: number of random queries per config

## Output format

Per-config table with per-stage breakdown:
```
=== doc_subset=10000 leaves=100 topk=100 queries=20 ===
path            score_ms   filter_ms   agg_ms    topk_ms   total_ms   recall
cpu_brute       12.34      45.67       0.89      1.23      60.13      1.000
npu_score       1.87       n/a         n/a       n/a       1.87       1.000  (A.2)
npu_full        1.87       0.65        0.12      0.34      2.98       1.000  (A.2)
```

## Defaults chosen (override via CLI)

| Decision | Default | Reason |
|---|---|---|
| Phase A scale start | 1K doc | debug-first, scale later |
| Filter kernel style | forward bitmap (sks_hw shape) | matches dataset_HW.bin directly, no conversion |
| Score input format | FP32 from disk, FP16 compute | simple disk read, Tianji-style compute |
| asyncContext parallel | no (single stream) | simplify Phase A, add in A.3 if needed |

## Files

```
README.md
CMakeLists.txt          CANN auto-detect, ascendc_library for kernels, link libascendcl
main.cpp                harness: arg parse, sweep loop, stage timing, --npu_score/--npu_filter paths
common/dataset_hw.h     header-only dataset_HW.bin reader (mmap, subset)
common/expr.h           random Boolean expr gen (TAG/NOT_TAG/AND/OR)
common/cpu_ref.h        CPU brute-force: score + filter + top-K
common/timer.h          latency measurement
npu/kernel_score.cpp    AscendC MMad: query · docs → scores, 256 docs/block multi-block
npu/kernel_filter.cpp   AscendC AIV filter: forward bitmap RPN, 16 docs/block, chunked launch
npu/kernel_aggregator.cpp AscendC AIV scalar compaction, 16 docs/block, strided count writes
npu/kernel_topk.cpp     AscendC AIV scalar min-heap top-K, single-block serial scan
npu/session.h           AclSession RAII: aclInit, device, stream
npu/score_launcher.h    ScoreLauncher: FP32→FP16, H2D, kernel launch, D2H
npu/filter_launcher.h   FilterLauncher: RPN encode, H2D bitmap+rpn, chunked launch + sync, D2H
npu/aggregator_launcher.h AggregatorLauncher: H2D scores+filter, chunked launch, host reduce
npu/topk_launcher.h     TopkLauncher: H2D compacted scores+ids+count, single-block launch, D2H top-K
npu/fused_pipeline.h    FusedPipeline: shared device buffers, chain 4 kernels no inter-stage H2D/D2H (score+filter only)
main.cpp --npu_full     Phase A.3 chained launchers
main.cpp --npu_fused    Phase A.3+ device-side fused
```
