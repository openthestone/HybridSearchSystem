# 多卡 NPU 分片 可行性评估

## 实测结论(2026-07-08):p50-p95 稳定真赢(~20%),recall 100%;p99 受限于共享机器抖动(非算法)

2 卡 in-process fan-out 全链路实现并跑通(branch `feat/multi-card`:converter `--doc_offset`、harness N-DataTable 加载 + fan-out + 跨卡合并、`--cpu_doc_offset` 分片对拍、`--shard_latency_dump` + analyze_shard_latency.py 尾部拆解)。frnew_syn_v2 全 10000 query、warmup=200、**3 次 rerun**(loop_count=16):

| 指标 | 单卡 10M(3 run) | 2 卡 2×5M(3 run) | 判断 |
|---|---|---|---|
| p50 | 4.20 / 4.20 / 4.74 | **3.35 / 3.39 / 3.36** | **-20%,更稳** |
| p90 | 4.73 / 4.75 / 6.18 | **3.80 / 3.86 / 4.00** | **-20%,更稳** |
| p95 | 4.92 / 4.94 / **12.9** | **4.02 / 4.09 / 4.45** | **赢且远更稳** |
| p99 | 5.5 / 6.1 / **25.4** | 11.9 / 12.4 / 15.3 | 见下 |
| max | 35 / 41 / 52 | 44 / 48 / 50 | 都是环境上限 |
| recall | 100% | 100% | 一致 |

**correctness 完全验证**:合并后的全局 top-3500 逐位等于对全 10M 的 CPU 暴力(recall 100%,全部 rerun 全 100%)——converter offset、fan-out、合并全对;分布式 top-K 精确性成立(每片 top-K 的并集必含全局 top-K)。

**优化后最终**(k-way merge + last-shard-inline,2 次 rerun,recall 100%):**p50 3.00 / p90 3.5-3.75 / p95 3.84-4.36**,即 **p50 -29% / p90 -25% / p95 -18%** vs 单卡。关键优化:`--shard_latency_dump` 显示所谓 ~0.56ms "overhead" 其实主要是**跨卡 merge**(0.43ms,concat 7000 候选 + partial_sort),线程 spawn 只 ~0.13ms;把 merge 换成 **O(K·N) k-way 合并**(每片有序前缀已排好、全局 top-K ⊆ 各片 top-K 并集)后 merge 0.434→0.083ms、p50 3.39→3.00。所以中位数增益从 ~20% 扩到 ~29%,全程 recall 100%。

**尾部拆解(--shard_latency_dump + analyzer)定位了 p99**:
- 两片**负载均衡**(mean 2.94/3.08,straggler 各 ~50%),**不是**负载不均。
- 尾部 top-1% 里 **98% 是"一张卡慢、另一张快"** → 独立 per-shard 尖峰被 **max-of-2** 放大。
- 关键:**单片 5M 单跑的 p99 只有 3.6,但 max=40**——那个 ~40ms 是**环境 ambient 尖峰**(单跑也有、落在 p99 之外),不是并发争用、代码不可修。
- **决定性反转**:单卡自己的 p99 就狂抖(3 次 rerun:5.5 / 6.1 / **25.4**;有一次 p95 都飙到 12.9)。**p99 是这台共享 910B3 的机器噪声,不是多卡引入的缺陷**。之前"p99 +160%(vs 5.84)"里的 5.84 是单卡的幸运值。
- loop_count 扫 12/16/20/24 的 p99 非单调(20.7/20.6/8.0/12.6)= 在追噪声,不是可控信号。

**修正后的定性**:
- **p50-p95:2 卡稳定真赢 ~20%,且比单卡更稳**(单卡有 p95 飙到 12.9 的坏 run,2 卡三次 p95 都 ~4.0-4.5)。流式段(向量+filter ~2.9ms)减半的收益兑现了(净掉固定 topK+合并+overhead)。
- **p99:两者都被 ambient 抖动主导**。单卡"通常好(5.5)偶尔崩(25)",2 卡因 max-of-2 "稳定地中等(12-15)"。是 fan-out 的固有效应(`P(max_N>t)≈N·P(single>t)`)叠加**这台机器的 ~40ms ambient 尖峰**,**代码不可修**。

**Caveats**:(1) 这台是共享/噪声 910B3,ambient ~40ms 尖峰是环境(单片单跑也有);**专用干净机器上 per-card 尾部紧,max-of-2 放大很小,2 卡 p99 大概率能持平甚至赢——但这台测不到**。(2) 只有 2 卡;8 卡 max-of-N 放大更重。(3) fan-out overhead(spawn/sched)~0.56ms/query 已优化(最后一片主线程内联,2→1 spawn),进一步压中位数。

**结论**:多卡分片在**中位数~p95 是明确、可复现、且更稳的赢(~20%)**,recall 100%。**p99 在这台共享机器上受 ambient 抖动主导、代码不可修**——不能说多卡"改善 p99",但也不是它的锅(单卡同样抖)。若目标含吞吐 / 中位数 / p95,多卡值得用;若唯 p99 且只有这台噪声机器,收益不确定。真正判定多卡对 p99 的效果需专用干净多卡机器。以下是原始可行性评估(架构判断仍准确),保留供参考。

---

## 结论(2026-07-08):可行,且比预期便宜 — 难点已在 `new/` 里写好

多卡分片是之前定的两条"能赢带宽墙"正方向之一(另一条是 INT8)。评估结论:

- **架构上完全可行,recall-safe**:把 10M 语料按 doc-id 切成 N 份、每卡搜 1/N、跨卡合并 top-K,是**精确全召回**(每片对自己那 1/N 穷举,合并是精确 top-K)。
- **直击已证明的瓶颈**:msprof 证过 ~4ms 是 **HBM 带宽墙**(向量 MMad `aic_mte2_ratio=0.997`)。每卡只流式自己那 1/N 的向量 → 带宽墙按 ~N× 削。**这正是"加流式通道"这类赢法。**
- **难的部分华为已经实现**:per-device 内存/流池、per-shard doc-id 映射、跨卡 top-K 合并——全在 `new/` 里,只是 port 层没接。缺口全是 port 层的"接线"。

**关键区别于 parallel 负结果**:之前 `parallel filter‖vector` 负优化是因为两个核**抢同一张卡的 HBM 带宽**。多卡不同卡有**各自独立的 HBM 通道**,不共享带宽——所以多卡能赢,而同卡并行赢不了。这是同一条带宽墙理论的正反两面。

---

## 硬件现实:这台 box 只有 2 卡

`ASCEND_RT_VISIBLE_DEVICES` 为空但只挂载了 davinci3 + davinci7,ACL 重编号 DEVICE_ID=0→NPU3、1→NPU7。所以:

> **代码设计支持 1-8 卡,但这台机器最多实测 2 卡(≈2× 带宽)。8 卡是架构外推,需要更多卡的机器才能验证。**

---

## 现有骨架(agent 实地扫码,file:line 为证)

难点三件套**已存在**,只是不被 port 编译:

| 能力 | 位置 | 状态 |
|---|---|---|
| `isMultiShard` 结果打包(保留 `scoreWithIndexes`+全局 id 供合并) | `new/common/full_recall_result.h:41-53`、`result_aggregator.cpp:190-217` | ✅ 已实现 |
| **跨卡 top-K 合并**(fan-out N 片 → concat → 全局 sort → 取 K) | `new/process/full_search_proxy.cpp:76-141` (`SearchBatcherProcessMulti`) | ✅ 已实现,~60 行 |
| per-device 内存池 / 流池(按 deviceId keyed) | `gm_memory_manager.h:19-30`、`stream_manager.cpp:14-47` | ✅ 天然多卡 |
| 每处 NPU 阶段都 `aclrtSetDevice(m_deviceId)` | `full_recall_searcher.cpp:46,66,82,...` | ✅ 可寻址多卡 |
| per-shard doc-id 映射(`GetGDocId` 是 offset-agnostic 数组查表) | `doc_id_mapping.cpp:169-176` | ✅ 每片各自全局 id,零冲突 |
| **完整多片/多卡加载器**(`indexLoadCards[replica][shard]→deviceId`) | `new/data/data_table_repository.cpp` | ✅ 存在,但 port 排除(`port/CMakeLists.txt:277-282`) |

`fr_builder` 的 `Build(schema, shardId, ...)` 签名本就带 `shardId`,只是 CLI 硬编码传 0(`new/build/builder/main.cpp:48`)。

---

## 缺口(全在 port 层)+ 工作量

| # | 缺口 | 位置 | 工作量 | 说明 |
|---|---|---|---|---|
| 1 | 转换器加 `--doc_offset` | `port/converter/build_input.cpp:265-317` | **S** | 现在 `--docs N` 只能切前缀 `[0,N)`;`gdocid=seg_begin+i` 同时当"写入的全局 id"和"源行下标"。加 `OFF`:`gdocid=OFF+seg_begin+i`,即可切任意 `[OFF, OFF+N)` 片,全局 id 正确。 |
| 2 | 建 N 份分片索引 | 脚本(run.sh) | **S** | 对每片跑一遍 converter(带 offset)+ builder。**零代码**,纯脚本循环。 |
| 3 | harness 多卡驱动 + 合并 | `port/harness/main.cpp` | **M(核心)** | 加载 N 个 DataTable(每个 `LoadData(deviceId_k, shard_k)`)、N 线程 fan-out(每线程 pin 一张卡、`isMultiShard=true`)、host 侧 concat+sort+取 K。合并逻辑照抄 `SearchBatcherProcessMulti`(~60 行)。 |
| 4 | 设备派发 | 复用现有 | **0** | per-device 池/流已就绪,无需改。 |
| 5 | recall 验证 | 复用现有 | **S** | 合并结果必须 == 单卡精确 top-K,用现成 CPU 暴力 `--recall_queries` 对拍。 |

**总量级:S+S+M ≈ 3-5 天出可测版本。** 比 filter-first(2-3 周)小得多——因为难点华为写好了。

---

## 预期收益 + Amdahl 上限(重要)

不是干净的 N× 加速。逐阶段:

| 阶段 | ~当前(单卡,v2) | 分 N 卡后 |
|---|---|---|
| 向量 MMad(HBM-bound) | ~1.76ms | **~N× 削**(每卡流 1/N 向量) |
| 文本 filter(HBM-bound) | ~1.1ms | **~N× 削**(每卡 filter 1/N posting) |
| per-shard 聚合 + **固定 K=3500 的 topK** | ~1.5ms | **不随 N 缩** —— 每片仍须返回自己的 top-3500(某片可能含全部全局 top-3500),K 固定 |
| **跨卡合并**(新增) | 0 | **随 N 涨** —— concat N×3500 候选 + host sort(N=8 → 28000) |
| host 固定开销(query 构建、fan-out 派发、D2H) | ~fixed | 不缩,派发还加线程启动开销 |

所以:**HBM-bound 的流式段(~2.9ms)按 ~N× 缩,但固定-K 的 topK + 跨卡合并不缩且合并随 N 涨** → **亚线性,且在 8 卡处趋平**。

粗估(p50 4.0ms 起):
- **2 卡**:向量+filter 段 ~减半 → **p50 ~2.7-3.2ms**(~1.3-1.5×,不是 2×,被固定 topK+合并拖)
- **4 卡**:流式段 ~1/4 → **p50 ~2.2-2.6ms**,合并开始吃掉增益
- **8 卡**:流式段 ~0.35ms,但每片 topK-3500 + 28000-merge 主导 → **收益趋平**,最佳性价比在 **2-4 卡**

**这个 Amdahl 结构本身值得先量化**——固定 K=3500 是华为真实工作负载,topK 段不缩是硬约束。

---

## 两条实现路径

- **(A) in-process fan-out(推荐,给 p99 延迟)**:一个 `fr_search` 进程,N 线程各 pin 一张卡、共享 query、host 合并。复用 reference 的 `SearchBatcherProcessMulti`。**单 query 延迟真降 N×**(流式段)。风险:进程内多卡并发——但不同卡不共享 HBM,不会像同卡 parallel 那样抢带宽;`aclrtSetDevice` 是 thread-local,池按 device keyed,天然安全。需 `TIANJI_EXECUTOR_THREADS≥N`。
- **(B) multi-process(零代码,但只提吞吐)**:launch N 个 `fr_search`,各搜一片,外部合并 topk 文件。零引擎改动,但**不降单 query 延迟**(除非再搭 RPC fan-out)。适合离线吞吐,不适合 p99 目标。

**主目标是 p99 → 走 (A)。** (B) 可作为第一步快速验证"分片索引 + 合并 recall 正确"再上 (A)。

---

## recall 验证计划

分片必须**逐位等于单卡**结果:
1. 建 2 片索引(each 5M,带 offset),单独各自 `--recall_queries` 对 CPU 暴力 → 每片对自己 1/N 正确。
2. in-process 合并后的全局 top-3500 == 单卡 10M 的 top-3500(用现成 `--dataset_hw` CPU 暴力对拍,阈值 100%)。
3. 边界:某 query 的全局 top-K 全落在一片时(合并不能丢),per-shard K ≥ global K 必须成立(现在都 3500,天然满足)。

---

## 建议增量顺序

1. **converter `--doc_offset`**(S)+ 建 2 片 5M 索引(脚本)→ 各自 recall 100% 自检。
2. **harness 多卡驱动 + 合并**(M,照抄 `SearchBatcherProcessMulti`)→ in-process 2 卡 fan-out。
3. **A/B**:2 卡 vs 单卡,量化流式段的实际缩放 + 合并新增开销,验证 recall==单卡。
4. 若 2 卡收益兑现,记录**外推到 8 卡的 Amdahl 模型**(固定 topK+合并的占比决定天花板);8 卡需更多卡的机器实测。

**与 INT8 正交可叠加**(INT8 减每卡流式字节、多卡加通道数)。多卡是两条正方向里**唯一 recall-safe + 复用已有实现**的,工作量最小、风险最低。
