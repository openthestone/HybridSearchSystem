# dataset_HW.bin bitmap 生成改进方案 v2

> 配套文档: `纯NPU路径优化草案.md`、`真实数据测试结果-L2桶信息.md`
> 目标: 替换 `generate_dataset.py` 中全随机 Bernoulli bitmap,生成更贴近真实数据的 bitmap section。

---

## 0. 问题陈述

当前 `dataset_HW.bin` bitmap 部分采用 `random.random() < tag_density` 独立采样,
vector 部分与真实数据相近,bitmap 部分与真实数据严重失真。

### 失真点

| 维度 | 全随机生成 | 真实数据 | 后果 |
|---|---|---|---|
| Tag freq 分布 | 全部 ≈ p,σ 极小 | power-law,universal+rare 二极 | sentinel folding 失效 |
| Tag 共现 | i.i.d. 独立 | 强相关(topic cluster) | AND 选择率预测错误 |
| Per-doc tag 数 | 近似正态集中 | long-tail,均值<1000,尾部到 3000+ | 桶内 mask 计算量失真 |
| L2 桶 All0 比例 | ≈ 0 | 实测 P50=27,P99=1250 | IVF pre-filter 形同虚设 |
| 选择率范围 | 独立乘积,集中在 0.001-0.01 | 实测 0.028-0.593 跨度 | 性能基准无意义 |

---

## 1. 真实数据特征回顾(来自 §L2桶信息 + §草案 §5.3.2)

### 1.1 Tag 频率结构

- 总 tag 数: **35672**
- Universal tag: 极个别 freq ≈ 1.0
- 稀有 tag: 很多 ≤10 docs / 10M
- 中间区间: Zipf-like long tail

### 1.2 Per-doc tag 数

- 均值 < 1000
- 长尾分布(少数 doc 数千 tag)

### 1.3 选择率分布(实测 P50-P99 跨度)

| 延迟段 | 平均选择率 Sel |
|---|---|
| P50 | 0.378 |
| P50-P80 | 0.593 |
| P80-P90 | 0.404 |
| P90-P99 | 0.056 |
| ≥P99 | 0.028 |

### 1.4 L2 桶统计

| 延迟段 | Total 桶 | Non0 桶 | All0 桶 |
|---|---|---|---|
| P50 | 96 | 70 | 27 |
| P50-P80 | 132 | 87 | 44 |
| P80-P90 | 165 | 102 | 63 |
| P90-P99 | 535 | 207 | 328 |
| ≥P99 | 1454 | 204 | 1250 |

All0 桶比例随查询选择性上升而急升 → IVF sentinel folding 关键路径。

---

## 2. 生成方案: 三层 tag 模型 + cluster affinity

### 2.1 三层 tag 划分

```
tag_num = 35672
universal_tag_num  = 20       # freq 0.90-1.00
common_tag_num     = 3052     # Zipf s≈1.0
rare_tag_num       = 32600    # freq ≤ 1e-5
```

| 层 | tag id 区间 | freq | 文件角色 |
|---|---|---|---|
| universal | [0, 20) | U(0.90, 1.00) | 驱动 `kSentOnes`(has_tag=all) |
| common | [20, 3072) | `0.3 / (rank+5)^1.0` | 主体桶过滤 |
| rare | [3072, 35672) | `≤10 doc / 10M` | 驱动 `kSentZero`(no_have) + 高选择性 query |

### 2.2 Topic cluster 划分(common 层)

```
cluster_num = 256
cluster_size_avg = common_tag_num / cluster_num ≈ 12 tag/cluster
```

- common tag 按 hash 分配到 256 cluster
- universal tag 跨 cluster 共享
- rare tag 不归属 cluster(独立稀疏)

### 2.3 Doc→tag 采样流程

每 doc:

1. **选 cluster**: 加权采样 1-3 个 cluster(cluster popularity 本身 Zipf)
2. **抽 tag 数**: `n = LogNormal(μ=6.0, σ=0.6)`,截断 [50, 5000]
3. **universal 注入**: 强制纳入 3-8 个 universal tag(真实场景: 大部分 doc 都有"性别=未知""平台=通用"这类元 tag)
4. **common 采样**: 在所选 cluster 内,按 Zipf 概率无放回采样 `n - universal_count` 个 tag
5. **rare 采样**: 30% 概率附加 0-3 个 rare tag(模拟偶发的精确分类)

### 2.4 关键参数与对应效果

| 参数 | 调节目的是 | 目标值 |
|---|---|---|
| `universal_tag_num` | freq≈1 比例 | 20 |
| `universal_freq_min` | has_tag=all 桶比例下限 | 0.90 |
| `common_zipf_s` | common 层 freq 衰减陡度 | 1.0 |
| `common_freq_max` | 最热 common tag freq | 0.5 |
| `cluster_num` | topic 粒度 | 256 |
| `cluster_per_doc` | 每 doc 跨 cluster 度 | U(1, 3) |
| `lognormal_mu` | per-doc tag 数对数均值 | ln(400) ≈ 6.0 |
| `lognormal_sigma` | per-doc tag 数方差 | 0.6 |
| `rare_attach_prob` | doc 含 rare tag 概率 | 0.30 |
| `rare_attach_max` | 单 doc rare tag 上限 | 3 |

---

## 3. 校准 (Calibration)

### 3.1 校准目标

对齐 `真实数据测试结果-L2桶信息.md`:

| 指标 | 实测目标 | 调节杠杆 |
|---|---|---|
| AND-expr P50 Sel | 0.378 ± 0.05 | common_freq_max ↑、cluster_size ↑ |
| AND-expr P80-P90 Sel | 0.404 ± 0.05 | common_zipf_s ↓ |
| AND-expr P99 Sel | 0.056 ± 0.01 | rare_attach_prob ↑ |
| AND-expr ≥P99 Sel | 0.028 ± 0.01 | rare_tag_num ↑ |
| L2 All0 桶(P50) | 27 ± 5 | rare 集中度 ↑ |
| L2 All0 桶(≥P99) | 1250 ± 100 | cluster 隔离 ↑ |
| has_tag=all 桶比例 | 30-50% | universal_tag_num ↑ |

### 3.2 校准流程

```
loop:
  1. 用当前参数生成 mini dataset (100K docs)
  2. 随机生成 6-leaf filter expr × 10000,跑 forward bitmap filter
  3. 统计 Sel 直方图 vs 实测 P50/P80/P90/P99
  4. 若偏差 > 10%,二分调 (zipf_s, rare_attach_prob, cluster_num)
  5. 收敛后导出最终参数,跑全量 10M
```

### 3.3 sentinel folding 命中率检查

```
对生成的 dataset:
  - 计算每 L1 桶 × 每 tag 的 (has_tag, all_have_tag)
  - 统计 kSentOnes / kSentZero 比例
  - 期望: kSentOnes + kSentZero > 30% (否则 IVF pre-filter 失效)
```

---

## 4. 文件格式(保持兼容)

按 CLAUDE.md §6 dataset_DEEP.bin 格式不变,只改 bitmap section 内容。

### 4.1 Header

```c
struct DatasetCacheHeaderDisk {
    char     magic[8];      // "HYDSET2\0"
    uint32_t version;       // 2
    uint32_t padding;       // 0
    uint64_t doc_num;       // 10,000,000
    uint32_t vector_dim;    // 64
    uint32_t tag_num;       // 35672
    uint32_t reserved;      // 0
    uint32_t tail_padding;  // 0
};  // 40 bytes
```

### 4.2 Bitmap section layout

```
tag_stride = ceil(tag_num / 64) = ceil(35672 / 64) = 558 uint64
bitmap_size = doc_num × 558 × 8 = 10M × 558 × 8 = ~44.6 GB
```

> **注意**: 44.6 GB 偏大。若 disk 紧张,可降 doc_num 到 1M 试跑(校准阶段足够)。

### 4.3 Tag id 顺序

- universal tag 占 id [0, 20)
- common tag 占 id [20, 3072)
- rare tag 占 id [3072, 35672)

→ bitmap 行内 bit 位置可推断 tag 类别,便于 verify。

---

## 5. 参考实现(伪代码)

```python
import numpy as np
from collections import defaultdict
from multiprocessing import Pool
import random

# ============ Layer 1: tag tier setup ============
TAG_NUM         = 35672
UNIVERSAL_NUM   = 20
COMMON_NUM      = 3052
RARE_NUM        = TAG_NUM - UNIVERSAL_NUM - COMMON_NUM  # 32600

UNIVERSAL_IDS   = list(range(0, UNIVERSAL_NUM))
COMMON_IDS      = list(range(UNIVERSAL_NUM, UNIVERSAL_NUM + COMMON_NUM))
RARE_IDS        = list(range(UNIVERSAL_NUM + COMMON_NUM, TAG_NUM))

# ============ Layer 2: cluster assignment ============
CLUSTER_NUM     = 256
random.seed(42)
cluster_of      = {t: i % CLUSTER_NUM for i, t in enumerate(COMMON_IDS)}
tags_in_cluster = defaultdict(list)
for t, c in cluster_of.items():
    tags_in_cluster[c].append(t)
all_clusters    = list(range(CLUSTER_NUM))

# ============ Layer 3: freq tables ============
# common freq: Zipf
COMMON_FREQ = {}
for rank, t in enumerate(COMMON_IDS):
    COMMON_FREQ[t] = min(0.5, 0.3 / (rank + 5) ** 1.0)

# universal freq: U(0.90, 1.00)
UNIVERSAL_FREQ = {t: random.uniform(0.90, 1.00) for t in UNIVERSAL_IDS}

# rare freq: ≤ 1e-5
RARE_FREQ = {t: random.uniform(1e-7, 1e-5) for t in RARE_IDS}

# cluster popularity: Zipf
CLUSTER_POP = np.array([1.0 / (i + 1) ** 0.8 for i in range(CLUSTER_NUM)])
CLUSTER_POP /= CLUSTER_POP.sum()


def gen_one_doc(doc_id, rng):
    # 1. pick clusters (1-3)
    k = rng.integers(1, 4)
    chosen = rng.choice(all_clusters, size=k, replace=False, p=CLUSTER_POP)

    # 2. log-normal tag count
    n = int(rng.lognormal(mean=6.0, sigma=0.6))
    n = max(50, min(5000, n))

    # 3. universal injection
    doc_tags = set()
    for t in UNIVERSAL_IDS:
        if rng.random() < UNIVERSAL_FREQ[t]:
            doc_tags.add(t)
    # ensure at least 3 universal
    while len(doc_tags & set(UNIVERSAL_IDS)) < 3:
        doc_tags.add(rng.choice(UNIVERSAL_IDS))

    # 4. common sampling from chosen clusters
    candidate = []
    candidate_w = []
    for c in chosen:
        for t in tags_in_cluster[c]:
            candidate.append(t)
            candidate_w.append(COMMON_FREQ[t])
    if candidate:
        candidate_w = np.array(candidate_w)
        candidate_w /= candidate_w.sum()
        pick_n = min(n - len(doc_tags), len(candidate))
        picked = rng.choice(candidate, size=pick_n, replace=False, p=candidate_w)
        doc_tags.update(picked.tolist())

    # 5. rare tag injection
    if rng.random() < 0.30:
        rare_pick = rng.integers(0, 4)
        if rare_pick > 0:
            doc_tags.update(rng.choice(RARE_IDS, size=rare_pick, replace=False).tolist())

    return doc_id, sorted(doc_tags)


def worker(doc_range, seed):
    rng = np.random.default_rng(seed)
    out = []
    for d in doc_range:
        out.append(gen_one_doc(d, rng))
    return out


def main(doc_num=10_000_000, out_path="dataset_HW.bin"):
    # parallel
    chunks = np.array_split(range(doc_num), 64)
    with Pool(64) as pool:
        results = pool.starmap(worker, [(chunk, i) for i, chunk in enumerate(chunks)])
    # ... write header + vector section + bitmap section
```

---

## 6. 验证 checklist

生成后必须验证:

- [ ] **Tag freq 直方图**: 排序后应为 power-law 曲线,头部 20 个 freq>0.9,尾部稀疏 ≤1e-5
- [ ] **Per-doc tag 数直方图**: log-normal 形状,均值 400-800,99 分位 ≤3000
- [ ] **随机 6-leaf AND expr Sel**: P50 落在 0.30-0.45
- [ ] **`(rare_1 AND rare_2)` Sel**: <0.001,大多数桶 All0
- [ ] **universal tag 单独 filter Sel**: >0.9,几乎所有桶 has_tag=all
- [ ] **IVF sentinel folding 命中率**: kSentOnes + kSentZero > 30%
- [ ] **All0 桶数(P50 query)**: 20-35
- [ ] **CRC / size check**: bitmap_size = doc_num × 558 × 8

---

## 7. 风险与边界

### 7.1 生成时间

10M doc × Zipf 采样单进程约 8-12 小时。强制多进程(64 worker 并行,per-doc 独立),墙钟 <15 min。

### 7.2 内存峰值

每 worker 持有 candidate 数组(平均 ~30 tag)→ 可忽略。最终 bitmap 写盘流式,无需全内存。

### 7.3 随机种子

固定 `seed=42`,保证 sks_hw vs Tianji 同 bitmap → recall/perf 比对公平。

### 7.4 Cluster 数选择

256 cluster 是基于"每 cluster ~12 common tag"启发式。如真实数据有 ground truth cluster 标签
(从 tag co-occurrence 矩阵反推 NMF/spectral clustering),优先用真实 cluster 划分。

### 7.5 与 §草案 §5.3.2 一致性

本方案直接服务 §5.3.2 IVF sentinel folding:
- universal tag → kSentOnes 路径覆盖
- rare tag → kSentZero 路径覆盖
- cluster → All0 桶自然涌现

若 dataset 不改进,sentinel folding 优化在合成数据上无法体现收益。

---

## 8. 后续工作(超出 v2 范围)

- **真实数据采样**: 若能拿到真实 dataset 样本(脱敏),直接 fit Zipf 参数 + cluster 数
- **时间衰减**: tag freq 可加时间维度(模拟 trend tag),生成多版本 bitmap 对比
- **Vector-bitmap correlation**: 当前 vector 与 bitmap 独立生成。真实场景中,相近 vector 倾向共享 tag
  (同 cluster)。可在生成 bitmap 后,用 cluster label 重排 vector 顺序,模拟该相关性。

---

## 9. 落地步骤

1. 写 `dataset_gen_v2.py`(基于本伪代码)
2. 先 100K docs 跑校准(§3.2),调参 1-2 轮
3. 跑 1M docs,验证 §6 全部 checklist
4. 跑全量 10M docs(并行 <15min)
5. 替换 `dataset_HW.bin`,重跑 sks_hw serial/parallel + Tianji recall
6. 对比新旧 dataset 在 `真实数据测试结果-L2桶信息.md` 的指标差异
7. 若 sentinel folding 命中率显著提升 → 更新 `纯NPU路径优化草案.md` §5.3.2 收益预估

---

## 附录 A: 参数速查表

```python
TAG_NUM              = 35672
UNIVERSAL_NUM        = 20
COMMON_NUM           = 3052
RARE_NUM             = 32600
CLUSTER_NUM          = 256
CLUSTER_PER_DOC_MIN  = 1
CLUSTER_PER_DOC_MAX  = 3
LOGNORMAL_MU         = 6.0      # ln(400)
LOGNORMAL_SIGMA      = 0.6
TAG_COUNT_MIN        = 50
TAG_COUNT_MAX        = 5000
UNIVERSAL_FREQ_MIN   = 0.90
UNIVERSAL_FREQ_MAX   = 1.00
COMMON_ZIPF_S        = 1.0
COMMON_FREQ_MAX      = 0.5
COMMON_FREQ_FLOOR    = 0.001
RARE_FREQ_MAX        = 1e-5
RARE_FREQ_MIN        = 1e-7
RARE_ATTACH_PROB     = 0.30
RARE_ATTACH_MAX      = 3
SEED                 = 42
```

## 附录 B: 目标统计量

| 量 | 目标 |
|---|---|
| tag freq P50 | 1e-4 ~ 1e-3 |
| tag freq P99 | 0.05 ~ 0.10 |
| tag freq max | ~1.0 |
| per-doc tag count mean | 400-800 |
| per-doc tag count P99 | 2000-3000 |
| AND-expr Sel P50 | 0.378 |
| AND-expr Sel P99 | 0.056 |
| All0 桶 P50 | 27 |
| sentinel folding rate | >30% |
