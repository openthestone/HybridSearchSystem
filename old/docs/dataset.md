## 数据集

**HW 检索数据集** — 业务向量 `relevance_learning2rank`，64 维，**1000 万**文档，**35840** 个属性 tag。带语法过滤表达式。

来源路径（`config.txt`）指原始生产数据：
- 向量：`.../dim_64_4.17/.../section.relevance_softLtrAfmConditionv1.0`
- 属性：`.../dim_64_4.17/.../inverted_union`
- 查询 JSON：`.../falcon_request_new.txt`

本地跑 benchmark 用的是**预处理后缓存**，不是直接读上面原始路径。

---

## 文件在哪

全部在 `/root/xihe_v1/`：

| 文件 | 路径 | 大小 | 用途 |
|------|------|------|------|
| 主库 | `dataset_HW.bin` | 45G | 向量+属性缓存（`LoadDatasetCache`） |
| 查询 | `datasets/hw_queries.fvecs` | 2.5M | **serial/parallel 实际读这个** |
| 过滤 | `filter_expr_600.txt` | 51K | 10 条 filter，随机分给 10000 query |
| GT | `ground_truth_cache.bin` | 70M | recall 对照（19679 entries） |
| 聚类 | `centroids.bin` + `buckets/` | 3.4M + 13738 桶 | L1/L2 桶布局 |
| 索引 | `bucket_index.bin` | 45G | 桶内 doc 索引 |
| IVF | `bucket_ivf_index.bin` | 141M | 桶级 IVF |
| 映射 | `l2_to_l0_map.bin` | 54K | L2→L0 |
| 频率 | `tag_freq_cache.bin` | 141K | 全局 tag 频率 |

---

## 运行时实际加载

终端日志已确认：

```
Docs=10000000, Dim=64, Tags=35840, Buckets(Level1)=8192
Queries loaded from ../../datasets/hw_queries.fvecs
Final query count: 10000
Assigned 10 filter expressions to 10000 queries
```

cwd 是 `out/bin/`，所以 `../../` = 项目根。

---

### 查询 `datasets/hw_queries.fvecs`

保存的是 10000 条查询向量，用于对 `dataset_HW.bin` 里的 1000 万文档向量做检索。

```text
uint32 n_queries = 10000  # 前 4 字节，查询数量
uint32 dim       = 64  # 接下来 4 字节，每条向量维度
float32 vectors[10000][64]  # 后面全部是 float32 向量值
```
