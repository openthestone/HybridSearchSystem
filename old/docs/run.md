# 运行文档

## 环境

- **机器**：ARM64 + 昇腾 910B NPU
- **CANN**：默认找 `/usr/local/Ascend/ascend-toolkit/latest` 或 `/home/lcy/Ascend/ascend-toolkit/latest`
- **依赖**：CMake 3.16+、OpenMP、folly、glog、gflags、Eigen、BLAS、fm

```bash
# 可选：手动指定
export ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
export SOC_VERSION=Ascend910B3   # B1/B2/B3/B4
```

---

## 基本命令

```bash
cd /root/xihe_v1

./run.sh -v s -p 0 -c 1  # 编译 + 运行 serial（调试友好）

./run.sh -v p -p 0 -c 1  # 编译 + 运行 parallel（吞吐）

./run.sh -v s -p 0 -c 0  # 已编译，跳过编译
./run.sh -v p -p 0 -c 0
```

| 参数               | 含义                 |
| ---------------- | ------------------ |
| `-v s`           | serial             |
| `-v p`           | parallel           |
| `-v a1/a2/a3/a8` | analyze 工具         |
| `-c 1`           | 先编译                |
| `-c 0`           | 不编译，直接跑 `out/bin/` |
| `-p 1`           | perf 采样 + 火焰图      |

---

## 运行流程（`run.sh` 内部）

1. 检测 NPU / CANN / SOC
2. `-c 1`：cmake 编译 → 安装到 `out/`
3. 部署 vendor kernel 到 CANN
4. 清空 `result/`（config 里 `query_result_root`）
5. `cd out/bin/` 执行二进制

产物：

- 二进制：`out/bin/serial`、`parallel` 等
- 配置：`out/bin/config.txt`（从根目录 install）
- 结果：`result/recall/`、`result/latency/`、`result/log/`

---

## 运行前数据

程序 cwd 是 `out/bin/`，config 里 `../../` 指向项目根 `/root/xihe_v1/`。

**必需（当前环境）：**


| 文件                           | 状态                                                   |
| ---------------------------- | ---------------------------------------------------- |
| `dataset_HW.bin`             | ✅                                                    |
| `bucket_index.bin`           | ✅                                                    |
| `bucket_ivf_index.bin`       | 需有                                                   |
| `centroids.bin` + `buckets/` | ✅                                                    |
| `l2_to_l0_map.bin`           | 需有                                                   |
| `tag_freq_cache.bin`         | 需有                                                   |
| `filter_expr_600.txt`        | ✅                                                    |
| `datasets/hw_queries.fvecs`  | ✅（主 `src/` 硬编码读这个，不是 config 的 `QueryData_10000.txt`） |

**可选：**

- `ground_truth_cache.bin` — recall 对照；不要 recall 可在 `config.txt` 设 `ground_truth_cache_file = ""`

---

## 配置

改 `/root/xihe_v1/config.txt`，然后：

```bash
./run.sh -v s -p 0 -c 1   # 重新 install config + 编译
# 或只改 config、不编译：
cp config.txt out/bin/config.txt
./run.sh -v s -p 0 -c 0
```

关键项：`npu_device_id_start`、`cpu_core_count_*`、`valid_bucket_num_*`、`query_result_root`。

---

## 看结果

```bash
ls result/latency/ result/recall/ result/log/

# 可选后处理
python3 analyze_latency_buckets.py
python3 analyze_result_p99.py
```

---

## 常见问题

| 现象                   | 处理                       |
| -------------------- | ------------------------ |
| CANN not found       | 设 `ASCEND_INSTALL_PATH`  |
| npu-smi 失败           | 查 NPU 驱动/设备              |
| 缺 .so                | 用 `-c 1` 重编，确认 vendor 部署 |
| Executable not found | 先 `-c 1` 编译              |

当前 `out/bin/` 已有编译产物，可直接：

```bash
cd /root/xihe_v1 && ./run.sh -v s -p 0 -c 0
```
