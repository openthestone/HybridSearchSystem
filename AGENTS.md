## 项目简介

这个任务是一个带复杂布尔属性过滤的高维向量 Top-K 检索优化问题。数据集规模约为一千万条记录，每条记录由一个 64 维浮点向量和一个约 3 万多个标签属性组成的 0/1 位图向量构成。每条查询也包含两部分：一个 64 维查询向量，以及一个由标签通过 AND、OR、NOT 组合而成的复杂过滤表达式。系统需要先根据过滤表达式筛出满足条件的文档，再在这些文档中找到与查询向量距离最近的前 3500 条结果。最终验收目标是平均召回率不低于 99%，同时 P99 查询延迟控制在 2 毫秒以内。

## 项目目录

该项目下主要有两个代码目录，分别是 old/ 和 src/。

old/ 是我们自己实现的一个原始版本，延迟较高。整体思路是离线分桶建索引 + 在线粗筛精排。该代码主要用于思路参考。

src/ 是目前实现的一个新版本，src/engine 由华为编写，由于输入数据格式不同、缺少完整链路，我们编写了一些适配代码，使其能成功运行。

## 项目环境

运行该项目需要 NPU 服务器，目前工作环境在本地开发环境（并无 NPU），项目代码通过 Git 在本地开发环境和 NPU 服务器上同步。因此，不可以在本地直接运行项目。我会在 NPU 服务器运行代码，你不需要尝试在本地运行代码。

## 运行实验

实验一律用 dtach 后台跑，日志由脚本内部重定向。不要用 `dtach ... | tee log`：断开后管道断掉，日志就不再增长了。

```shell
dtach -n /tmp/<name>.dtach bash -c '<script> > <log> 2>&1'
tail -f <log>                 # 看进度
dtach -a /tmp/<name>.dtach    # 连回会话，Ctrl+\ 断开（Ctrl+C 会杀掉任务）
```

每次换脚本内容就换个 socket 名，避免撞上 `Address already in use`，也避免覆盖到正在运行的脚本（bash 边读边执行）。

运行时要添加 NOTIFY 已打开通知。

## Git Commit Convention

Format: `type: subject` (subject starts with lowercase, one concise sentence only)

- `feat`: New or modified feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting, whitespace
- `refactor`: Code restructure
- `perf`: Performance improvement
- `test`: Adding tests
- `chore`: Build tools, maintenance
- `revert`: Revert a previous commit

Examples:

- `feat: add session open/close lines toggle`
- `fix: disable fetch cache to get fresh data`
- `chore: upgrade lightweight-charts to v5`

# Code Style

```shell
scripts/format.sh  # format staged files
scripts/format.sh --all  # format all files
```

# 运行代码

这是一段基本的运行该项目的代码，以供参考：

```
dtach -n /tmp/seg131k_pin.dtach bash -c 'exec > /root/xihe_v1/seg131k.log 2>&1
set -e
cd /root/xihe_v1

R=/root/xihe_v1/frnew_syn_v2/runs
RAPIDJSON_INCLUDE_DIR=/usr/include \
DATASET_FILE=/root/xihe_v1/frnew_syn_v2/dataset.bin \
DEVICE_IDS=0,1 DOCS=10000000 \
DOC_NUM_PER_SEGMENT=131072 DENSITY_THRESHOLD=0 \
WORK_DIR=$R/seg131072_2c/work \
RESULT_DIR=$R/seg131072_2c/result \
QUERY_OUT_DIR=$R/seg131072_den0/work/queries \
NUM_QUERIES=0 \
RECALL_REF_FILE=$R/shard2_den0/recall_ref.bin RECALL_REF_THREADS=0 \
NPU_TOPK_LOOP_COUNT=48 \
SHARD_TOPK_RATIO=0.6 \
STREAM_MERGE=1 PACKED_SORT=1 RADIX_SORT=1 NPUR_OVERLAP_POSTING=1 \
NOTIFY=1 \
BATCH_SIZE=16 WARMUP=25 \
taskset -c 0-23 ./run.sh --compile 1 --convert-query 0 --convert-data 0 --build-index 0 --profile 2 --repeat 3'
```

# STREAM_MERGE=1 PACKED_SORT=1 RADIX_SORT=1 NPUR_OVERLAP_POSTING=1 \

```
dtach -n /tmp/seg262k.dtach bash -c 'exec > /root/xihe_v1/seg262k.log 2>&1
set -e
cd /root/xihe_v1

R=/root/xihe_v1/frnew_syn_v2/runs
RAPIDJSON_INCLUDE_DIR=/usr/include \
DATASET_FILE=/root/xihe_v1/frnew_syn_v2/dataset.bin \
DEVICE_IDS=0,1 DOCS=10000000 \
DOC_NUM_PER_SEGMENT=262144 DENSITY_THRESHOLD=0 \
WORK_DIR=$R/seg262144_2c/work \
RESULT_DIR=$R/seg262144_2c/result \
QUERY_OUT_DIR=$R/seg131072_den0/work/queries \
NUM_QUERIES=0 \
RECALL_REF_FILE=$R/shard2_den0/recall_ref.bin RECALL_REF_THREADS=0 \
NPU_TOPK_LOOP_COUNT=48 \
SHARD_TOPK_RATIO=0.6 \
STREAM_MERGE=1 PACKED_SORT=1 RADIX_SORT=1 NPUR_OVERLAP_POSTING=1 \
NOTIFY=1 \
BATCH_SIZE=1 WARMUP=25 \
taskset -c 0-23 ./run.sh --compile 0 --convert-query 0 --convert-data 0 --build-index 0 --profile 2 --repeat 3'
```
