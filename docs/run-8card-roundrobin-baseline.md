# 八卡 Round-Robin Baseline 运行文档

对照 baseline：每卡持全量语料，查询轮询发到空闲卡，各查询整条在一张卡上跑完（无 merge）。开关 `ROUND_ROBIN=1`。

## 填这几个

```bash
ROOT=/path/to/xihe_v1
DATA=$ROOT/xxx/dataset.bin
Q=$ROOT/xxx/queries
OUT=$ROOT/xxx/runs/8card_rr
TOTAL=10000000
FULL=$OUT/full/work/index             # 全量（非分片）索引，见下一步
PIN=0-23
```

## 建全量索引（一次性，单卡模式，不设 DEVICE_IDS）

```bash
cd $ROOT
RAPIDJSON_INCLUDE_DIR=/usr/include \
DATASET_FILE=$DATA \
DOCS=$TOTAL DENSITY_THRESHOLD=0 DOC_NUM_PER_SEGMENT=131072 \
WORK_DIR=$OUT/full/work RESULT_DIR=$OUT/full/result \
QUERY_OUT_DIR=$Q NUM_QUERIES=0 \
./run.sh --compile 1 --convert-query 0 --convert-data 1 --build-index 1 --profile 0 --repeat 1
```

## Round-robin 搜索（8 卡，时延 BS=1）

```bash
cd $ROOT
RAPIDJSON_INCLUDE_DIR=/usr/include \
DATASET_FILE=$DATA \
DEVICE_IDS=0,1,2,3,4,5,6,7 \
DOCS=$TOTAL DENSITY_THRESHOLD=0 DOC_NUM_PER_SEGMENT=131072 \
WORK_DIR=$OUT/work RESULT_DIR=$OUT/result \
QUERY_OUT_DIR=$Q NUM_QUERIES=0 \
RECALL_REF_FILE=$OUT/recall_ref.bin RECALL_REF_THREADS=0 \
SHARD_INDEX_DIRS=$FULL \
NPU_TOPK_LOOP_COUNT=48 SHARD_TOPK_RATIO=1.0 \
ROUND_ROBIN=1 WARMUP=5 BATCH_SIZE=1 \
taskset -c $PIN ./run.sh --compile 0 --convert-query 0 --convert-data 0 --build-index 0 --profile 0 --repeat 3
```

## 测吞吐（BS=16）

同上，仅改：`BATCH_SIZE=16`、`RESULT_DIR=$OUT/bs16`。
