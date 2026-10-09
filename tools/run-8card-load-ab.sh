#!/usr/bin/env bash
# 八卡 / 4 组 / 每组 2 分片：固定任务分配 vs 动态分配，有负载与无负载，BS=2 和 4。
# 臂的顺序排成回文 dyn/static/static/dyn，每条臂两个样本，抵消机器的线性漂移。
# 负载走 NPU_LOAD_CARDS（fr_npuload 制造真实的 HBM/Cube 争用），不能和 SLOW_CARDS 同时开
# （run.sh:213）。负载只压一个组：四个组一样慢的话动态无处可调。
set -uo pipefail

ROOT="${ROOT:?ROOT 未设置}"
R="${R:?R 未设置}"
DATA="${DATA:?DATA 未设置}"
Q="${Q:?Q 未设置}"
W="${W:-$R/den0.00002/work}"          # 索引所在的 WORK_DIR
T="${T:-0.00002}"                      # 密度阈值
DEVS="${DEVS:-0,1,2,3,4,5,6,7}"
GROUP="${GROUP:-2}"                    # 每组分片数
LOAD_CARDS="${LOAD_CARDS:-0,1}"        # 压哪几张卡；默认组 0 的两张
BATCHES="${BATCHES:-2 4}"
REPEAT="${REPEAT:-3}"
WARMUP="${WARMUP:-5}"
PIN="${PIN:-}"                         # 留空不 pin；八卡跨 NUMA，先看 lscpu 再决定
OUT="${OUT:-$R/ab8card.log}"

NDEV=$(awk -F, '{print NF}' <<< "$DEVS")
IFS=',' read -r -a _sh <<< "$DEVS"

die() { echo "[ERROR] $*" >&2; exit 1; }
[[ -d "$W/index/shard0" && -d "$W/index/shard1" ]] || die "$W/index 下找不到 shard0/shard1"
(( NDEV % GROUP == 0 )) || die "设备数 $NDEV 不是 GROUP=$GROUP 的整数倍"
[[ -x "$ROOT/build/fr_npuload" ]] || die "缺 $ROOT/build/fr_npuload，先跑一次 --compile 1"
for c in ${LOAD_CARDS//,/ }; do
  [[ ",$DEVS," == *",$c,"* ]] || die "LOAD_CARDS 里的 $c 不在 DEVICE_IDS 内，压的不是引擎在用的卡"
done
mkdir -p "$(dirname "$OUT")"

echo "设备 $DEVS  分 $((NDEV / GROUP)) 组 × $GROUP 分片   索引 $W   阈值 $T"
echo "负载卡 $LOAD_CARDS   batch $BATCHES   每次 --repeat $REPEAT"
echo "输出 $OUT"
echo

one_run() {
  local load=$1 bs=$2 arm=$3 k=$4 static=0 cards=""
  [[ "$arm" == static ]] && static=1
  [[ "$load" == 1 ]] && cards="$LOAD_CARDS"
  echo "########## load=$load bs=$bs arm=$arm rep=$k ##########"
  env RAPIDJSON_INCLUDE_DIR=/usr/include \
      DATASET_FILE="$DATA" DEVICE_IDS="$DEVS" \
      DOCS=10000000 DENSITY_THRESHOLD="$T" DOC_NUM_PER_SEGMENT=131072 \
      WORK_DIR="$W" RESULT_DIR="$R/ab8_${load}_${bs}_${arm}_${k}" \
      QUERY_OUT_DIR="$Q" NUM_QUERIES=0 \
      SHARD_INDEX_DIRS="$W/index/shard0,$W/index/shard1" SHARD_GROUP_SIZE="$GROUP" \
      STATIC_ASSIGN="$static" NPU_LOAD_CARDS="$cards" SLOW_CARDS= \
      NPU_TOPK_LOOP_COUNT=48 SHARD_TOPK_RATIO=0.85 \
      STREAM_MERGE=1 PACKED_SORT=1 RADIX_SORT=1 \
      PIPELINE=1 SHARD_WORKER_POOL=1 GROUP_EXTRACT_PARALLEL=1 GROUP_PREFETCH=1 \
      NPUR_OVERLAP_POSTING=1 BATCH_AGGREGATE=1 BATCH_FILTER=1 SCORER_TRIM_WRITE=1 \
      BITLIST_ONE_H2D=1 BITLIST_POSTING_MAJOR=1 \
      PARALLEL_LOAD=1 WARMUP="$WARMUP" BATCH_SIZE="$bs" \
      ${PIN:+taskset -c "$PIN"} "$ROOT/run.sh" \
        --compile 0 --convert-query 0 --convert-data 0 \
        --build-index 0 --mem-report 1 --repeat "$REPEAT"
}

cd "$ROOT" || die "进不去 $ROOT"
{
  for load in 0 1; do
    for bs in $BATCHES; do
      k=0
      for arm in dyn static static dyn; do        # 回文
        k=$((k + 1))
        one_run "$load" "$bs" "$arm" "$k"
      done
    done
  done
} 2>&1 | tee "$OUT"

echo
echo "================ 汇总 ================"
OUT="$OUT" python3 - <<'PY'
import os, re, collections
cells = collections.defaultdict(lambda: collections.defaultdict(list))
key = None
for line in open(os.environ["OUT"], errors="replace"):
    m = re.match(r"#+ load=(\d) bs=(\d+) arm=(\w+) rep=(\d+) #+", line)
    if m:
        key = (int(m.group(1)), int(m.group(2)), m.group(3)); continue
    m = re.match(r"(p50|p99|qps)\s+(.*)", line)
    if m and key:
        f = m.group(2).split()
        if f:
            cells[key[:2]].setdefault(key[2], {}).setdefault(m.group(1), []).append(float(f[-1]))

def mean(v): return sum(v) / len(v) if v else float("nan")
for (load, bs) in sorted(cells):
    arms = cells[(load, bs)]
    missing = [a for a in ("dyn", "static")
               if a not in arms or not all(arms[a].get(k) for k in ("p50", "p99", "qps"))]
    if missing:
        print("\nload=%d BS=%d  %s 这一臂没有完整的 p50/p99/qps，跳过（那次跑挂了？看日志）"
              % (load, bs, "/".join(missing)))
        continue
    print("\nload=%d  BS=%d" % (load, bs))
    print("  %-8s %9s %9s %9s" % ("", "p50", "p99", "qps"))
    for a in ("static", "dyn"):
        print("  %-8s %9.4f %9.4f %9.1f"
              % (a, mean(arms[a].get("p50", [])), mean(arms[a].get("p99", [])),
                 mean(arms[a].get("qps", []))))
    d, s = arms["dyn"], arms["static"]
    print("  %-8s %+8.2f%% %+8.2f%% %+8.2f%%   <- 动态相对固定"
          % ("变化",
             100 * (mean(d["p50"]) / mean(s["p50"]) - 1),
             100 * (mean(d["p99"]) / mean(s["p99"]) - 1),
             100 * (mean(d["qps"]) / mean(s["qps"]) - 1)))
    # 同一条臂的两个样本之间差多少，就是这一格能分辨的下限。效应比它小就不算数。
    spread = []
    for a in ("static", "dyn"):
        for k in ("p50", "p99", "qps"):
            v = arms[a].get(k, [])
            if len(v) >= 2 and mean(v):
                spread.append(abs(v[0] - v[-1]) / mean(v))
    if spread:
        print("  %-8s %8.2f%%   <- 同臂两样本的最大差；效应必须大过它"
              % ("分辨率", 100 * max(spread)))
PY
