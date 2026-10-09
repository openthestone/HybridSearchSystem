#!/usr/bin/env bash
# 一条命令跑一组对照实验，代替每次手敷二十多行 env。
#
#   tools/exp.sh latency   按给定配置跑一遍，报时延。
#                          CARD_SETS / BATCH_SIZES 给多个值则跑交叉矩阵，出一张表：
#                            tools/exp.sh latency CARD_SETS="0,1 0,1,2,3" BATCH_SIZES="1 2 4"
#                          MODE=serial|groups|group_pipeline|pipeline|round_robin
#                          默认 group_pipeline
#                          CARDS_PER_GROUP=<每组几张卡>  DEVICE_IDS=<用哪几张卡>
#                          MODE=round_robin 另需 FULL_INDEX_DIR=<每卡一份的完整索引>
#   tools/exp.sh load      固定任务分配 vs 动态分配，无负载/有负载，看吞吐
#   tools/exp.sh index     索引加载时间（含每分片）+ 大小按后缀拆分。单次运行。
#                          PARALLEL_LOAD=0|1（默认 1）
#   tools/exp.sh size      只量索引大小，不跑检索；给两个目录出百分比
#   tools/exp.sh build     建索引：转查询 + 转数据 + 建库，完事打大小。已有索引要 FORCE=1
#                          SHARDS=<把语料切几份>  DENSITY_THRESHOLD=<阈值>  DOCS=<文档数>
#                          CONVERT_QUERY 默认 1，要 tag_map.bin 和 QueryData_10000.txt；
#                          已经转好过就 CONVERT_QUERY=0 跳过（查询和阈值无关）
#                          RECALL_REF=1 顺带把召回参考缓存建好（很慢，只需一次，之后所有场景命中）
#
# 前台跑，输出到 stdout。要后台跑就在外面套 dtach，日志由重定向落地（别用 dtach ... | tee，
# 断开后管道就断了）：
#   dtach -n /tmp/exp-latency.dtach bash -c 'tools/exp.sh latency > logs/latency.log 2>&1'
#
# 默认指向校内两卡的 v3；覆盖用的变量名和 run.sh 一致，写在前面写在后面都行：
#   DEVICE_IDS=0,1,2,3,4,5,6,7 CARDS_PER_GROUP=2 tools/exp.sh load
#   tools/exp.sh load DEVICE_IDS=0,1,2,3,4,5,6,7 CARDS_PER_GROUP=2
#
# load 是对照实验：两种配置的先后顺序排成回文（A B B A），抵消机器的线性漂移；同一种配置
# 两个样本之间的差就是这一格能分辨的下限，效应比它小就不算数。这两条来自
# tools/run-8card-load-ab.sh，别改。latency / index 是单次测量，不适用。
set -uo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCENARIO="${1:-}"; [[ $# -gt 0 ]] && shift
[[ -n "${SCENARIO}" ]] || { sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1; }

# 变量写在场景名后面也行：tools/exp.sh build DENSITY_THRESHOLD=0，和写在前面等价。
# 名册里有几个（HBM_SAMPLE / NPU_UTIL / TARGET_QPS / SLOW_CARDS）本脚本自己不用，是 run.sh
# 从环境里读的；env 不带 -i，所以导出后会继承下去。
# 不是赋值的（比如 size 的目录）原样留给后面。名字必须在册——打错一个字母却静默用默认值，
# 会让你以为建的是阈值 0、其实不是。
KNOBS="ROOT DATA_ROOT RUNS_DIR DENSITY_THRESHOLD WORK_DIR DATASET_FILE QUERY_OUT_DIR
       DEVICE_IDS DOCS DOC_NUM_PER_SEGMENT NUM_QUERIES NUM_QUERIES_INDEX WARMUP
       RECALL_QUERIES MEM_REPORT RECALL_REF_FILE CARDS_PER_GROUP INDEX_DIRS FULL_INDEX_DIR
       TASKSET_CPUS CARD_CPUS LOG_DIR PARSE_LOG ROUNDS NOTIFY NPU_LOAD_CARDS DROP_CACHES
       MODE ARMS BATCH_SIZE BATCH_SIZES CARD_SETS REPEAT SHARDS FORCE COMPILE
       CONVERT_QUERY PARALLEL_LOAD RECALL_REF
       HBM_SAMPLE NPU_UTIL TARGET_QPS SLOW_CARDS TAG_MAP_FILE QUERYDATA_FILE
       NPU_LOAD_DUTY NPU_LOAD_PERIOD_MS NPU_LOAD_COMPUTE_WORKERS NPU_LOAD_COPY_WORKERS
       NPU_LOAD_COPY_GAP_MS NPU_LOAD_COPY_BUFFER_MB
       RAPIDJSON_INCLUDE_DIR"
REST=()
for arg in "$@"; do
    if [[ "${arg}" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
        name="${arg%%=*}"
        [[ " ${KNOBS//[$'\n'] / } " == *" ${name} "* ]] \
            || { echo "[ERROR] 不认识的变量：${name}。可设的是：" >&2
                 echo "${KNOBS}" | sed 's/^ */  /' >&2; exit 1; }
        export "${arg}"
    else
        REST+=("${arg}")
    fi
done
set -- ${REST[@]+"${REST[@]}"}

# ---- 路径与规模。能和 run.sh 同名的就同名，省掉一层翻译 ------------------------
DATA_ROOT="${DATA_ROOT:-${ROOT}/frnew_syn_v3}"
RUNS_DIR="${RUNS_DIR:-${DATA_ROOT}/runs}"
DENSITY_THRESHOLD="${DENSITY_THRESHOLD:-0.00026}"
WORK_DIR="${WORK_DIR:-${RUNS_DIR}/den${DENSITY_THRESHOLD}/work}"
DATASET_FILE="${DATASET_FILE:-${DATA_ROOT}/dataset.bin}"
QUERY_OUT_DIR="${QUERY_OUT_DIR:-${RUNS_DIR}/den0/work/queries}"
DEVICE_IDS="${DEVICE_IDS:-0,1}"
DOCS="${DOCS:-10000000}"
DOC_NUM_PER_SEGMENT="${DOC_NUM_PER_SEGMENT:-131072}"
NUM_QUERIES="${NUM_QUERIES:-0}"
WARMUP="${WARMUP:-5}"
RECALL_QUERIES="${RECALL_QUERIES:-20}"
MEM_REPORT="${MEM_REPORT:-0}"
# 显式置空则不用召回缓存（用 ${VAR-默认} 而不是 ${VAR:-默认}，空值才留得住）
# 显式给的路径即使文件不存在也照用（那是"建缓存"的意思）；只有默认路径缺文件才丢掉。
RECALL_REF_GIVEN=0; [[ -n "${RECALL_REF_FILE:-}" ]] && RECALL_REF_GIVEN=1
RECALL_REF_FILE="${RECALL_REF_FILE-${RUNS_DIR}/recall_ref.bin}"

# ---- 各场景要改写的东西，必须和上面的 run.sh 同名变量分开 ----------------------
# INDEX_DIRS 是这里配置的分片索引，写进 run.sh 的 SHARD_INDEX_DIRS；
# CARDS_PER_GROUP 是这里配置的每组卡数，写进 run.sh 的 SHARD_GROUP_SIZE。
# 同名会分不清"这里配的值"和"这一次实际传的值"：MODE=serial 就是要传 0 而不是配置里的 2。
CARDS_PER_GROUP="${CARDS_PER_GROUP:-2}"
# 默认列一个组的分片。分组路径下 run.sh 会把这一组复制到每个组（run.sh:188），serial /
# pipeline 则要求条数等于卡数，对不上 run.sh:440 会直接报错。
shard_dir_list() {   # <几份>
    local i out=""
    for ((i = 0; i < $1; i++)); do out+="${out:+,}${WORK_DIR}/index/shard${i}"; done
    echo "${out}"
}
INDEX_DIRS_GIVEN=0; [[ -n "${INDEX_DIRS:-}" ]] && INDEX_DIRS_GIVEN=1
INDEX_DIRS="${INDEX_DIRS:-$(shard_dir_list "${CARDS_PER_GROUP}")}"

# 这一次该用几份分片索引：serial / pipeline 每张卡一份，分组路径只要一个组的份数。
# 显式给了 INDEX_DIRS 就照用，不猜。
index_dirs_for() {   # <卡数>
    if (( INDEX_DIRS_GIVEN )); then
        echo "${INDEX_DIRS}"
    elif [[ "${MODE:-}" == "serial" || "${MODE:-}" == "pipeline" ]]; then
        shard_dir_list "$1"
    else
        shard_dir_list "${CARDS_PER_GROUP}"
    fi
}
FULL_INDEX_DIR="${FULL_INDEX_DIR:-}"       # MODE=round_robin 用：一份完整语料的索引

# ---- 实验本身的参数 ----------------------------------------------------------
TASKSET_CPUS="${TASKSET_CPUS-0-23}"        # 线程迁移是本机 p99 波动的首因；八卡跨 NUMA 时置空关掉
# 每卡一份核列表（';' 分隔，对应 DEVICE_IDS）。卡不共享 NUMA 节点，进程级 taskset 对至少
# 一张卡是远程的（cpu_affinity.cpp:18）。run.sh 从环境读，这里只为透传。
CARD_CPUS="${CARD_CPUS:-}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
ROUNDS="${ROUNDS:-2}"                      # load 档每种配置跑几个样本
NOTIFY="${NOTIFY:-0}"                      # 1=只在最后一次跑完发一封
NPU_LOAD_CARDS="${NPU_LOAD_CARDS:-}"       # load 档用；空则取 0 号卡，见下
DROP_CACHES="${DROP_CACHES:-0}"            # 1=每次运行前清页缓存，量冷加载用；清的是整机的，要 root
CARD_COUNT="$(awk -F, '{print NF}' <<< "${DEVICE_IDS}")"

die() { echo "[ERROR] $*" >&2; exit 1; }

# 已调好的开关，各场景共用；每次运行自己的赋值追加在后面覆盖它（env 里后一个赋值胜出）。
# SHARD_GROUP_SIZE / PIPELINE / ROUND_ROBIN / STATIC_ASSIGN / PARALLEL_LOAD 故意不在这里，
# 它们正是各场景要对比的东西。
BASE_SWITCHES=(
    STREAM_POOL=1 POOL_SMALL_H2D=1 TOPK_COUNTS_IN_PLACE=1 EARLY_FILTER_PREP=1
    SHARE_FILTER_STREAM=1 OVERLAP_FILTER=1 EXTRACT_FAST=1 FUSE_AGG_TOPK=1
    NPU_TOPK_LOOP_COUNT=48 SHARD_TOPK_RATIO=0.85 STREAM_MERGE=1 PACKED_SORT=1 RADIX_SORT=1
    SHARD_WORKER_POOL=1 GROUP_EXTRACT_PARALLEL=1 GROUP_PREFETCH=1
    NPUR_OVERLAP_POSTING=1 BATCH_AGGREGATE=1 BATCH_FILTER=1 SCORER_TRIM_WRITE=1
    BITLIST_ONE_H2D=1 BITLIST_POSTING_MAJOR=1 PARALLEL_LOAD=1
)

arm_env() {   # 配置名 -> 要覆盖的赋值。调度三个开关一律显式写全，不靠默认值。
    case "$1" in
        serial)            echo "SHARD_GROUP_SIZE=0 PIPELINE=0 ROUND_ROBIN=0" ;;
        groups)            echo "SHARD_GROUP_SIZE=${CARDS_PER_GROUP} PIPELINE=0 ROUND_ROBIN=0" ;;
        group_pipeline)    echo "SHARD_GROUP_SIZE=${CARDS_PER_GROUP} PIPELINE=1 ROUND_ROBIN=0" ;;
        pipeline)          echo "SHARD_GROUP_SIZE=0 PIPELINE=1 ROUND_ROBIN=0" ;;
        # 轮转没有跨卡 merge，ratio<1 会直接砍召回，run.sh:151 也会强制回 1.0；这里显式写上。
        round_robin)       echo "SHARD_GROUP_SIZE=0 PIPELINE=0 ROUND_ROBIN=1 SHARD_TOPK_RATIO=1.0 SHARD_INDEX_DIRS=${FULL_INDEX_DIR}" ;;
        fixed)             echo "STATIC_ASSIGN=1" ;;
        dynamic)           echo "STATIC_ASSIGN=0" ;;
        *) return 1 ;;
    esac
}

# ---- 索引大小：按后缀拆 -------------------------------------------------------
# builder 写出这四种（src/engine/core/constant_definition.h:20-23）：.posting 是倒排、.vector 是
# 向量、.idm 是 docid 映射、.meta 是元信息。密度阈值只动 .posting，所以"索引小了多少"看那一栏。
# du -b 报的是 apparent size，和 find -printf %s 的加和同一个尺度，两边能对上。
size_row() {   # <名字> <字节> <该目录总字节>
    # 名字放行尾：中文标签按字节补宽会错位，放最后就无所谓了
    printf '  %12s  %6s   %s\n' "$(numfmt --to=iec --suffix=B "$2")" \
        "$(awk -v n="$2" -v t="$3" 'BEGIN{printf "%.1f%%", t ? 100*n/t : 0}')" "$1"
}

size_report() {   # <目录...>  字节数留在全局 SIZE_BYTES / SIZE_TOTAL
    local dir total_bytes suffix suffix_bytes accounted
    du -sb . >/dev/null 2>&1 || die "需要 GNU du 的 -sb（apparent size）；这台机器上没有"
    command -v numfmt >/dev/null || die "需要 numfmt（GNU coreutils）"
    SIZE_BYTES=(); SIZE_TOTAL=0
    for dir in "$@"; do
        [[ -d "${dir}" ]] || die "不是目录：${dir}"
        total_bytes="$(du -sb "${dir}" | cut -f1)"
        SIZE_BYTES+=("${total_bytes}"); SIZE_TOTAL=$((SIZE_TOTAL + total_bytes))
        printf '  %12s  %s\n' "$(numfmt --to=iec --suffix=B "${total_bytes}")" "${dir}"
        accounted=0
        for suffix in posting vector idm meta; do
            suffix_bytes="$(find "${dir}" -type f -name "*.${suffix}" -printf '%s\n' 2>/dev/null \
                            | awk '{s+=$1} END {printf "%d", s+0}')"
            (( suffix_bytes == 0 )) && continue
            accounted=$((accounted + suffix_bytes))
            size_row ".${suffix}" "${suffix_bytes}" "${total_bytes}"
        done
        # 差额单列出来，免得四种后缀之外的东西被静悄悄漏掉
        (( total_bytes > accounted )) && size_row "其他" "$((total_bytes - accounted))" "${total_bytes}"
    done
    (( $# > 1 )) && printf '  %12s  合计 %d 个目录\n' \
        "$(numfmt --to=iec --suffix=B "${SIZE_TOTAL}")" "$#"
    return 0
}

# ---- size：不跑检索 -----------------------------------------------------------
if [[ "${SCENARIO}" == "size" ]]; then
    dirs=("${@:-}"); [[ -n "${dirs[0]:-}" ]] || IFS=',' read -r -a dirs <<< "${INDEX_DIRS}"
    size_report "${dirs[@]}"
    # 两个目录时给出比值，用来对阈值/布局改动前后的索引大小（验收要求小 20%）
    (( ${#dirs[@]} == 2 )) && awk -v a="${SIZE_BYTES[0]}" -v b="${SIZE_BYTES[1]}" \
        'BEGIN{printf "\n第二个相对第一个 %+.2f%%\n", 100*(b/a-1)}'
    exit 0
fi

# ---- build：转换 + 建库 --------------------------------------------------------
if [[ "${SCENARIO}" == "build" ]]; then
    SHARDS="${SHARDS:-${CARDS_PER_GROUP}}"
    (( SHARDS >= 1 )) || die "SHARDS 要 >= 1"
    (( DOCS > 0 )) || die "DOCS 要 > 0：run.sh 按 DOCS 把语料切给各分片（run.sh:215）"
    INDEX_ROOT="${WORK_DIR}/index"
    COMPILE="${COMPILE:-0}"
    if [[ "${COMPILE}" != "1" ]]; then
        for b in fr_converter fr_builder fr_search; do
            [[ -x "${ROOT}/build/${b}" ]] || die "缺 ${ROOT}/build/${b}，加 COMPILE=1 连编译一起做"
        done
    fi
    [[ -f "${DATASET_FILE}" ]] || die "数据集不存在：${DATASET_FILE}"
    CONVERT_QUERY="${CONVERT_QUERY:-1}"
    if [[ "${CONVERT_QUERY}" == "1" ]]; then
        # 真实数据这两个文件在仓库根（run.sh 的默认），合成数据由 gen_synthetic_data_v3.py 写在
        # DATA_ROOT 下、查询文件叫 QueryData.txt。哪边有就用哪边，并打出选中的路径。
        pick() { local f; for f in "$@"; do [[ -f "${f}" ]] && { echo "${f}"; return 0; }; done; return 1; }
        TAG_MAP_FILE="${TAG_MAP_FILE:-$(pick "${ROOT}/tag_map.bin" "${DATA_ROOT}/tag_map.bin")}"
        QUERYDATA_FILE="${QUERYDATA_FILE:-$(pick "${ROOT}/QueryData_10000.txt" \
            "${DATA_ROOT}/QueryData.txt" "${DATA_ROOT}/QueryData_10000.txt")}"
        # run.sh 也会检查，但那是在编译之后——COMPILE=1 时等编译完才报就太晚了
        [[ -f "${TAG_MAP_FILE}" && -f "${QUERYDATA_FILE}" ]] || die "$(cat <<EOT
转换查询要 tag_map.bin 和 QueryData 文本，这些位置都没有：
  ${ROOT}/tag_map.bin          ${ROOT}/QueryData_10000.txt
  ${DATA_ROOT}/tag_map.bin     ${DATA_ROOT}/QueryData.txt
在别处就指过去：TAG_MAP_FILE=... QUERYDATA_FILE=...
已经转好过、不想再转：CONVERT_QUERY=0
EOT
)"
        export TAG_MAP_FILE QUERYDATA_FILE
        echo "[转换查询] ${QUERYDATA_FILE} + ${TAG_MAP_FILE}"
    elif [[ ! -f "${QUERY_OUT_DIR}/queries.fvecs" ]]; then
        # run.sh 在转换阶段之后无条件 require 这个文件，跳过转换就得已经有
        die "CONVERT_QUERY=0 但 ${QUERY_OUT_DIR} 下没有 queries.fvecs，run.sh 建完索引后要用它"
    fi
    if [[ -d "${INDEX_ROOT}" && -n "$(ls -A "${INDEX_ROOT}" 2>/dev/null)" ]]; then
        [[ "${FORCE:-0}" == "1" ]] || die "$(cat <<EOT
${INDEX_ROOT} 下已经有东西了。run.sh 对每个分片是先 rm -rf 再重建（run.sh:1000），
这一步会把现有索引覆盖掉。
  确认要重建：FORCE=1 tools/exp.sh build
  想留着旧的做对比：换阈值即可，WORK_DIR 是按 DENSITY_THRESHOLD 派生的，两份互不影响：
    DENSITY_THRESHOLD=0 tools/exp.sh build
    tools/exp.sh size <旧索引目录> <新索引目录>
EOT
)"
        echo "[Warn] 覆盖 ${INDEX_ROOT}"
    fi
    # 不用 seq -s,：BSD 的 seq 会在末尾多一个分隔符，GNU 的不会
    BUILD_DEVICE_IDS=""
    for ((i = 0; i < SHARDS; i++)); do BUILD_DEVICE_IDS+="${BUILD_DEVICE_IDS:+,}${i}"; done
    echo "== 建索引 == 阈值 ${DENSITY_THRESHOLD}  切 ${SHARDS} 份  DOCS=${DOCS}  段 ${DOC_NUM_PER_SEGMENT}"
    echo "   -> ${INDEX_ROOT}/shard0 .. shard$((SHARDS - 1))"
    echo
    env RAPIDJSON_INCLUDE_DIR="${RAPIDJSON_INCLUDE_DIR:-/usr/include}" \
        DATASET_FILE="${DATASET_FILE}" DEVICE_IDS="${BUILD_DEVICE_IDS}" \
        DOCS="${DOCS}" DENSITY_THRESHOLD="${DENSITY_THRESHOLD}" \
        DOC_NUM_PER_SEGMENT="${DOC_NUM_PER_SEGMENT}" \
        WORK_DIR="${WORK_DIR}" RESULT_DIR="${RUNS_DIR}/exp_build_den${DENSITY_THRESHOLD}" \
        QUERY_OUT_DIR="${QUERY_OUT_DIR}" \
        "${ROOT}/run.sh" --compile "${COMPILE}" --convert-query "${CONVERT_QUERY}" \
                         --convert-data 1 --build-index 1 --search 0
    rc=$?
    (( rc == 0 )) || die "建索引失败，rc=${rc}"
    echo
    echo "[索引大小]"
    IFS=',' read -r -a built_dirs <<< "$(shard_dir_list "${SHARDS}")"
    size_report "${built_dirs[@]}"

    # 召回参考缓存。指纹只含四个输入文件的大小+mtime、docNum、nq 等，不含卡数/分组/BATCH_SIZE
    # （recall_ref.cpp:53），所以覆盖全语料建一次，后面所有场景都命中。很慢，默认不做。
    if [[ "${RECALL_REF:-0}" == "1" ]]; then
        echo
        if [[ -f "${RECALL_REF_FILE}" ]]; then
            echo "[召回缓存] 已存在，跳过：${RECALL_REF_FILE}"
        else
            echo "== 建召回参考缓存 == ${RECALL_REF_FILE}（全部查询暴力计算，很慢，只一次）"
            env RAPIDJSON_INCLUDE_DIR="${RAPIDJSON_INCLUDE_DIR:-/usr/include}" \
                DATASET_FILE="${DATASET_FILE}" DEVICE_IDS="${BUILD_DEVICE_IDS}" \
                SHARD_INDEX_DIRS="$(shard_dir_list "${SHARDS}")" \
                DOCS="${DOCS}" DENSITY_THRESHOLD="${DENSITY_THRESHOLD}" \
                DOC_NUM_PER_SEGMENT="${DOC_NUM_PER_SEGMENT}" \
                WORK_DIR="${WORK_DIR}" RESULT_DIR="${RUNS_DIR}/exp_build_ref" \
                QUERY_OUT_DIR="${QUERY_OUT_DIR}" NUM_QUERIES="${NUM_QUERIES}" \
                RECALL_REF_FILE="${RECALL_REF_FILE}" RECALL_REF_THREADS=0 \
                SHARD_GROUP_SIZE="${SHARDS}" PIPELINE=0 ROUND_ROBIN=0 \
                WARMUP=1 BATCH_SIZE=1 "${BASE_SWITCHES[@]}" \
                "${ROOT}/run.sh" --compile 0 --convert-query 0 --convert-data 0 \
                                 --build-index 0 --profile 0 --repeat 1
            rc=$?
            (( rc == 0 )) || die "建召回缓存失败，rc=${rc}"
        fi
    fi
    exit 0
fi

# ---- 前置检查（先做完再开跑，错了立刻停，不会留下半套结果）----------------------
case "${SCENARIO}" in
    latency|load|index) ;;
    *) die "不认识的场景：${SCENARIO}（latency / load / index / size / build）" ;;
esac
[[ -x "${ROOT}/build/fr_search" ]] || die "缺 ${ROOT}/build/fr_search，先跑一次 --compile 1"
[[ -d "${QUERY_OUT_DIR}" ]] || die "查询目录不存在：${QUERY_OUT_DIR}"
[[ -f "${DATASET_FILE}" ]] || die "数据集不存在：${DATASET_FILE}"
# 给了 --recall_ref_file 则召回校验全部查询，没给只校验 RECALL_QUERIES 条（report.cpp:101-102）
if [[ -n "${RECALL_REF_FILE}" && ! -f "${RECALL_REF_FILE}" ]]; then
    if (( RECALL_REF_GIVEN )); then
        echo "[Warn] 缓存 ${RECALL_REF_FILE} 不存在：本次全量暴力计算并写入，很慢，只一次" >&2
    else
        echo "[Warn] 缓存 ${RECALL_REF_FILE} 不存在：召回只校验 ${RECALL_QUERIES} 条（日志显示 cache" >&2
        echo "       disabled）。要全量校验就显式给 RECALL_REF_FILE=${RECALL_REF_FILE} 建一次" >&2
        RECALL_REF_FILE=""
    fi
fi
if [[ "${SCENARIO}" == "latency" ]]; then
    MODE="${MODE:-group_pipeline}"
    arm_env "${MODE}" >/dev/null \
        || die "不认识的 MODE：${MODE}（serial / groups / group_pipeline / pipeline / round_robin）"
    if [[ "${MODE}" == "round_robin" ]]; then
        [[ -n "${FULL_INDEX_DIR}" ]] || die "$(cat <<EOT
MODE=round_robin 要给 FULL_INDEX_DIR。轮转是每张卡各搜全量语料、卡之间不 merge，
拿切两半的分片索引跑，召回只有一半。给一份完整语料的索引：
  MODE=round_robin FULL_INDEX_DIR=<完整索引目录> tools/exp.sh latency
EOT
)"
        [[ -d "${FULL_INDEX_DIR}" ]] || die "FULL_INDEX_DIR 不是目录：${FULL_INDEX_DIR}"
    fi
fi
# MODE 要先定下来，index_dirs_for 靠它判断份数；load / index 没有 MODE，故报错里有则显示。
_mode_note="${MODE:+MODE=${MODE} }"

# 每一档卡数都查一遍：20 个点跑到第三个才炸太浪费
for _set in ${CARD_SETS:-${DEVICE_IDS}}; do
    _n="$(awk -F, '{print NF}' <<< "${_set}")"
    (( _n % CARDS_PER_GROUP == 0 )) \
        || die "卡 ${_set} 是 ${_n} 张，不是 CARDS_PER_GROUP=${CARDS_PER_GROUP} 的整数倍"
    _want="$(index_dirs_for "${_n}" | tr ',' '\n' | wc -l | tr -d ' ')"
    for d in $(index_dirs_for "${_n}" | tr ',' ' '); do
        [[ -d "${d}" ]] || die "$(cat <<EOT
索引目录不存在：${d}（${_mode_note}${_n} 张卡要 ${_want} 份）
先建 tools/exp.sh build SHARDS=${_want} WORK_DIR=<目录>，或给 INDEX_DIRS=<逗号分隔>
EOT
)"
    done
    # 只取一份多分片索引的前几份 = 只搜一部分语料，而召回查不出来：分组路径的 docNum 只按
    # 组内分片之和算（setup.cpp:130），CPU 参考用同一个子集，照样报 100%。
    if (( INDEX_DIRS_GIVEN == 0 )); then
        _have="$(find "${WORK_DIR}/index" -maxdepth 1 -type d -name 'shard*' 2>/dev/null | wc -l | tr -d ' ')"
        (( _have <= _want )) || die "$(cat <<EOT
${WORK_DIR}/index 有 ${_have} 份分片，这次只用前 ${_want} 份 = 只搜 ${_want}/${_have} 语料（召回仍报 100%，查不出）
改 CARDS_PER_GROUP=${_have} 或 MODE=pipeline 用满，或换一份 ${_want} 分片的索引
EOT
)"
    fi
done

if [[ "${SCENARIO}" == "load" ]]; then
    GROUP_COUNT=$((CARD_COUNT / CARDS_PER_GROUP))
    # 组只有一个时 staticChunk 的递增步长就是 1，和工作窃取的序列完全一样
    # （schedule_groups.cpp:282），STATIC_ASSIGN 是个空操作，A/B 必然得出零效应。
    # 这里直接拦掉，不要跑出一个假的 neutral。
    (( GROUP_COUNT >= 2 )) || die "$(cat <<EOT
${CARD_COUNT} 张卡 / CARDS_PER_GROUP=${CARDS_PER_GROUP} 只有 1 个组，STATIC_ASSIGN 无从生效
（组内没有分配可言）。两条出路：
  八卡：DEVICE_IDS=0,1,2,3,4,5,6,7 CARDS_PER_GROUP=2 tools/exp.sh load
  两卡：每张卡各持一份完整索引 ——
        CARDS_PER_GROUP=1 INDEX_DIRS=<完整索引目录> tools/exp.sh load
        （run.sh 会把这一个目录复制成每卡一份；CARDS_PER_GROUP=1 时 SHARD_TOPK_RATIO 回 1.0）
EOT
)"
    [[ -x "${ROOT}/build/fr_npuload" ]] || die "缺 ${ROOT}/build/fr_npuload，先跑一次 --compile 1"
    # 默认只压 0 号卡。一组里任一张卡慢，整组就慢（组内扇出要等齐），所以压一张就够；
    # 而所有组都一样慢的话动态分配无处可调，A/B 就没有意义了。
    NPU_LOAD_CARDS="${NPU_LOAD_CARDS:-0}"
    for c in ${NPU_LOAD_CARDS//,/ }; do
        [[ ",${DEVICE_IDS}," == *",${c},"* ]] \
            || die "NPU_LOAD_CARDS 的 ${c} 不在 DEVICE_IDS 里，压的不是引擎在用的卡"
    done
fi

# ---- 解析日志 ----------------------------------------------------------------
# 末尾的汇总要从各次运行的输出里抓指标，而 stdout 归谁管由调用方决定（可能是终端，也可能是
# 外层 dtach 的重定向），脚本没法假设自己的输出落到了哪个文件。所以另外留一份固定路径的，
# 专门给汇总解析用。内容和你看到的一样，只是位置确定。
mkdir -p "${LOG_DIR}"
PARSE_LOG="${PARSE_LOG:-${LOG_DIR}/exp-${SCENARIO}-$(date +%m%d-%H%M%S)-parse.log}"
: > "${PARSE_LOG}" || die "解析日志写不了：${PARSE_LOG}"
echo "场景 ${SCENARIO}   卡 ${CARD_SETS:-${DEVICE_IDS}}   索引 ${WORK_DIR}/index"
echo

# ---- 一次运行 ----------------------------------------------------------------
RUN_INDEX=0; RUN_TOTAL=0
one_run() {   # <tag> <第几个样本> [额外的 env 赋值...]
    local tag="$1" sample="$2"; shift 2
    RUN_INDEX=$((RUN_INDEX + 1))
    local notify_this=0
    [[ "${NOTIFY}" == "1" && "${RUN_INDEX}" -eq "${RUN_TOTAL}" ]] && notify_this=1
    local -a environment=(
        RAPIDJSON_INCLUDE_DIR="${RAPIDJSON_INCLUDE_DIR:-/usr/include}"
        DATASET_FILE="${DATASET_FILE}" DEVICE_IDS="${DEVICE_IDS}" SHARD_INDEX_DIRS="${INDEX_DIRS}"
        DOCS="${DOCS}" DENSITY_THRESHOLD="${DENSITY_THRESHOLD}"
        DOC_NUM_PER_SEGMENT="${DOC_NUM_PER_SEGMENT}"
        WORK_DIR="${WORK_DIR}" RESULT_DIR="${RUNS_DIR}/exp_${SCENARIO}_${tag}_${sample}"
        QUERY_OUT_DIR="${QUERY_OUT_DIR}" NUM_QUERIES="${NUM_QUERIES}"
        RECALL_QUERIES="${RECALL_QUERIES}" WARMUP="${WARMUP}" BATCH_SIZE="${BATCH_SIZE}"
        MEM_REPORT="${MEM_REPORT}" NOTIFY="${notify_this}"
        SLOW_CARDS= NPU_LOAD_CARDS="${LOAD_CARDS_THIS_ARM}"
    )
    [[ -n "${RECALL_REF_FILE}" ]] \
        && environment+=(RECALL_REF_FILE="${RECALL_REF_FILE}" RECALL_REF_THREADS=0)
    environment+=("${BASE_SWITCHES[@]}" "$@")
    local -a command=()
    [[ -n "${TASKSET_CPUS}" ]] && command=(taskset -c "${TASKSET_CPUS}")   # 空数组在 set -u 下
    command+=("${ROOT}/run.sh" --compile 0 --convert-query 0 --convert-data 0 --build-index 0
              --profile 0 --repeat "${REPEAT}")                            # 不能展开，所以并成一个
    # 整段经 tee 落进解析日志。注意这是个子 shell，里面别改外面要用的变量
    # （RUN_INDEX 在上面已经加过了）。
    {
        echo "########## tag=${tag} rep=${sample} (${RUN_INDEX}/${RUN_TOTAL}) $(date '+%F %T') ##########"
        if [[ "${DROP_CACHES}" == "1" ]]; then
            sync
            echo 3 > /proc/sys/vm/drop_caches 2>/dev/null \
                && echo "  [页缓存已清]" || echo "  [Warn] 清不了页缓存（要 root），这次是热的"
        fi
        pkill -9 python
        env "${environment[@]}" "${command[@]}"
        echo "########## end tag=${tag} rep=${sample} rc=$? ##########"
    } 2>&1 | tee -a "${PARSE_LOG}"
}

# 回文：偶数轮正序、奇数轮逆序
run_arms() {   # <tag 前缀（可空）> <配置名...>  只有 load 档用
    local prefix="$1"; shift
    local -a arms=("$@") order
    local round i arm
    for ((round = 0; round < ROUNDS; round++)); do
        order=("${arms[@]}")
        (( round % 2 == 1 )) && {
            order=(); for ((i = ${#arms[@]} - 1; i >= 0; i--)); do order+=("${arms[i]}"); done; }
        for arm in "${order[@]}"; do
            one_run "${prefix:+${prefix}-}${arm}" "$((round + 1))" \
                    $(arm_env "${arm}") ${SCENARIO_ENV:-}
        done
    done
}

# ---- 场景 --------------------------------------------------------------------
LOAD_CARDS_THIS_ARM=""; SHOW_SHARD_COLUMNS=0; BASELINE_ARM=""
case "${SCENARIO}" in
latency)
    # 四条路径的区别不在「用几张卡」——多分片时四种都会在批内用满所有卡：
    #   serial          一批一批做。批内向所有分片并行扇出（一分片一线程，run_context.cpp:185），
    #                   等齐后在主机 merge，再做下一批。批之间不重叠，所以 BATCH_SIZE=1 的
    #                   时延就是真的端到端时延。走 BatchSearch（设备段和 Extract 合在一起）。
    #   groups          卡分成 nShards/CARDS_PER_GROUP 组，每组持一份完整语料。查询在组内扇出，
    #                   不同组同时取不同的 chunk。走 BatchSearchDevice + BatchSearchExtract。
    #   group_pipeline  groups 再加组内双缓冲：chunk N 的 Extract 压在 chunk N+1 的设备段上。
    #   pipeline        不分组，批内向所有分片扇出，但上一批的 Extract 和这一批的设备段重叠。
    #   round_robin     每张卡持一份完整语料，整个 chunk 只在一张卡上做，卡之间不 merge。
    #
    # 延迟数的含义：只有 serial 支持开环（rc.paced / arrivalOf 只在 schedule_serial.cpp 里），
    # 其余四种的 --target_qps 静默无效，报的是服务时间——这一批自己的耗时，不含排队。
    # 流水重叠的是相邻两批（批 N 的 Extract 对批 N+1 的设备段），批 N 自己仍然是设备段做完
    # 才轮到它的 Extract，所以 devMs[n]+extMs[n] 就是它的真实耗时。流水的影响是吞吐和延迟
    # 脱钩：qps 不再等于 1/延迟，因为同时有两批在跑。
    # MODE 已在前置检查里定好并验过名字。
    BATCH_SIZE="${BATCH_SIZE:-1}"; REPEAT="${REPEAT:-3}"
    # 默认一个点。CARD_SETS / BATCH_SIZES 给多个值就跑交叉矩阵，一条命令出一张表。
    # CARD_SETS 用空格分隔若干组卡号（每组内部仍是逗号）：CARD_SETS="0,1 0,1,2,3"
    read -r -a card_sets <<< "${CARD_SETS:-${DEVICE_IDS}}"
    read -r -a batch_sizes <<< "${BATCH_SIZES:-${BATCH_SIZE}}"
    RUN_TOTAL=$((${#card_sets[@]} * ${#batch_sizes[@]}))
    if (( RUN_TOTAL == 1 )); then
        echo "== 时延 == MODE=${MODE}  ${CARD_COUNT} 卡 ${DEVICE_IDS}  每组 ${CARDS_PER_GROUP} 张" \
             " BATCH_SIZE=${BATCH_SIZE}  NUM_QUERIES=${NUM_QUERIES}  --repeat ${REPEAT}"
    else
        echo "== 时延矩阵 == MODE=${MODE}  每组 ${CARDS_PER_GROUP} 张  卡 ${#card_sets[@]} 档" \
             " × BATCH_SIZE ${#batch_sizes[@]} 档 = ${RUN_TOTAL} 次运行  --repeat ${REPEAT}"
        echo "   卡 ${CARD_SETS:-${DEVICE_IDS}}   BATCH_SIZE ${BATCH_SIZES:-${BATCH_SIZE}}"
    fi
    for card_set in "${card_sets[@]}"; do
        DEVICE_IDS="${card_set}"
        CARD_COUNT="$(awk -F, '{print NF}' <<< "${card_set}")"
        INDEX_DIRS="$(index_dirs_for "${CARD_COUNT}")"
        for batch in "${batch_sizes[@]}"; do
            BATCH_SIZE="${batch}"
            # tag 补零，否则汇总里 bs16 会排在 bs2 前面（按字符串排的）
            one_run "$(printf 'c%d-bs%02d' "${CARD_COUNT}" "${batch}")" 1 $(arm_env "${MODE}")
        done
    done
    BASELINE_ARM=""          # 没有对照配置，汇总只列绝对值
    ;;
load)
    ARMS="${ARMS:-dynamic fixed}"
    BATCH_SIZE="${BATCH_SIZE:-2}"; REPEAT="${REPEAT:-3}"
    read -r -a arm_list <<< "${ARMS}"
    RUN_TOTAL=$((${#arm_list[@]} * ROUNDS * 2))
    # 负载 A/B 必须走分组路径，STATIC_ASSIGN 只在那里（和 ROUND_ROBIN）有意义；吞吐用流水。
    SCENARIO_ENV="SHARD_GROUP_SIZE=${CARDS_PER_GROUP} PIPELINE=1 ROUND_ROBIN=0"
    (( CARDS_PER_GROUP == 1 )) && SCENARIO_ENV+=" SHARD_TOPK_RATIO=1.0"
    # fr_npuload 的扰动强度。比 run.sh 的默认激进得多：周期 50ms 而不是 1000ms，扰动的开合
    # 落在查询的时间尺度上而不是几百条查询之后；6 个计算 + 4 个拷贝 worker、拷贝之间不留间隔。
    # run.sh 从环境里读这几个，env 不带 -i，所以 export 后会继承下去。
    NPU_LOAD_DUTY="${NPU_LOAD_DUTY:-0.5}"
    NPU_LOAD_PERIOD_MS="${NPU_LOAD_PERIOD_MS:-50}"
    NPU_LOAD_COMPUTE_WORKERS="${NPU_LOAD_COMPUTE_WORKERS:-6}"
    NPU_LOAD_COPY_WORKERS="${NPU_LOAD_COPY_WORKERS:-4}"
    NPU_LOAD_COPY_GAP_MS="${NPU_LOAD_COPY_GAP_MS:-0}"
    export NPU_LOAD_DUTY NPU_LOAD_PERIOD_MS NPU_LOAD_COMPUTE_WORKERS \
           NPU_LOAD_COPY_WORKERS NPU_LOAD_COPY_GAP_MS
    echo "== 负载波动对比 == ${CARD_COUNT} 卡 / ${GROUP_COUNT} 组 × ${CARDS_PER_GROUP}" \
         " 负载加在卡 ${NPU_LOAD_CARDS}  对比 ${ARMS}，变化相对 fixed"
    echo "   BATCH_SIZE=${BATCH_SIZE}  --repeat ${REPEAT}  共 ${RUN_TOTAL} 次运行"
    echo "   扰动 duty=${NPU_LOAD_DUTY} period=${NPU_LOAD_PERIOD_MS}ms" \
         "compute=${NPU_LOAD_COMPUTE_WORKERS} copy=${NPU_LOAD_COPY_WORKERS}" \
         "copy_gap=${NPU_LOAD_COPY_GAP_MS}ms buffer=${NPU_LOAD_COPY_BUFFER_MB:-256}MB"
    for loaded in 0 1; do
        LOAD_CARDS_THIS_ARM=""
        [[ "${loaded}" == "1" ]] && LOAD_CARDS_THIS_ARM="${NPU_LOAD_CARDS}"
        run_arms "$([[ "${loaded}" == "1" ]] && echo loaded || echo noload)" "${arm_list[@]}"
    done
    BASELINE_ARM=fixed
    ;;
index)
    # 加载只在进程启动时发生一次，所以每次只跑少量查询；--repeat 3 就是三次加载样本。
    PARALLEL_LOAD="${PARALLEL_LOAD:-1}"
    BATCH_SIZE="${BATCH_SIZE:-1}"; REPEAT="${REPEAT:-3}"
    NUM_QUERIES="${NUM_QUERIES_INDEX:-10888}"
    RUN_TOTAL=1
    SHOW_SHARD_COLUMNS=1
    echo "== 索引加载与大小 == PARALLEL_LOAD=${PARALLEL_LOAD}  ${CARD_COUNT} 卡 ${DEVICE_IDS}" \
         " NUM_QUERIES=${NUM_QUERIES}  --repeat ${REPEAT}"
    echo
    echo "[索引大小]"
    IFS=',' read -r -a index_dir_list <<< "${INDEX_DIRS}"
    size_report "${index_dir_list[@]}"
    echo
    # 调度方式不影响加载（加载发生在任何调度之前），固定成最简单的一种，少一个变量
    one_run index 1 "SHARD_GROUP_SIZE=${CARDS_PER_GROUP}" PIPELINE=0 ROUND_ROBIN=0 \
            "PARALLEL_LOAD=${PARALLEL_LOAD}"
    BASELINE_ARM=""
    ;;
esac

# ---- 汇总 --------------------------------------------------------------------
echo
echo "===== 结果 ====="
PARSE_LOG="${PARSE_LOG}" BASELINE_ARM="${BASELINE_ARM}" SHOW_SHARD_COLUMNS="${SHOW_SHARD_COLUMNS}" \
python3 - <<'PY'
import os, re, statistics, collections

log_path = os.environ["PARSE_LOG"]
baseline_arm = os.environ.get("BASELINE_ARM", "")
show_shard_columns = os.environ.get("SHOW_SHARD_COLUMNS", "0") == "1"

samples = collections.defaultdict(lambda: collections.defaultdict(list))
seen_tags = set()
tag = None
for line in open(log_path, errors="replace"):
    match = re.match(r"#+ tag=(\S+) rep=(\d+)", line)
    if match:
        tag = match.group(1); seen_tags.add(tag); continue
    if tag is None:
        continue
    match = re.match(r"\[Load\] shard (\d+) on device \d+:.* in ([0-9.]+)s", line)
    if match:
        samples[tag]["shard" + match.group(1)].append(float(match.group(2))); continue
    match = re.match(r"\[Load\] \d+ shard\(s\).* in ([0-9.]+)s", line)
    if match:
        samples[tag]["load_sec"].append(float(match.group(1))); continue
    # --repeat>1 时 run.sh 打出跨轮中位数表，行尾就是中位数；优先用它
    match = re.match(r"(avg|p50|p95|p99|qps)((?:\s+[0-9.]+)+)\s*$", line)
    if match:
        samples[tag][match.group(1)].append(float(match.group(2).split()[-1])); continue
    match = re.match(r"\[Result\] latency ms: (.*)", line)
    if match:
        fields = dict(re.findall(r"([a-z0-9]+)=([0-9.]+)", match.group(1)))
        for key in ("avg", "p50", "p95", "p99", "qps"):
            if key in fields:
                samples[tag][key + "!single"].append(float(fields[key]))
        continue
    match = re.search(r"CPU-recall over \d+ queries: avg=([0-9.]+)%", line)
    if match:
        samples[tag]["recall"].append(float(match.group(1)))

if not seen_tags:
    raise SystemExit("解析日志里没有 tag 标记，一次运行都没起来。")
if not samples:
    raise SystemExit("%d 种配置都跑了，但没有一行可解析的指标——run.sh 应该是挂了，看解析日志里的 rc="
                     % len(seen_tags))

shard_columns = sorted({k for t in samples for k in samples[t] if re.fullmatch(r"shard\d+", k)},
                       key=lambda k: int(k[5:]))
columns = (["load_sec"] + (shard_columns if show_shard_columns else [])
           + ["avg", "p50", "p99", "qps", "recall"])
# 中文是双宽字符，按字符数补宽会错位，按显示宽度补
def display_width(text):
    return sum(2 if ord(ch) > 0x2E7F else 1 for ch in text)

def pad_right(text, width):   # 右对齐（数据列）
    return " " * max(0, width - display_width(text)) + text

def pad_left(text, width):    # 左对齐（首列）
    return text + " " * max(0, width - display_width(text))

# 表头用看得懂的词；avg/p50/p99/qps 是 run.sh 自己的叫法，保持一致
header = {"load_sec": "加载s", "recall": "召回"}
for c in shard_columns:
    header[c] = "分片" + c[5:]
column_format = {"qps": "%10.1f", "recall": "%10.2f"}
for key in ["load_sec"] + shard_columns:
    column_format[key] = "%10.1f"

def values(tag, key):
    return samples[tag].get(key) or samples[tag].get(key + "!single") or []

def median(v):
    return statistics.median(v) if v else float("nan")

# 同一格里几个样本之间差多少，就是这一格能分辨的下限
def spread_percent(tag, key):
    v = values(tag, key)
    if len(v) < 2 or not median(v):
        return float("nan")
    return 100 * (max(v) - min(v)) / median(v)

by_prefix = collections.defaultdict(list)
for tag in samples:
    prefix, _, arm = tag.rpartition("-")
    by_prefix[prefix].append((arm, tag))

# 只有一个配置（latency 的单次测量）：不摆表，直接说数
if len(samples) == 1:
    tag = next(iter(samples))
    def one(key, unit=""):
        v = values(tag, key)
        return None if not v else "%s%s" % (
            ("%.3f" if key in ("avg", "p50", "p99") else "%.1f" if key != "recall" else "%.2f")
            % median(v), unit)
    lat = [x for x in ("avg " + (one("avg", " ms") or ""), "p50 " + (one("p50", " ms") or ""),
                       "p99 " + (one("p99", " ms") or "")) if not x.endswith(" ")]
    print("  " + "   ".join(lat))
    line = []
    if one("qps"):
        line.append("吞吐 %s qps" % one("qps"))
    if one("recall"):
        line.append("召回 %s%%" % one("recall"))
    if line:
        print("  " + "   ".join(line))
    if one("load_sec"):
        n = len(values(tag, "load_sec"))
        sp = spread_percent(tag, "load_sec")
        extra = "" if n < 2 or sp != sp else "（%d 次，相差 %.1f%%）" % (n, sp)
        line = "  索引加载 %s s%s" % (one("load_sec"), extra)
        if show_shard_columns:
            per = ["分片%s %s s" % (c[5:], one(c)) for c in shard_columns if one(c)]
            if per:
                line += "   " + "   ".join(per)
        print(line)
    raise SystemExit(0)

for prefix in sorted(by_prefix):
    arms = sorted(by_prefix[prefix])
    if prefix:
        print("\n[%s]" % prefix)
    print("  " + pad_left("配置", 21) + pad_right("次数", 4)
          + "".join(pad_right(header.get(c, c), 10) for c in columns))
    for arm, tag in arms:
        n = max(len(values(tag, c)) for c in columns)
        print("  " + pad_left(arm, 21) + pad_right(str(n), 4) + "".join(
            (column_format.get(c, "%10.4f") % median(values(tag, c))) if values(tag, c)
            else "%10s" % "-" for c in columns))
    baseline = dict(arms).get(baseline_arm)
    if not baseline:
        continue
    for arm, tag in arms:
        if tag == baseline:
            continue
        row = []
        for c in columns:
            base_value, arm_value = median(values(baseline, c)), median(values(tag, c))
            row.append("%+9.2f%%" % (100 * (arm_value / base_value - 1))
                       if (values(baseline, c) and values(tag, c) and base_value)
                       else "%10s" % "-")
        print("  " + pad_left("变化 " + arm, 21) + " " * 4 + "".join(row))
    row = []
    for c in columns:
        spreads = [x for x in (spread_percent(t, c) for _, t in arms) if x == x]
        row.append("%10s" % ("-" if not spreads else "%.2f%%" % max(spreads)))
    print("  " + pad_left("同配置波动", 21) + " " * 4 + "".join(row))

PY
echo
echo "跑完。解析日志 ${PARSE_LOG}"
