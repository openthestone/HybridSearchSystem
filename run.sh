#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE:-$0}")" && pwd)"

_last_cmd=""
trap '_last_cmd=${BASH_COMMAND}' DEBUG
_on_exit() {
    local rc=$?
    [[ ${rc} -eq 0 ]] && return 0
    echo "" >&2
    echo "[FATAL] run.sh aborted: exit ${rc} at line ${BASH_LINENO[0]}" >&2
    echo "        last command: ${_last_cmd}" >&2
    echo "        (re-run with DEBUG=1 for stage checkpoints, DEBUG=2 for a full trace)" >&2
}
trap _on_exit EXIT

DEBUG="${DEBUG:-0}"
dbg() {
    [[ "${DEBUG}" != "0" ]] || return 0
    echo "[DEBUG] $*" >&2
}
if [[ "${DEBUG}" == "2" ]]; then
    export PS4='+ ${BASH_SOURCE##*/}:${LINENO}: '
    set -x
fi
dbg "run.sh start: pid=$$ bash=${BASH_VERSION} pwd=$(pwd)"

# Local, gitignored secrets/config (e.g. NOTIFY_API_KEY). Never committed.
if [[ -f "${ROOT_DIR}/.notify.env" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/.notify.env"
fi

if [[ -z "${SOC_VERSION:-}" ]]; then
    _soc_detected="$(npu-smi info 2>/dev/null | grep -oiE '910B[0-9]' | head -1 | tr '[:lower:]' '[:upper:]')"
    if [[ -n "${_soc_detected}" ]]; then
        SOC_VERSION="Ascend${_soc_detected}"
    else
        SOC_VERSION="Ascend910B4"
    fi
fi
CANN="${CANN:-${ASCEND_CANN_PACKAGE_PATH:-${ASCEND_INSTALL_PATH:-}}}"
if [[ -z "${CANN}" ]]; then
    if [[ -d "${HOME}/Ascend/ascend-toolkit/latest" ]]; then
        CANN="${HOME}/Ascend/ascend-toolkit/latest"
    else
        CANN="/usr/local/Ascend/ascend-toolkit/latest"
    fi
fi

DATASET_FILE="${DATASET_FILE:-${ROOT_DIR}/dataset.bin}"
TAG_MAP_FILE="${TAG_MAP_FILE:-${ROOT_DIR}/tag_map.bin}"
QUERYDATA_FILE="${QUERYDATA_FILE:-${ROOT_DIR}/QueryData_10000.txt}"

# Defined before WORK_DIR so the default work/result dirs can be tagged with them: each
# (segment, density) combo gets its own index dir.
DOC_NUM_PER_SEGMENT="${DOC_NUM_PER_SEGMENT:-262144}"
DENSITY_THRESHOLD="${DENSITY_THRESHOLD:-0.05}"
# Both are no-ops at threshold 0 (no sparse postings). Set either to 0 explicitly as a control arm.
DENSITY_NONZERO=$(awk -v t="${DENSITY_THRESHOLD}" 'BEGIN { print (t + 0 > 0) ? 1 : 0 }')
SPARSE_DIRECT="${SPARSE_DIRECT:-${DENSITY_NONZERO}}"
OVERLAP_POSTING_BITLIST="${OVERLAP_POSTING_BITLIST:-${DENSITY_NONZERO}}"
# Only ApplySparse reads it, and only for a tagged operand, so it is a no-op without SPARSE_DIRECT.
OR_NO_TILE_CHECK="${OR_NO_TILE_CHECK:-${SPARSE_DIRECT}}"
CONFIG_TAG="seg${DOC_NUM_PER_SEGMENT}_den${DENSITY_THRESHOLD}"
RUNS_DIR="${RUNS_DIR:-${ROOT_DIR}/runs}"

WORK_DIR="${WORK_DIR:-${RUNS_DIR}/${CONFIG_TAG}/work}"
QUERY_OUT_DIR="${QUERY_OUT_DIR:-${WORK_DIR}/queries}"
CONVERTER_IN_DIR="${CONVERTER_IN_DIR:-${WORK_DIR}/builder_input}"
INDEX_DIR="${INDEX_DIR:-${WORK_DIR}/index}"
RESULT_DIR="${RESULT_DIR:-${RUNS_DIR}/${CONFIG_TAG}/result}"

GFLAGS_LIBRARY_DIR="${GFLAGS_LIBRARY_DIR:-}"
PROTOBUF_LIBRARY_DIR="${PROTOBUF_LIBRARY_DIR:-}"
# FindProtobuf matches libprotobuf.so/.a only -- a versioned libprotobuf.so.25.1.0 with no bare
# symlink is NOT found, and the error reads "found version 4.25.1, missing Protobuf_LIBRARIES".
PROTOBUF_ROOT="${PROTOBUF_ROOT:-}"
PROTOBUF_LIBRARY="${PROTOBUF_LIBRARY:-}"
PROTOBUF_INCLUDE_DIR="${PROTOBUF_INCLUDE_DIR:-}"
PROTOC="${PROTOC:-}"
ABSL_LIBRARY_DIR="${ABSL_LIBRARY_DIR:-}"
EXTRA_LD_LIBRARY_PATH="${EXTRA_LD_LIBRARY_PATH:-}"

RAPIDJSON_INCLUDE_DIR="${RAPIDJSON_INCLUDE_DIR:-}"
if [[ -z "${RAPIDJSON_INCLUDE_DIR}" ]]; then
    for candidate in \
        "${ROOT_DIR}/rapidjson/include" \
        "${ROOT_DIR}/rapidjson-1.1.0/include" \
        "${ROOT_DIR}/third_party/rapidjson/include" \
        "${ROOT_DIR}/third_party/rapidjson-1.1.0/include" \
        "${ROOT_DIR}/../rapidjson/include" \
        "${ROOT_DIR}/../rapidjson-1.1.0/include" \
        "/opt/rapidjson/include" \
        "/opt/rapidjson-1.1.0/include" \
        "/usr/include" \
        "/usr/local/include"; do
        if [[ -f "${candidate}/rapidjson/document.h" ]]; then
            RAPIDJSON_INCLUDE_DIR="${candidate}"
            break
        fi
    done
fi
if [[ -n "${RAPIDJSON_INCLUDE_DIR}" \
    && ! -f "${RAPIDJSON_INCLUDE_DIR}/rapidjson/document.h" \
    && -f "${RAPIDJSON_INCLUDE_DIR}/include/rapidjson/document.h" ]]; then
    RAPIDJSON_INCLUDE_DIR="${RAPIDJSON_INCLUDE_DIR}/include"
fi

DOCS="${DOCS:-0}"
# Scorer on Cube (20 cores on 910B3), filter on Vector (40). The filter is quantized by SEGMENT
# count, so its useful ceiling is min(40, segments).
FULL_RECALL_TEXT_FILTER_BLOCK_DIM="${FULL_RECALL_TEXT_FILTER_BLOCK_DIM:-40}"
# Also the block count of four GmMemoryPools, so it multiplies the transient device memory.
BATCH_SEARCH_THREADS="${BATCH_SEARCH_THREADS:-8}"
SCORER_BLOCK_DIM="${SCORER_BLOCK_DIM:-20}"
# Capped at 40. The TAIL core gets the remainder, so pick a count that divides
# blockNumber = ceil(docs/5888): at 5M docs/shard that is 850 = 2*5^2*17.
AGGREGATOR_BLOCK_DIM="${AGGREGATOR_BLOCK_DIM:-16}"
# Any value <= 1.0 disables early-quit -- the check is unsatisfiable there. See kernel_topk.h.
NPU_TOPK_FINISH_BUFFER_RATIO="${NPU_TOPK_FINISH_BUFFER_RATIO:-1.2}"
NUM_QUERIES="${NUM_QUERIES:-0}"
# At >1, "latency ms avg" is ms/query, p99 degenerates (every query gets the batch average) and a
# query's TRUE latency is avg*BATCH_SIZE. Use 1 for the p99 target.
BATCH_SIZE="${BATCH_SIZE:-1}"
WARMUP="${WARMUP:-5}"
# Pass ONE full-index dir in SHARD_INDEX_DIRS; it is replicated across DEVICE_IDS below.
ROUND_ROBIN="${ROUND_ROBIN:-0}"
# The contract's "固定任务分配" baseline. Only differs from dynamic when workers are NOT equally fast.
STATIC_ASSIGN="${STATIC_ASSIGN:-0}"
# Simulated NPU load fluctuation (contract: 模拟的 NPU 负载波动), two mechanisms: SLOW_CARDS is a
# host-side delay, NPU_LOAD_CARDS is real contention from fr_npuload (prefer it for acceptance).
# Do NOT enable both at once or the disturbance is counted twice.
NPU_LOAD_CARDS="${NPU_LOAD_CARDS:-}"
NPU_LOAD_DUTY="${NPU_LOAD_DUTY:-0.5}"
NPU_LOAD_PERIOD_MS="${NPU_LOAD_PERIOD_MS:-1000}"
NPU_LOAD_COMPUTE_WORKERS="${NPU_LOAD_COMPUTE_WORKERS:-1}"
NPU_LOAD_COPY_WORKERS="${NPU_LOAD_COPY_WORKERS:-1}"
NPU_LOAD_COPY_BUFFER_MB="${NPU_LOAD_COPY_BUFFER_MB:-256}"
NPU_LOAD_COPY_GAP_MS="${NPU_LOAD_COPY_GAP_MS:-2.0}"
SLOW_CARDS="${SLOW_CARDS:-}"
SLOW_FACTOR="${SLOW_FACTOR:-2.0}"
SLOW_PERIOD_MS="${SLOW_PERIOD_MS:-1000}"
SLOW_DUTY="${SLOW_DUTY:-0.5}"
# Open-loop pacing: latency is measured from each query's ARRIVAL, so queueing counts. 0 = off.
TARGET_QPS="${TARGET_QPS:-0}"
# Multi-card only, and latency is NOT meaningful in this mode. Composes with SHARD_GROUP_SIZE.
PIPELINE="${PIPELINE:-0}"
# No cross-card merge here, so a per-shard ratio <1 would silently truncate (0.6 -> recall ~60%).
if [[ "${ROUND_ROBIN}" == "1" && -n "${SHARD_TOPK_RATIO:-}" \
      && "${SHARD_TOPK_RATIO}" != "1" && "${SHARD_TOPK_RATIO}" != "1.0" ]]; then
    echo "[Warn] ROUND_ROBIN=1: ignoring SHARD_TOPK_RATIO=${SHARD_TOPK_RATIO} (no merge to restore the" >&2
    echo "       dropped results; it would cap recall at ~${SHARD_TOPK_RATIO}x). Forcing 1.0." >&2
    SHARD_TOPK_RATIO=1.0
fi
RECALL_QUERIES="${RECALL_QUERIES:-20}"
DEFAULT_TOPK="${DEFAULT_TOPK:-100}"
DEVICE_ID="${DEVICE_ID:-0}"
DEVICE_IDS="${DEVICE_IDS:-}"
SHARD_INDEX_DIRS="${SHARD_INDEX_DIRS:-}"
RECALL_REF_FILE="${RECALL_REF_FILE:-}"
RECALL_REF_THREADS="${RECALL_REF_THREADS:-0}"
BUILD_THREADS="${BUILD_THREADS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 32)}"
CONVERT_THREADS="${CONVERT_THREADS:-${BUILD_THREADS}}"

SHARD_DEVICES=()
if [[ -n "${DEVICE_IDS}" ]]; then
    IFS=',' read -r -a SHARD_DEVICES <<< "${DEVICE_IDS}"
fi
SHARD_NUM="${#SHARD_DEVICES[@]}"
SHARD_INDEX_LIST=()
EXTERNAL_SHARDS=0
if [[ -n "${SHARD_INDEX_DIRS}" ]]; then
    EXTERNAL_SHARDS=1
    IFS=',' read -r -a SHARD_INDEX_LIST <<< "${SHARD_INDEX_DIRS}"
fi
# Replicate the single full-index dir across all DEVICE_IDS.
if [[ "${ROUND_ROBIN}" == "1" && "${EXTERNAL_SHARDS}" == "1" && "${#SHARD_INDEX_LIST[@]}" -eq 1 && "${SHARD_NUM}" -gt 1 ]]; then
    _full="${SHARD_INDEX_LIST[0]}"
    SHARD_INDEX_LIST=()
    for ((_i = 0; _i < SHARD_NUM; _i++)); do SHARD_INDEX_LIST+=("${_full}"); done
    SHARD_INDEX_DIRS="$(IFS=,; echo "${SHARD_INDEX_LIST[*]}")"
fi
# Every GROUP holds a full corpus split SHARD_GROUP_SIZE ways, so pass ONE group's shard dirs and
# they are repeated per group. Cards group contiguously: the list must read s0,s1,...,s0,s1,...
SHARD_GROUP_SIZE="${SHARD_GROUP_SIZE:-0}"
if [[ "${SHARD_GROUP_SIZE}" -gt 0 && "${EXTERNAL_SHARDS}" == "1" \
      && "${#SHARD_INDEX_LIST[@]}" -eq "${SHARD_GROUP_SIZE}" && "${SHARD_NUM}" -gt "${SHARD_GROUP_SIZE}" ]]; then
    if (( SHARD_NUM % SHARD_GROUP_SIZE != 0 )); then
        echo "[ERROR] DEVICE_IDS count (${SHARD_NUM}) is not a multiple of SHARD_GROUP_SIZE (${SHARD_GROUP_SIZE})." >&2
        exit 1
    fi
    _grp=("${SHARD_INDEX_LIST[@]}")
    SHARD_INDEX_LIST=()
    for ((_g = 0; _g < SHARD_NUM / SHARD_GROUP_SIZE; _g++)); do SHARD_INDEX_LIST+=("${_grp[@]}"); done
    SHARD_INDEX_DIRS="$(IFS=,; echo "${SHARD_INDEX_LIST[*]}")"
fi
# A group's merge spans only its OWN shards, so the ratio cannot go below 1/group-size.
if [[ "${SHARD_GROUP_SIZE}" -gt 0 && -n "${SHARD_TOPK_RATIO:-}" \
      && "${SHARD_TOPK_RATIO}" != "1" && "${SHARD_TOPK_RATIO}" != "1.0" ]]; then
    if [[ "${SHARD_GROUP_SIZE}" -eq 1 ]]; then
        echo "[Warn] SHARD_GROUP_SIZE=1: ignoring SHARD_TOPK_RATIO=${SHARD_TOPK_RATIO} (a one-shard group has" >&2
        echo "       no merge to restore the dropped results; it would cap recall at ~${SHARD_TOPK_RATIO}x). Forcing 1.0." >&2
        SHARD_TOPK_RATIO=1.0
    elif awk -v r="${SHARD_TOPK_RATIO}" -v g="${SHARD_GROUP_SIZE}" 'BEGIN{exit !(r < 1/g)}'; then
        echo "[ERROR] SHARD_TOPK_RATIO=${SHARD_TOPK_RATIO} is below 1/SHARD_GROUP_SIZE (1/${SHARD_GROUP_SIZE}): the" >&2
        echo "        group's ${SHARD_GROUP_SIZE} shards cannot together supply a full topK, so recall is capped." >&2
        exit 1
    fi
fi
MULTICARD=0
{ [[ "${SHARD_NUM}" -gt 0 ]] || [[ "${EXTERNAL_SHARDS}" == "1" ]]; } && MULTICARD=1

shard_doc_range() {
    local i="$1"
    local base=$(( DOCS / SHARD_NUM ))
    local rem=$(( DOCS % SHARD_NUM ))
    local off=$(( i * base ))
    local cnt="${base}"
    [[ "${i}" -eq $(( SHARD_NUM - 1 )) ]] && cnt=$(( base + rem ))
    echo "${off} ${cnt}"
}

# Key in NOTIFY_API_KEY (the gitignored .notify.env). NOTIFY=1 with no key just skips the mail.
NOTIFY="${NOTIFY:-0}"
NOTIFY_URL="${NOTIFY_URL:-https://mail.xihe.me/api/send/notification}"
NOTIFY_API_KEY="${NOTIFY_API_KEY:-}"

DO_COMPILE=1
DO_CONVERT_QUERY=1
DO_CONVERT_DATA=1
DO_BUILD_INDEX=1
DO_SEARCH=1
PROFILE_MODE=0
# Cross-run MEDIAN: a single 10k-query run's p99 is not reproducible run to run.
REPEAT="${REPEAT:-1}"
# 1=pooled (default), 0=raw per-query aclrtMalloc/Free. Same binary either way; pair with --repeat.
POOL_POSTINGS="${NPUR_POOL_POSTINGS:-1}"
MEM_REPORT="${MEM_REPORT:-0}"
NPU_UTIL="${NPU_UTIL:-0}"

usage() {
    cat <<USAGE
Usage: ./run.sh [options]

Options:
  -c, --compile <0|1>        1=clean all previous build outputs and rebuild; 0=reuse build outputs. Default: ${DO_COMPILE}
  --convert-query <0|1>      Convert QueryData_10000.txt to fvecs/filter/topk. Default: ${DO_CONVERT_QUERY}
  --convert-data <0|1>       Convert dataset.bin to builder input. Default: ${DO_CONVERT_DATA}
  --build-index <0|1>        Build new-project full-recall index. Default: ${DO_BUILD_INDEX}
  --search <0|1>             Run fr_search. Default: ${DO_SEARCH}
  --profile <0|1|2>          1=per-stage NPU timing (NPUR_PERF): aggregate [Perf] lines into an
                             avg-per-stage table (add PERF_DUMP=1 for p99 tail attribution;
                             single-card only -- parallel shards scramble its buckets).
                             2=same timing, quiet: convert/build chatter goes to a log file and
                             the per-query recall mismatch lines collapse to a count. Default: ${PROFILE_MODE}
  --repeat <N>               Run the search N times and print a cross-run median of the
                             end-to-end latency stats (de-noises the p99 tail). Default: ${REPEAT}
  --pool-postings <0|1>      Toggle the devicePostings pooling optimization (engine env
                             NPUR_POOL_POSTINGS): 1=pooled, 0=raw per-query malloc baseline.
                             Same binary either way; pair with --repeat for an A/B. Default: ${POOL_POSTINGS}
  --mem-report <0|1>         After the search, print host peak RSS + per-device NPU HBM
                             usage ([MEM] lines). Default: ${MEM_REPORT}
  --npu-util <0|1>           Background-sample per-device NPU AI Core usage (%) with npu-smi
                             during the search; report avg/max/min ([NPUUTIL] lines). Default: ${NPU_UTIL}
  -h, --help                 Show this help

Important environment overrides:
  CANN or ASCEND_CANN_PACKAGE_PATH   CANN toolkit path. Current: ${CANN}
  SOC_VERSION                        Ascend SoC. Current: ${SOC_VERSION}
  DATASET_FILE                       HYDSET2 dataset cache. Current: ${DATASET_FILE}
  TAG_MAP_FILE                       Original tag_map.bin. Current: ${TAG_MAP_FILE}
  QUERYDATA_FILE                     Original QueryData_10000.txt. Current: ${QUERYDATA_FILE}
  RAPIDJSON_INCLUDE_DIR              Directory containing rapidjson/document.h. Current: ${RAPIDJSON_INCLUDE_DIR:-not found}
  GFLAGS_LIBRARY_DIR                 Directory containing libgflags.so, if not in a standard path. Current: ${GFLAGS_LIBRARY_DIR:-auto}
  PROTOBUF_ROOT                      Install prefix for find_package(Protobuf). Current: ${PROTOBUF_ROOT:-auto}
  PROTOBUF_LIBRARY                   Full path to libprotobuf.so, if the prefix layout does not fit. Current: ${PROTOBUF_LIBRARY:-auto}
  PROTOBUF_INCLUDE_DIR               Directory containing google/protobuf/. Current: ${PROTOBUF_INCLUDE_DIR:-auto}
  PROTOC                             Full path to the protoc binary. Current: ${PROTOC:-auto}
  EXTRA_LD_LIBRARY_PATH              Extra runtime library dirs, colon-separated. Current: ${EXTRA_LD_LIBRARY_PATH:-empty}
  WORK_DIR                           Intermediate output root. Current: ${WORK_DIR}
  DOCS                               Docs to index, 0=all. Current: ${DOCS}
  DOC_NUM_PER_SEGMENT                Builder segment size. Current: ${DOC_NUM_PER_SEGMENT}
  DENSITY_THRESHOLD                  Converter bitlist/bitset cutoff (density<thr -> bitlist). Lower = more bitset, less runtime conversion, bigger index. Current: ${DENSITY_THRESHOLD}
  FULL_RECALL_TEXT_FILTER_BLOCK_DIM  Runtime filter AI-core block dim. Current: ${FULL_RECALL_TEXT_FILTER_BLOCK_DIM}
  SCORER_BLOCK_DIM                   Runtime scorer (MMad) AI-core block dim. Current: ${SCORER_BLOCK_DIM}
  NUM_QUERIES                        Queries to run, 0=all converted queries. Current: ${NUM_QUERIES}
  RECALL_QUERIES                     CPU brute-force recall checks, 0=off. Current: ${RECALL_QUERIES}
  HBM_SAMPLE                         Poll NPU memory every N seconds DURING the search and report the peak, 0=off. The before/after reports bracket the process and cannot see it. Costs one npu-smi fork per poll, so leave it off while measuring. Current: ${HBM_SAMPLE:-0}
  DEVICE_ID                          NPU device id (single-card). Current: ${DEVICE_ID}
  DEVICE_IDS                         comma-separated device ids for multi-card (e.g. 0,1); splits the
                                     corpus into one shard per card. Requires DOCS>0. Current: ${DEVICE_IDS:-unset (single-card)}
  SHARD_INDEX_DIRS                   comma-separated pre-built shard index dirs (one per DEVICE_IDS entry);
                                     reuse them and skip convert+build. Current: ${SHARD_INDEX_DIRS:-unset}
  RECALL_REF_FILE                    optional recall ground-truth cache (fr_search --recall_ref_file). Current: ${RECALL_REF_FILE:-unset}
  NOTIFY                             1=email a summary after a successful search. Current: ${NOTIFY}
  NOTIFY_API_KEY                     API key for NOTIFY (put in gitignored .notify.env). Current: ${NOTIFY_API_KEY:+set}${NOTIFY_API_KEY:-unset}
  NOTIFY_URL                         Notification endpoint. Current: ${NOTIFY_URL}
USAGE
}

parse_bool_arg() {
    local name="$1"
    local value="$2"
    if [[ "${value}" != "0" && "${value}" != "1" ]]; then
        echo "[ERROR] ${name} only accepts 0 or 1, got: ${value}" >&2
        exit 1
    fi
}

parse_profile_arg() {
    local name="$1"
    local value="$2"
    if [[ "${value}" != "0" && "${value}" != "1" && "${value}" != "2" ]]; then
        echo "[ERROR] ${name} only accepts 0, 1 or 2, got: ${value}" >&2
        exit 1
    fi
}

SKIPPED_STEPS=()

run_build_tool() {
    if [[ "${VERBOSE_BUILD:-0}" == "1" ]]; then
        "$@"
        return $?
    fi
    local build_log="${RESULT_DIR}/log/build.log"
    mkdir -p "${RESULT_DIR}/log"
    local rc=0
    "$@" >> "${build_log}" 2>&1 || rc=$?
    if [[ "${rc}" -ne 0 ]]; then
        echo "[ERROR] $(basename "$1") failed (rc=${rc}); last 30 lines of ${build_log}:" >&2
        tail -30 "${build_log}" >&2 || true
    fi
    return "${rc}"
}

# Export <engine var>=1 when the wrapper switch is "1". A third argument is the wrapper's
# default; WITHOUT one the wrapper must already be set, so `set -u` still catches a switch that
# lost its definition. Always returns 0 so an off switch does not trip `set -e`.
npur_switch() {
    local val
    if [[ $# -ge 3 ]]; then
        val="${!1:-$3}"
    elif [[ -z "${!1+set}" ]]; then
        echo "[ERROR] $1 is unset; it has no default here and must be assigned above" >&2
        exit 1
    else
        val="${!1}"
    fi
    [[ "${val}" == "1" ]] && export "$2=1"
    return 0
}

parse_posint_arg() {
    local name="$1"
    local value="$2"
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] ${name} only accepts a positive integer, got: ${value}" >&2
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--compile)
            [[ $# -ge 2 ]] || { usage; exit 1; }
            parse_bool_arg "$1" "$2"
            DO_COMPILE="$2"
            shift 2
            ;;
        --convert-query)
            [[ $# -ge 2 ]] || { usage; exit 1; }
            parse_bool_arg "$1" "$2"
            DO_CONVERT_QUERY="$2"
            shift 2
            ;;
        --convert-data)
            [[ $# -ge 2 ]] || { usage; exit 1; }
            parse_bool_arg "$1" "$2"
            DO_CONVERT_DATA="$2"
            shift 2
            ;;
        --build-index)
            [[ $# -ge 2 ]] || { usage; exit 1; }
            parse_bool_arg "$1" "$2"
            DO_BUILD_INDEX="$2"
            shift 2
            ;;
        --search)
            [[ $# -ge 2 ]] || { usage; exit 1; }
            parse_bool_arg "$1" "$2"
            DO_SEARCH="$2"
            shift 2
            ;;
        --profile)
            [[ $# -ge 2 ]] || { usage; exit 1; }
            parse_profile_arg "$1" "$2"
            PROFILE_MODE="$2"
            shift 2
            ;;
        --repeat)
            [[ $# -ge 2 ]] || { usage; exit 1; }
            parse_posint_arg "$1" "$2"
            REPEAT="$2"
            shift 2
            ;;
        --pool-postings)
            [[ $# -ge 2 ]] || { usage; exit 1; }
            parse_bool_arg "$1" "$2"
            POOL_POSTINGS="$2"
            shift 2
            ;;
        --mem-report)
            [[ $# -ge 2 ]] || { usage; exit 1; }
            parse_bool_arg "$1" "$2"
            MEM_REPORT="$2"
            shift 2
            ;;
        --npu-util)
            [[ $# -ge 2 ]] || { usage; exit 1; }
            parse_bool_arg "$1" "$2"
            NPU_UTIL="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ "${EXTERNAL_SHARDS}" == "1" ]]; then
    if [[ "${SHARD_NUM}" -ne "${#SHARD_INDEX_LIST[@]}" ]]; then
        echo "[ERROR] SHARD_INDEX_DIRS has ${#SHARD_INDEX_LIST[@]} dir(s) but DEVICE_IDS has ${SHARD_NUM} id(s);" >&2
        echo "        they must match (one device per shard index dir). Set DEVICE_IDS accordingly." >&2
        exit 1
    fi
    DO_CONVERT_DATA=0
    DO_BUILD_INDEX=0
fi

require_file() {
    local path="$1"
    local label="$2"
    if [[ ! -f "${path}" ]]; then
        echo "[ERROR] Missing ${label}: ${path}" >&2
        exit 1
    fi
}

require_executable() {
    local path="$1"
    if [[ ! -x "${path}" ]]; then
        echo "[ERROR] Executable not found: ${path}" >&2
        echo "        Build first with ./run.sh -c 1, or check the previous build logs." >&2
        exit 1
    fi
}

RUNTIME_LIBRARY_DIRS=()

append_runtime_library_dir() {
    local path="$1"
    local existing
    [[ -n "${path}" ]] || return 0
    for existing in "${RUNTIME_LIBRARY_DIRS[@]}"; do
        [[ "${existing}" == "${path}" ]] && return 0
    done
    RUNTIME_LIBRARY_DIRS+=("${path}")
}

append_runtime_library_path_list() {
    local paths="$1"
    local old_ifs="${IFS}"
    local parts=()
    local part
    [[ -n "${paths}" ]] || return 0
    IFS=':'
    read -r -a parts <<< "${paths}"
    IFS="${old_ifs}"
    for part in "${parts[@]}"; do
        append_runtime_library_dir "${part}"
    done
}

join_by_colon() {
    local old_ifs="${IFS}"
    IFS=':'
    printf '%s' "$*"
    IFS="${old_ifs}"
}

join_by_semicolon() {
    local old_ifs="${IFS}"
    IFS=';'
    printf '%s' "$*"
    IFS="${old_ifs}"
}

NPU_UTIL_DIR=""
NPU_UTIL_PID=""
NPU_UTIL_DEVS=()

# Sampled BEFORE fr_search starts, because --mem-report's aclrtGetMemInfo reports DEVICE-WIDE HBM
# and cannot tell our allocations from another tenant's. The two samples BRACKET the search rather
# than covering it, so the pair shows whether the cards came back to where they started, not what
# we used. HBM_SAMPLE=<seconds> polls WHILE it runs and reports the per-device peak instead.
npu_hbm_sample_start() {
    [[ "${HBM_SAMPLE:-0}" == "0" ]] && return 0
    command -v npu-smi >/dev/null 2>&1 || return 0
    HBM_SAMPLE_FILE="$(mktemp)"
    (
        while :; do
            npu-smi info 2>/dev/null || true
            sleep "${HBM_SAMPLE}"
        done
    ) > "${HBM_SAMPLE_FILE}" 2>/dev/null &
    HBM_SAMPLE_PID=$!
    echo "[Step] sampling NPU memory every ${HBM_SAMPLE}s while the search runs (HBM_SAMPLE)"
}

npu_hbm_sample_stop() {
    [[ -z "${HBM_SAMPLE_PID:-}" ]] && return 0
    kill "${HBM_SAMPLE_PID}" 2>/dev/null || true
    wait "${HBM_SAMPLE_PID}" 2>/dev/null || true
    HBM_SAMPLE_PID=""
    echo "[Step] NPU memory peak DURING the search:"
    # Process row "| 3  0 | 594489 | fr_search | 18629 |": $2 is "<npu> <chip>", $3 a pid, $5 the MB.
    awk -F"|" '
        $2 ~ /^ *[0-9]+ +[0-9]+ *$/ && $3 ~ /^ *[0-9]+ *$/ {
            split($2,a," "); split($4,nm," "); split($5,m," ")
            if (nm[1] != "" && m[1]+0 > peak[a[1]]) { peak[a[1]]=m[1]+0; who[a[1]]=nm[1] }
        }
        END {
            if (length(peak) == 0) {
                print "  (no process rows sampled -- the run may be shorter than one HBM_SAMPLE interval)"
                exit
            }
            for (d in peak) printf "  NPU %-4s %-14s peak %6d MB\n", d, who[d], peak[d]
        }' "${HBM_SAMPLE_FILE}" || true
    rm -f "${HBM_SAMPLE_FILE}"
}

npu_hbm_report() {
    local label="$1"
    command -v npu-smi >/dev/null 2>&1 || return 0
    local out parsed
    out="$(npu-smi info 2>/dev/null)" || return 0
    parsed="$(printf '%s\n' "${out}" | awk -F'|' '
        # Chip header "| 3  910B3 | OK |": process rows look the same in $2, so key off $3 being alphabetic.
        $2 ~ /^ *[0-9]+ +[0-9A-Za-z]+ *$/ && $3 ~ /^ *[A-Za-z]/ { split($2,a," "); npu=a[1]; next }
        # Bus-Id row: HBM sits at the end of $4 as "21990/ 65536" or "4090 / 65536".
        $3 ~ /[0-9A-Fa-f]+:[0-9A-Fa-f]+:[0-9A-Fa-f]+\./ {
            if (match($4, /[0-9]+ *\/ *[0-9]+ *$/)) {
                hbm=substr($4, RSTART, RLENGTH); gsub(/ /,"",hbm);
                split(hbm,h,"/"); used[npu]=h[1]; tot[npu]=h[2]; order[++n]=npu
            } next }
        $2 ~ /^ *[0-9]+ +[0-9]+ *$/ && $3 ~ /^ *[0-9]+ *$/ {
            split($2,a," "); split($4,nm," "); split($5,m," ");
            proc[a[1]] = proc[a[1]] nm[1] "(" m[1] "MB) "; pmem[a[1]] += m[1] }
        END {
            if (n == 0) exit 1
            printf "  %-4s %13s %7s  %-24s %s\n","NPU","HBM(MB)","used","processes","not-ours"
            for(i=1;i<=n;i++){ d=order[i]
                printf "  %-4s %6d/%-6d %6.1f%%  %-24s %d MB\n", d, used[d], tot[d],
                       used[d]/tot[d]*100, (proc[d]==""?"-":proc[d]), used[d]-pmem[d] }
        }' 2>/dev/null)" || parsed=""  # awk exits 1 on an unrecognised table; set -e must not see it
    echo "[Step] NPU memory ${label} the search (fr_search is not running at this point; see HBM_SAMPLE):"
    if [[ -n "${parsed}" ]]; then
        printf '%s\n' "${parsed}"
        echo "  (note: npu-smi ids are physical; DEVICE_IDS are ACL logical ids and may differ)"
    else
        printf '%s\n' "${out}" | sed 's/^/  /'
    fi
}

npu_util_start() {
    [[ "${NPU_UTIL}" == "1" ]] || return 0
    if ! command -v npu-smi >/dev/null 2>&1; then
        echo "[NPUUTIL] npu-smi not found; --npu-util skipped." >&2
        NPU_UTIL=0
        return 0
    fi
    if [[ "${MULTICARD}" == "1" ]]; then
        IFS=',' read -r -a NPU_UTIL_DEVS <<< "${DEVICE_IDS}"
    else
        NPU_UTIL_DEVS=("${DEVICE_ID}")
    fi
    NPU_UTIL_DIR="$(mktemp -d)"
    # The id MUST come from the chip HEADER row ("| 3  910B3 | OK |"), not the Bus-Id row: field 2
    # there is the Chip index, which is 0 on every card, so all cards would be labelled 0.
    (
        set +e
        while :; do
            npu-smi info 2>/dev/null | awk -F'|' '
                $2 ~ /^ *[0-9]+ +[0-9A-Za-z]+ *$/ && $3 ~ /^ *[A-Za-z]/ { split($2,a," "); npu=a[1]; next }
                $3 ~ /[0-9A-Fa-f]+:[0-9A-Fa-f]+:[0-9A-Fa-f]+\./ {
                    split($4, b, " ");
                    hbm = "";
                    if (match($4, /[0-9]+ *\/ *[0-9]+ *$/)) {
                        hbm = substr($4, RSTART, RLENGTH); gsub(/ /, "", hbm); split(hbm, h, "/"); hbm = h[1]
                    }
                    if (npu != "" && b[1] ~ /^[0-9]+$/ && hbm != "") print npu, b[1], hbm
                }' >> "${NPU_UTIL_DIR}/raw" 2>/dev/null
            sleep 0.5
        done
    ) &
    NPU_UTIL_PID=$!
    trap '[[ -n "${NPU_UTIL_PID}" ]] && kill "${NPU_UTIL_PID}" 2>/dev/null' EXIT
    echo "[NPUUTIL] sampling AI Core usage for devices: ${NPU_UTIL_DEVS[*]}"
}

# Started just before the timed runs and killed right after, so it covers exactly them.
NPU_LOAD_PID=""
npu_load_start() {
    [[ -n "${NPU_LOAD_CARDS}" ]] || return 0
    local bin="${ROOT_DIR}/build/fr_npuload"
    if [[ ! -x "${bin}" ]]; then
        echo "[ERROR] NPU_LOAD_CARDS set but ${bin} is missing; rebuild with --compile 1." >&2
        exit 1
    fi
    "${bin}" --devices "${NPU_LOAD_CARDS}" --duty "${NPU_LOAD_DUTY}" \
             --period_ms "${NPU_LOAD_PERIOD_MS}" \
             --compute_workers "${NPU_LOAD_COMPUTE_WORKERS}" \
             --copy_workers "${NPU_LOAD_COPY_WORKERS}" \
             --copy_buffer_mb "${NPU_LOAD_COPY_BUFFER_MB}" \
             --copy_gap_ms "${NPU_LOAD_COPY_GAP_MS}" > "${RESULT_DIR}/log/npuload.log" 2>&1 &
    NPU_LOAD_PID=$!
    sleep 3  # let it allocate and reach steady state before the first timed query
    if ! kill -0 "${NPU_LOAD_PID}" 2>/dev/null; then
        echo "[ERROR] fr_npuload died on startup; see ${RESULT_DIR}/log/npuload.log" >&2
        exit 1
    fi
    echo "[NpuLoad] real contention on cards ${NPU_LOAD_CARDS}: duty ${NPU_LOAD_DUTY} of every ${NPU_LOAD_PERIOD_MS}ms, ${NPU_LOAD_COMPUTE_WORKERS} compute + ${NPU_LOAD_COPY_WORKERS} copy worker(s) (pid ${NPU_LOAD_PID}, log: ${RESULT_DIR}/log/npuload.log)"
}

npu_load_stop() {
    [[ -n "${NPU_LOAD_PID}" ]] || return 0
    kill "${NPU_LOAD_PID}" 2>/dev/null || true
    wait "${NPU_LOAD_PID}" 2>/dev/null || true
    NPU_LOAD_PID=""
}

npu_util_stop() {
    [[ -n "${NPU_UTIL_PID}" ]] || return 0
    # || true: wait on a killed job returns 143 and kill of a gone job is non-zero; `set -e` would abort.
    kill "${NPU_UTIL_PID}" 2>/dev/null || true
    wait "${NPU_UTIL_PID}" 2>/dev/null || true
    NPU_UTIL_PID=""
    # npu-smi ids are PHYSICAL and need not line up with DEVICE_IDS (ACL logical) -- report every card.
    echo "[NPUUTIL] ==== per-card AI Core (%) and HBM (MB) during search ===="
    awk '{ n[$1]++; s[$1]+=$2; if($2>umx[$1]) umx[$1]=$2;
           if(hmn[$1]==""||$3<hmn[$1]) hmn[$1]=$3; if($3>hmx[$1]) hmx[$1]=$3 }
         END {
           if (length(n) == 0) { print "[NPUUTIL] no samples (npu-smi table parse failed?)"; exit }
           for (d in n)
             printf "[NPUUTIL] npu%s: AICore avg=%.1f%% max=%.0f%%   HBM %d->%d MB (peak %d)  (n=%d)\n",
                    d, s[d]/n[d], umx[d], hmn[d], hmx[d], hmx[d], n[d]
         }' "${NPU_UTIL_DIR}/raw" 2>/dev/null | sort
    rm -rf "${NPU_UTIL_DIR}"
    NPU_UTIL_DIR=""
}

if [[ "${PROFILE_MODE}" != "0" ]]; then
    echo "[Info] --profile ${PROFILE_MODE}: per-stage timing enabled (NPUR_PERF=1); the search"
    echo "       step aggregates [Perf] RecordGuard lines into an avg-per-stage table."
    [[ "${PROFILE_MODE}" == "2" ]] && \
        echo "       quiet: convert/build output -> log file, per-query recall lines -> count."
fi

# The KEY tokens ("Mode:", "Aggregate:", ...) are grep targets in the run docs -- keep them verbatim.
brow() { printf '  %-26s%s\n' "$1" "$2"; }
bsep() { printf -- '--[ %s ]%s\n' "$1" "$(printf -- '-%.0s' $(seq 1 $((72 - ${#1} - 7))))"; }
printf -- '=%.0s' $(seq 1 72); echo
echo "  FullRecall wrapper  |  ${SOC_VERSION}  |  $(date '+%F %T')"
printf -- '=%.0s' $(seq 1 72); echo
bsep "environment"
brow "Root:" "${ROOT_DIR}"
brow "CANN:" "${CANN}"
brow "RapidJSON:" "${RAPIDJSON_INCLUDE_DIR:-not found}"
bsep "data"
brow "Dataset:" "${DATASET_FILE}"
if [[ "${DO_CONVERT_QUERY}" == "1" ]]; then
    brow "Tag map:" "${TAG_MAP_FILE}"
    brow "QueryData:" "${QUERYDATA_FILE}"
else
    brow "Queries:" "reusing ${QUERY_OUT_DIR} (conversion skipped)"
fi
brow "Work dir:" "${WORK_DIR}"
if [[ -n "${RECALL_REF_FILE}" ]]; then
    brow "Recall ref:" "${RECALL_REF_FILE} (cached, all queries)"
else
    brow "Recall ref:" "unset (spot-check ${RECALL_QUERIES} queries)"
fi
bsep "run"
if [[ "${ROUND_ROBIN}" == "1" ]]; then
    if [[ "${STATIC_ASSIGN}" == "1" ]]; then
        brow "Mode:" "ROUND-ROBIN baseline, FIXED assignment (${SHARD_NUM} cards, each = FULL corpus; chunk k bound to card k%N, no work stealing, no merge)"
    else
        brow "Mode:" "ROUND-ROBIN baseline, dynamic (${SHARD_NUM} cards, each = FULL corpus; queries work-stolen to idle cards, no merge)"
    fi
elif [[ "${MULTICARD}" == "1" && "${SHARD_GROUP_SIZE}" -gt 0 && "${SHARD_NUM}" -ge "${SHARD_GROUP_SIZE}" ]]; then
    _grp_pipe=""
    [[ "${PIPELINE}" == "1" ]] && _grp_pipe=", pipelined"
    _grp_disp="groups work-steal"
    [[ "${STATIC_ASSIGN}" == "1" ]] && _grp_disp="FIXED assignment, chunk k bound to group k%G, no work stealing"
    brow "Mode:" "shard GROUPS ($((SHARD_NUM / SHARD_GROUP_SIZE)) groups x ${SHARD_GROUP_SIZE} shards on ${DEVICE_IDS}${_grp_pipe}; a query fans out inside one group, ${_grp_disp})"
elif [[ "${MULTICARD}" == "1" && "${PIPELINE}" == "1" ]]; then
    brow "Mode:" "multi-card PIPELINE (${SHARD_NUM} shards on ${DEVICE_IDS}; batch N Extract overlaps batch N+1 Device)"
elif [[ "${MULTICARD}" == "1" ]]; then
    brow "Mode:" "multi-card (${SHARD_NUM} shards on devices ${DEVICE_IDS})"
else
    brow "Mode:" "single-card (device ${DEVICE_ID})"
fi
if [[ "${BATCH_SIZE}" != "1" ]]; then
    brow "Batch:" "${BATCH_SIZE} queries/batch (avg = ms/query = 1/throughput; p99 not a tail here)"
fi
brow "Cores:" "scorer ${SCORER_BLOCK_DIM} (Cube) + filter ${FULL_RECALL_TEXT_FILTER_BLOCK_DIM} (Vector) + aggregator ${AGGREGATOR_BLOCK_DIM}"
if [[ -n "${CARD_CPUS:-}" ]]; then
    brow "Affinity:" "per-card pinning ${CARD_CPUS} (CARD_CPUS, one core list per DEVICE_IDS entry; keeps each card's DMA and host buffers on its own NUMA node)"
fi
if [[ "${TARGET_QPS}" != "0" ]]; then
    brow "Load:" "open-loop, offered ${TARGET_QPS} qps (TARGET_QPS; latency measured from each query's arrival, so queueing counts)"
fi
if [[ -n "${NPU_LOAD_CARDS}" ]]; then
    brow "Fluctuation:" "REAL contention on cards ${NPU_LOAD_CARDS} (NPU_LOAD_CARDS; fr_npuload takes Cube + HBM ${NPU_LOAD_DUTY} of every ${NPU_LOAD_PERIOD_MS}ms)"
fi
if [[ -n "${SLOW_CARDS}" ]]; then
    brow "Fluctuation:" "cards ${SLOW_CARDS} run ${SLOW_FACTOR}x slower for ${SLOW_DUTY} of every ${SLOW_PERIOD_MS}ms (SLOW_CARDS; host-side delay injection)"
fi
if [[ -n "${NPU_LOAD_CARDS}" && -n "${SLOW_CARDS}" ]]; then
    echo "[ERROR] NPU_LOAD_CARDS and SLOW_CARDS are both set; the disturbance would be applied twice." >&2
    echo "        Use NPU_LOAD_CARDS (real contention) for acceptance numbers, or SLOW_CARDS alone." >&2
    exit 1
fi
bsep "optimizations"
# Every row starts with two spaces + the CAPS env name, so what is ON can be grepped.
if [[ "${POOL_POSTINGS}" == "1" ]]; then
    brow "POOL_POSTINGS=1" "posting tables pooled"
else
    brow "POOL_POSTINGS=0" "raw-malloc baseline"
fi
case "${NPUR_PARALLEL_SCORE_FILTER:-0}:${NPUR_OVERLAP_POSTEXPR:-1}" in
    1:*) brow "PARALLEL_SCORE_FILTER=1" "scorer/filter on two threads (worse p99)" ;;
    *:0) brow "OVERLAP_POSTEXPR=0" "serial via executor (pre-overlap baseline)" ;;
    *:2) brow "OVERLAP_POSTEXPR=2" "no overlap (control arm)" ;;
    *)   brow "OVERLAP_POSTEXPR=1" "post-expr built inside the scorer kernel window" ;;
esac
[[ "${NPUR_OVERLAP_POSTING:-0}" == "1" ]] && \
    brow "OVERLAP_POSTING=1" "posting prep inside the scorer window too"
[[ "${STREAM_MERGE:-0}" == "1" ]] && \
    brow "STREAM_MERGE=1" "k-way merge folds shards as they finish"
if [[ "${PACKED_SORT:-0}" == "1" || "${RADIX_SORT:-0}" == "1" ]]; then
    _hostsort="host sort on packed 8-byte keys"
    [[ "${RADIX_SORT:-0}" == "1" ]] && _hostsort="${_hostsort} + radix"
    brow "PACKED_SORT=1" "${_hostsort}"
fi
[[ -n "${SHARD_TOPK_RATIO:-}" && "${SHARD_TOPK_RATIO}" != "1" && "${SHARD_TOPK_RATIO}" != "1.0" ]] && \
    brow "SHARD_TOPK_RATIO=${SHARD_TOPK_RATIO}" "each shard aggregates only topK x ratio (merge returns full topK)"
[[ "${BATCH_AGGREGATE:-0}" == "1" ]] && \
    brow "BATCH_AGGREGATE=1" "aggregation launched/synced once per chunk, not per query"
[[ "${BATCH_FILTER:-0}" == "1" ]] && \
    brow "BATCH_FILTER=1" "filter launched/synced once per chunk, not per query"
[[ "${TOPK_CONCURRENT:-0}" == "1" ]] && \
    brow "TOPK_CONCURRENT=1" "a batch's single-core TopK kernels run side by side on their own streams"
[[ "${POOL_SMALL_H2D:-0}" == "1" ]] && \
    brow "POOL_SMALL_H2D=1" "scorer query matrix + filter postExpr from pools, no per-batch aclrtMalloc/Free"
[[ "${EARLY_FILTER_PREP:-0}" == "1" ]] && \
    brow "EARLY_FILTER_PREP=1" "filter setup + postExpr H2D inside the scorer window (pays at BS=1, not BS=4)"
[[ "${SHARE_FILTER_STREAM:-0}" == "1" ]] && \
    brow "SHARE_FILTER_STREAM=1" "aggregator queues behind the filter kernels; one sync for both"
[[ "${TOPK_COUNTS_IN_PLACE:-0}" == "1" ]] && \
    brow "TOPK_COUNTS_IN_PLACE=1" "TopK reads the aggregator's per-core counts on device; no re-upload"
[[ "${FUSE_AGG_TOPK:-0}" == "1" ]] && \
    brow "FUSE_AGG_TOPK=1" "TopK queued behind the aggregator, one sync and one count read for both"
[[ "${DEFER_CONV_SYNC:-0}" == "1" ]] && \
    brow "DEFER_CONV_SYNC=1" "in-window bitlist conversions launched without a wait; drained after the scorer"
[[ "${STREAM_POOL:-0}" == "1" ]] && \
    brow "STREAM_POOL=1" "each card gets the service's stream pool; kernels leave the default stream"
[[ "${OVERLAP_FILTER:-0}" == "1" ]] && \
    brow "OVERLAP_FILTER=1" "filter kernels launched inside the scorer window; Vector runs beside Cube"
[[ "${EXTRACT_FAST:-0}" == "1" ]] && \
    brow "EXTRACT_FAST=1" "host extract: 4-pass radix on the score bytes + prefetched global-id lookups"
[[ -n "${AGG_CONCURRENT:-}" && "${AGG_CONCURRENT}" != "1" && "${AGG_CONCURRENT}" != "0" ]] && \
    brow "AGG_CONCURRENT=${AGG_CONCURRENT}" "a batch's Aggregator kernels spread over ${AGG_CONCURRENT} streams"
[[ "${SCORER_TRIM_WRITE:-0}" == "1" ]] && \
    brow "SCORER_TRIM_WRITE=1" "scorer writes back only the real query rows"
[[ "${PARALLEL_LOAD:-0}" == "1" ]] && \
    brow "PARALLEL_LOAD=1" "index shards load on one thread each"
if [[ "${DENSITY_NONZERO}" == "1" ]]; then
    case "${SPARSE_DIRECT}" in
    1) brow "SPARSE_DIRECT=1" "OR nodes read sparse postings in place; no bitlist->bitset pass" ;;
    *) brow "SPARSE_DIRECT=0" "every sparse posting is converted to a 16KB bitset first" ;;
    esac
    case "${OVERLAP_POSTING_BITLIST}" in
    1) brow "OVERLAP_POSTING_BITLIST=1" "their posting tables are built in the scorer window" ;;
    *) brow "OVERLAP_POSTING_BITLIST=0" "posting tables serialised after the scorer sync" ;;
    esac
    case "${OR_NO_TILE_CHECK}" in
    1) brow "OR_NO_TILE_CHECK=1" "ApplySparse places units without the dead tile bound" ;;
    *) brow "OR_NO_TILE_CHECK=0" "ApplySparse bounds-checks every unit against its tile" ;;
    esac
    case "${OR_ALIGNED_COPY:-${SPARSE_DIRECT}}" in
    1) brow "OR_ALIGNED_COPY=1" "sparse CopyIn rounded to a block; no scalar tail" ;;
    *) brow "OR_ALIGNED_COPY=0" "DataCopyPadCustom walks each operand's tail" ;;
    esac
    case "${CLASSIFY_FAST:-${SPARSE_DIRECT}}" in
    1) brow "CLASSIFY_FAST=1" "OR operands written straight to the table" ;;
    *) brow "CLASSIFY_FAST=0" "every pair goes through emit's branch tree" ;;
    esac
fi
case "${LAZY_POSTING_WEIGHTS:-1}" in
1) brow "LAZY_POSTING_WEIGHTS=1" "posting weights built only if a switch reads them" ;;
*) brow "LAZY_POSTING_WEIGHTS=0" "weights built and looked up whether read or not" ;;
esac
printf -- '=%.0s' $(seq 1 72); echo

dbg "banner done; checking dataset: ${DATASET_FILE}"
require_file "${DATASET_FILE}" "dataset.bin"
dbg "dataset ok; MULTICARD=${MULTICARD} EXTERNAL_SHARDS=${EXTERNAL_SHARDS} DOCS=${DOCS} SHARD_NUM=${SHARD_NUM}"

if [[ "${MULTICARD}" == "1" && "${EXTERNAL_SHARDS}" == "0" && "${DOCS}" -eq 0 ]]; then
    echo "[ERROR] multi-card build (DEVICE_IDS=${DEVICE_IDS}) needs DOCS=<total corpus size> so the" >&2
    echo "        corpus can be split into ${SHARD_NUM} equal shards. Set DOCS, or pass" >&2
    echo "        SHARD_INDEX_DIRS to search pre-built shards instead." >&2
    exit 1
fi

dbg "sourcing CANN setenv: ${CANN}/bin/setenv.bash (exists=$([[ -f "${CANN}/bin/setenv.bash" ]] && echo yes || echo no))"
if [[ -f "${CANN}/bin/setenv.bash" ]]; then
    # Without the guard a setenv.bash whose last command returns non-zero would kill the run silently.
    set +u
    set +e
    # shellcheck disable=SC1090
    source "${CANN}/bin/setenv.bash"
    _setenv_rc=$?
    set -e
    set -u
    if [[ "${_setenv_rc}" -ne 0 ]]; then
        echo "[Warn] ${CANN}/bin/setenv.bash exited ${_setenv_rc}; continuing (library paths are set explicitly below)." >&2
    fi
    # A non-zero rc with all vars set is harmless; EMPTY vars mean it stopped before configuring CANN.
    dbg "setenv rc=${_setenv_rc} ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-<unset>}"
    dbg "  ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-<unset>}"
    dbg "  ASCEND_AICPU_PATH=${ASCEND_AICPU_PATH:-<unset>}"
    dbg "  ASCEND_TOOLKIT_HOME=${ASCEND_TOOLKIT_HOME:-<unset>}"
fi
dbg "setenv block done"

append_runtime_library_dir "${ROOT_DIR}/build_device/lib"
append_runtime_library_dir "${ROOT_DIR}/lib64"
append_runtime_library_dir "${ROOT_DIR}/lib"
append_runtime_library_dir "${ROOT_DIR}/third_party/lib64"
append_runtime_library_dir "${ROOT_DIR}/third_party/lib"
append_runtime_library_dir "${CANN}/lib64"
append_runtime_library_dir "${CANN}/opp/vendors/aicpu_mask/op_impl/cpu/aicpu_kernel/impl"
append_runtime_library_dir "${GFLAGS_LIBRARY_DIR}"
append_runtime_library_dir "${PROTOBUF_LIBRARY_DIR}"
append_runtime_library_dir "${ABSL_LIBRARY_DIR}"
append_runtime_library_path_list "${EXTRA_LD_LIBRARY_PATH}"
append_runtime_library_dir "/usr/local/lib64"
append_runtime_library_dir "/usr/local/lib"
append_runtime_library_dir "/usr/lib64"
append_runtime_library_dir "/usr/lib"
append_runtime_library_dir "/usr/lib/aarch64-linux-gnu"
append_runtime_library_dir "/opt/OpenBLAS/lib"
append_runtime_library_dir "/lib64"
append_runtime_library_dir "/lib"

RUNTIME_LD_LIBRARY_PATH="$(join_by_colon "${RUNTIME_LIBRARY_DIRS[@]}")"
RUNTIME_RPATH_DIRS="$(join_by_semicolon "${RUNTIME_LIBRARY_DIRS[@]}")"
if [[ -n "${RUNTIME_LD_LIBRARY_PATH}" ]]; then
    export LD_LIBRARY_PATH="${RUNTIME_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"
fi
dbg "Host libs: ${RUNTIME_LD_LIBRARY_PATH}"
echo "[Info] host library path: ${#RUNTIME_LIBRARY_DIRS[@]} dirs on LD_LIBRARY_PATH (DEBUG=1 to print them)"
dbg "entering stages: compile=${DO_COMPILE} convert_query=${DO_CONVERT_QUERY} convert_data=${DO_CONVERT_DATA} build_index=${DO_BUILD_INDEX} search=${DO_SEARCH}"

if [[ "${DO_COMPILE}" == "1" ]]; then
    if [[ -z "${RAPIDJSON_INCLUDE_DIR}" || ! -f "${RAPIDJSON_INCLUDE_DIR}/rapidjson/document.h" ]]; then
        echo "[ERROR] RapidJSON headers not found." >&2
        echo "        Put RapidJSON at ${ROOT_DIR}/rapidjson-1.1.0/include," >&2
        echo "        or run: RAPIDJSON_INCLUDE_DIR=/path/to/rapidjson/include ./run.sh -c 1" >&2
        exit 1
    fi

    echo "[Step] Cleaning previous compile outputs..."
    rm -rf "${ROOT_DIR}/build" "${ROOT_DIR}/build_device"
    [[ "${VERBOSE_BUILD:-0}" != "1" ]] && \
        echo "[Step] Compiling quietly; full output -> ${RESULT_DIR}/log/build.log (VERBOSE_BUILD=1 to stream it)"

    _build_t0=${SECONDS}
    echo "[Step] Building device kernels (serial; AscendC ExternalProject is not -j safe)..."
    run_build_tool cmake -S "${ROOT_DIR}/src/device" -B "${ROOT_DIR}/build_device" \
        -DASCEND_CANN_PACKAGE_PATH="${CANN}" \
        -DSOC_VERSION="${SOC_VERSION}"
    run_build_tool cmake --build "${ROOT_DIR}/build_device"
    echo "[Step]   device kernels done ($((SECONDS - _build_t0))s)"

    PROTOBUF_CMAKE_ARGS=()
    [[ -n "${PROTOBUF_ROOT}" ]]        && PROTOBUF_CMAKE_ARGS+=(-DProtobuf_ROOT="${PROTOBUF_ROOT}")
    [[ -n "${PROTOBUF_LIBRARY}" ]]     && PROTOBUF_CMAKE_ARGS+=(-DProtobuf_LIBRARY="${PROTOBUF_LIBRARY}")
    [[ -n "${PROTOBUF_INCLUDE_DIR}" ]] && PROTOBUF_CMAKE_ARGS+=(-DProtobuf_INCLUDE_DIR="${PROTOBUF_INCLUDE_DIR}")
    [[ -n "${PROTOC}" ]]               && PROTOBUF_CMAKE_ARGS+=(-DProtobuf_PROTOC_EXECUTABLE="${PROTOC}")

    _build_t0=${SECONDS}
    echo "[Step] Building host tools..."
    run_build_tool cmake -S "${ROOT_DIR}/src" -B "${ROOT_DIR}/build" \
        -DASCEND_CANN_PACKAGE_PATH="${CANN}" \
        -DSOC_VERSION="${SOC_VERSION}" \
        -DRAPIDJSON_INCLUDE_DIR="${RAPIDJSON_INCLUDE_DIR}" \
        -DNPUR_EXTRA_RPATH_DIRS="${RUNTIME_RPATH_DIRS}" \
        "${PROTOBUF_CMAKE_ARGS[@]}"
    run_build_tool cmake --build "${ROOT_DIR}/build" -j "${BUILD_THREADS}"
    echo "[Step]   host tools done ($((SECONDS - _build_t0))s)"
else
    echo "[Step] Skipping compile due to -c 0. Reusing existing build/fr_* artifacts."
fi

# Write-side switch, so the A/B is two indexes rather than two binaries; changing it needs
# --convert-data 1. An index written this way needs a binary that knows the layout -- an older
# one reads layout code 4 as a pair-encoded posting and computes K wrong.
export NPUR_SPARSE_PACKED="${SPARSE_PACKED:-1}"

require_executable "${ROOT_DIR}/build/fr_converter"
require_executable "${ROOT_DIR}/build/fr_builder"
require_executable "${ROOT_DIR}/build/fr_search"

mkdir -p "${WORK_DIR}" "${RESULT_DIR}"

if [[ "${DO_CONVERT_QUERY}" == "1" ]]; then
    require_file "${TAG_MAP_FILE}" "tag_map.bin"
    require_file "${QUERYDATA_FILE}" "QueryData_10000.txt"
    echo "[Step] Converting QueryData to fr_search inputs..."
    python3 "${ROOT_DIR}/tools/convert_querydata.py" \
        --query "${QUERYDATA_FILE}" \
        --tag-map "${TAG_MAP_FILE}" \
        --dataset "${DATASET_FILE}" \
        --out-dir "${QUERY_OUT_DIR}"
else
    SKIPPED_STEPS+=("query conversion")
fi

QUERY_FVECS="${QUERY_OUT_DIR}/queries.fvecs"
FILTER_FILE="${QUERY_OUT_DIR}/filters.txt"
TOPK_FILE="${QUERY_OUT_DIR}/topk.txt"
require_file "${QUERY_FVECS}" "converted queries.fvecs"
require_file "${FILTER_FILE}" "converted filters.txt"
require_file "${TOPK_FILE}" "converted topk.txt"

if [[ "${DO_CONVERT_DATA}" == "1" ]]; then
    echo "[Step] Converting HYDSET2 dataset to builder input..."
    if [[ "${MULTICARD}" == "1" ]]; then
        # fr_converter's mkdir is per-directory, so the per-shard --out needs its parent to exist.
        mkdir -p "${CONVERTER_IN_DIR}"
        for ((s = 0; s < SHARD_NUM; s++)); do
            read -r off cnt <<< "$(shard_doc_range "${s}")"
            shard_in="${CONVERTER_IN_DIR}/shard${s}"
            echo "  shard ${s}: docs [${off}, $((off + cnt))) -> ${shard_in}"
            rm -rf "${shard_in}"
            run_build_tool "${ROOT_DIR}/build/fr_converter" \
                --mode hw --dataset "${DATASET_FILE}" --out "${shard_in}" \
                --doc_offset "${off}" --docs "${cnt}" \
                --doc_num_per_segment "${DOC_NUM_PER_SEGMENT}" \
                --density_threshold "${DENSITY_THRESHOLD}" --threads "${CONVERT_THREADS}"
        done
    else
        rm -rf "${CONVERTER_IN_DIR}"
        converter_args=(
            --mode hw
            --dataset "${DATASET_FILE}"
            --out "${CONVERTER_IN_DIR}"
            --docs "${DOCS}"
            --doc_num_per_segment "${DOC_NUM_PER_SEGMENT}"
            --density_threshold "${DENSITY_THRESHOLD}"
            --threads "${CONVERT_THREADS}"
        )
        run_build_tool "${ROOT_DIR}/build/fr_converter" "${converter_args[@]}"
    fi
else
    SKIPPED_STEPS+=("dataset conversion")
fi

if [[ "${DO_BUILD_INDEX}" == "1" ]]; then
    echo "[Step] Building full-recall index..."
    if [[ "${MULTICARD}" == "1" ]]; then
        mkdir -p "${INDEX_DIR}"  # per-shard --data_out (INDEX_DIR/shardN) needs its parent
        for ((s = 0; s < SHARD_NUM; s++)); do
            shard_in="${CONVERTER_IN_DIR}/shard${s}"
            shard_index="${INDEX_DIR}/shard${s}"
            echo "  building shard ${s} -> ${shard_index}"
            rm -rf "${shard_index}"
            NPUR_EXECUTOR_THREADS="${BUILD_THREADS}" NPUR_LOG_LEVEL=WARN \
                run_build_tool "${ROOT_DIR}/build/fr_builder" --data_dir "${shard_in}" --data_out "${shard_index}"
        done
        # The only per-shard figure trustworthy on a shared card: --mem-report reads device-wide HBM.
        echo "[Step] Per-shard index size:"
        du -sh "${INDEX_DIR}"/shard* 2>/dev/null | sed 's/^/  /'
        du -sh "${INDEX_DIR}" 2>/dev/null | sed 's/^/  total: /'
    else
        rm -rf "${INDEX_DIR}"
        NPUR_EXECUTOR_THREADS="${BUILD_THREADS}" NPUR_LOG_LEVEL=WARN \
            run_build_tool "${ROOT_DIR}/build/fr_builder" --data_dir "${CONVERTER_IN_DIR}" --data_out "${INDEX_DIR}"
        echo "[Step] Index size:"
        du -sh "${INDEX_DIR}" 2>/dev/null | sed 's/^/  /'
    fi
else
    SKIPPED_STEPS+=("index build")
fi

if [[ "${DO_SEARCH}" == "1" ]]; then
    shard_index_dirs=""
    tag_freq_file="${CONVERTER_IN_DIR}/tag_doc_freq.txt"
    if [[ "${EXTERNAL_SHARDS}" == "1" ]]; then
        shard_index_dirs="${SHARD_INDEX_DIRS}"
        for d in "${SHARD_INDEX_LIST[@]}"; do
            shopt -s nullglob
            m=("${d}"/*.meta)
            shopt -u nullglob
            if [[ "${#m[@]}" -eq 0 ]]; then
                echo "[ERROR] Missing shard index metadata: ${d}/*.meta" >&2
                exit 1
            fi
        done
    elif [[ "${MULTICARD}" == "1" ]]; then
        for ((s = 0; s < SHARD_NUM; s++)); do
            shard_index="${INDEX_DIR}/shard${s}"
            shopt -s nullglob
            m=("${shard_index}"/*.meta)
            shopt -u nullglob
            if [[ "${#m[@]}" -eq 0 ]]; then
                echo "[ERROR] Missing shard ${s} index metadata: ${shard_index}/*.meta" >&2
                exit 1
            fi
            shard_index_dirs+="${shard_index_dirs:+,}${shard_index}"
        done
        tag_freq_file="${CONVERTER_IN_DIR}/shard0/tag_doc_freq.txt"  # diagnostics only (shard0 slice)
    else
        shopt -s nullglob
        meta_files=("${INDEX_DIR}"/*.meta)
        shopt -u nullglob
        if [[ "${#meta_files[@]}" -eq 0 ]]; then
            echo "[ERROR] Missing new-project index metadata: ${INDEX_DIR}/*.meta" >&2
            exit 1
        fi
    fi
    if [[ ${#SKIPPED_STEPS[@]} -gt 0 ]]; then
    echo "[Step] Skipping: $(IFS=', '; echo "${SKIPPED_STEPS[*]}")"
fi
echo "[Step] Index on disk:"
    if [[ -n "${shard_index_dirs}" ]]; then
        du -sh ${shard_index_dirs//,/ } 2>/dev/null | sed 's/^/  /' || true
    else
        du -sh "${INDEX_DIR}" 2>/dev/null | sed 's/^/  /' || true
    fi
    echo "[Step] Running full-recall search..."
    mkdir -p "${RESULT_DIR}/log"
    search_log="${RESULT_DIR}/log/fr_search.log"
    search_args=(
        --query_file "${QUERY_FVECS}"
        --filter_file "${FILTER_FILE}"
        --topk_file "${TOPK_FILE}"
        --dataset_hw "${DATASET_FILE}"
        --topk "${DEFAULT_TOPK}"
        --warmup "${WARMUP}"
        --recall_queries "${RECALL_QUERIES}"
        --full_recall_text_filter_block_dim="${FULL_RECALL_TEXT_FILTER_BLOCK_DIM}"
        --full_recall_batch_search_thread_num="${BATCH_SEARCH_THREADS}"
        --effective_filter_file "${RESULT_DIR}/log/effective_filter_expressions.txt"
    )
    [[ -f "${tag_freq_file}" ]] && search_args+=(--converted_tag_freq_file "${tag_freq_file}")
    if [[ "${MULTICARD}" == "1" ]]; then
        search_args+=(--shard_index_dirs "${shard_index_dirs}" --device_ids "${DEVICE_IDS}")
    else
        search_args+=(--index_dir "${INDEX_DIR}" --device_id "${DEVICE_ID}")
    fi
    if [[ -n "${RECALL_REF_FILE}" ]]; then
        search_args+=(--recall_ref_file "${RECALL_REF_FILE}" --recall_ref_threads "${RECALL_REF_THREADS}")
    fi
    if [[ "${NUM_QUERIES}" != "0" ]]; then
        search_args+=(--num_queries "${NUM_QUERIES}")
    fi
    search_args+=(--batch_size "${BATCH_SIZE}")
    # NPU_TOPK_LOOP_COUNT: more passes -> closer to topK, so fewer over-returned candidates.
    # NPU_TOPK_THRESHOLD_RATIO: matches > topK*ratio -> use NPU topk (lower = kicks in sooner).
    [[ -n "${NPU_TOPK_LOOP_COUNT:-}" ]] && \
        search_args+=(--full_recall_npu_topk_loop_count="${NPU_TOPK_LOOP_COUNT}")
    [[ -n "${NPU_TOPK_THRESHOLD_RATIO:-}" ]] && \
        search_args+=(--full_recall_npu_topk_enters_threshold_ratio="${NPU_TOPK_THRESHOLD_RATIO}")
    # Wrapper defaults overriding the gflag defaults (1.5 / 8 / 8).
    search_args+=(--full_recall_npu_topk_finish_buffer_ratio="${NPU_TOPK_FINISH_BUFFER_RATIO}")
    search_args+=(--full_recall_aggregator_block_dim="${AGGREGATOR_BLOCK_DIM}")
    search_args+=(--full_recall_scorer_block_dim="${SCORER_BLOCK_DIM}")
    # Defaults to 0.05 whatever the index was BUILT with, and the index meta does not carry the
    # threshold, so this is the only place the two can be kept equal.
    search_args+=(--full_recall_bitlist_denseness_threshold="${DENSITY_THRESHOLD}")
    # MEASURED AND REFUTED (+41.5% on the kernel). Control arm only -- do not turn it on.
    npur_switch BITLIST_CLEAR_SPARSE NPUR_BITLIST_CLEAR_SPARSE "0"
    npur_switch BITLIST_ONE_H2D NPUR_BITLIST_ONE_H2D "0"
    # Needs BITLIST_ONE_H2D=1 (the table has to be written by index).
    npur_switch BITLIST_POSTING_MAJOR NPUR_BITLIST_POSTING_MAJOR "0"
    # Pools grow a block to the largest size EVER requested, not the current one. Set 0 for the old.
    export NPUR_POOL_HIGH_WATER="${POOL_HIGH_WATER:-1}"
    # Scatter is a silent no-op on some CANN builds, so RECALL is the result here, not the timing.
    # 1 = offsets as stored (bytes), 2 = halved to element indices.
    if [[ -n "${BITLIST_SCATTER:-}" && "${BITLIST_SCATTER}" != "0" ]]; then
        export NPUR_BITLIST_SCATTER="${BITLIST_SCATTER}"
    fi
    # Needs NPUR_OVERLAP_POSTING=1, which alone skips these queries (~98% at a non-zero threshold).
    npur_switch OVERLAP_POSTING_BITLIST NPUR_OVERLAP_POSTING_BITLIST
    # [POOLSTAT] pool ids: 0 vector-score result, 1 filter result, 2 filter stack, 3 bitlist->bitset,
    # 4 aggregator, 5 aggregator scratch, 6 filter postings. No device id on the line, so with two
    # cards each id appears twice -- sum them.
    if [[ "${POOL_STATS:-0}" != "0" ]]; then
        export NPUR_POOL_STATS=1
    fi
    # =1 is free and can stay on for a timing run. =2 also reads each converted posting's header back
    # ([BITSTAT-DEEP]) -- one D2H per posting. NEVER leave 2 on for a run whose latency you read.
    if [[ -n "${BITLIST_STATS:-}" && "${BITLIST_STATS}" != "0" ]]; then
        export NPUR_BITLIST_STATS="${BITLIST_STATS}"
        [[ -n "${BITLIST_STATS_EVERY:-}" ]] && export NPUR_BITLIST_STATS_EVERY="${BITLIST_STATS_EVERY}"
        [[ -n "${BITLIST_STATS_CALLS:-}" ]] && export NPUR_BITLIST_STATS_CALLS="${BITLIST_STATS_CALLS}"
        # Level 2 only. A run-based encoding would only pay if the mean run exceeds 1.5 units.
        [[ -n "${BITLIST_STATS_SAMPLE:-}" ]] && export NPUR_BITLIST_STATS_SAMPLE="${BITLIST_STATS_SAMPLE}"
        if [[ "${BITLIST_STATS}" == "2" ]]; then
            echo "[WARN] BITLIST_STATS=2 adds a per-posting D2H on the first calls: diagnostic only,"
            echo "[WARN] the latency of those calls is meaningless."
        fi
    fi
    # Output-identical at tileNum == 1, which CreateContextData always sets. 0 = the checked path.
    npur_switch OR_NO_TILE_CHECK NPUR_OR_NO_TILE_CHECK
    # Needs SPARSE_DIRECT=1 (it reuses the same OR-operand mask).
    npur_switch OR_SKIP_EMPTY NPUR_OR_SKIP_EMPTY "0"
    # Walks every pair a second time, so leave it off for timing runs. _EVERY = print interval.
    if [[ "${EMPTY_STATS:-0}" == "1" ]]; then
        export NPUR_EMPTY_STATS=1
        [[ -n "${EMPTY_STATS_EVERY:-}" ]] && export NPUR_EMPTY_STATS_EVERY="${EMPTY_STATS_EVERY}"
        echo "[WARN] EMPTY_STATS=1 adds a second pass over every (posting, segment) pair:"
        echo "[WARN] diagnostic only, the latency under it is not comparable."
    fi
    # MEASURED NEUTRAL at both batch sizes; kept as a control arm, off.
    npur_switch ONE_COUNT_PASS NPUR_ONE_COUNT_PASS "0"
    # Builds the weight arrays only when something reads them -- only OR_SKIP_EMPTY and EMPTY_STATS do.
    npur_switch LAZY_POSTING_WEIGHTS NPUR_LAZY_POSTING_WEIGHTS "1"
    # Defaults to SPARSE_DIRECT -- without it nothing is an OR operand and this never triggers.
    npur_switch CLASSIFY_FAST NPUR_CLASSIFY_FAST "${SPARSE_DIRECT}"
    # Defaults to SPARSE_DIRECT. 0 = the padded helper that walks the remainder with scalar GM reads.
    npur_switch OR_ALIGNED_COPY NPUR_OR_ALIGNED_COPY "${SPARSE_DIRECT}"
    # OR_ABLATE=1|2|3 leaves one step out of ApplySparse; the three are strict subsets, so each piece
    # is a difference between arms:
    #   full - 1 = the placement loop   1 - 2 = the input DataCopy   3 = the floor
    #   2 - 3 = AllocTensor, both queue round trips and FreeTensor
    # WRONG RESULTS BY DESIGN: read only TextFilter_BitmapTextFilter_Kernel under --profile 2.
    if [[ -n "${OR_ABLATE:-}" && "${OR_ABLATE}" != "0" ]]; then
        export NPUR_OR_ABLATE="${OR_ABLATE}"
        echo "[WARN] OR_ABLATE=${OR_ABLATE}: FilterOrOp is deliberately skipping work, so the"
        echo "[WARN] filter results and the recall under it are WRONG BY DESIGN. Only the"
        echo "[WARN] per-stage kernel timing is meaningful."
    fi
    # OR nodes take their sparse postings encoded instead of converted; AND and NOT keep converting.
    # Automatically refused when GetMaxPostingLength exceeds the 16KB tile buffer (den >= ~0.021).
    npur_switch SPARSE_DIRECT NPUR_SPARSE_DIRECT
    # Cheap (it rides on CheckPostExpr's own walk) but it prints, so keep it off for timing runs.
    if [[ "${EXPR_STATS:-0}" == "1" ]]; then
        export NPUR_EXPR_STATS=1
        [[ -n "${EXPR_STATS_EVERY:-}" ]] && export NPUR_EXPR_STATS_EVERY="${EXPR_STATS_EVERY}"
    fi
    # WRONG RESULTS BY DESIGN: read only the per-stage kernel timing, never recall.
    if [[ -n "${BITLIST_ABLATE:-}" && "${BITLIST_ABLATE}" != "0" ]]; then
        export NPUR_BITLIST_ABLATE="${BITLIST_ABLATE}"
        echo "[WARN] BITLIST_ABLATE=${BITLIST_ABLATE}: the bitlist->bitset kernel is deliberately"
        echo "       incomplete. Results are WRONG; only the kernel timing is meaningful."
    fi
    if [[ "${LATENCY_DUMP:-0}" == "1" ]]; then
        search_args+=(--latency_dump "${RESULT_DIR}/log/per_query_latency.tsv")
    fi
    # Default ON; =0 restores the spawn path for A/B. Spawn jitter was the dominant p99 noise source.
    if [[ "${SHARD_WORKER_POOL:-1}" != "0" ]]; then
        search_args+=(--shard_worker_pool)
    fi
    # Comes off chunk LATENCY, not throughput. Groups only; needs group size > 1.
    if [[ "${GROUP_EXTRACT_PARALLEL:-0}" == "1" ]]; then
        search_args+=(--group_extract_parallel)
    fi
    # Needs PIPELINE=1 and the worker pool.
    if [[ "${GROUP_PREFETCH:-0}" == "1" ]]; then
        search_args+=(--group_prefetch)
    fi
    if [[ "${PARALLEL_LOAD:-0}" == "1" ]]; then
        search_args+=(--parallel_load)
    fi
    # One core list per DEVICE_IDS entry, ';'-separated. The cards do NOT share a NUMA node, so a
    # process-wide taskset is remote to at least one. Read the lists off `npu-smi info -t topo`
    # (CPU Affinity column); machine-specific, so there is no default.
    if [[ -n "${CARD_CPUS:-}" ]]; then
        search_args+=(--card_cpus "${CARD_CPUS}")
    fi
    # Hides the host merge behind the straggler. Byte-identical results (deterministic tie-break).
    if [[ "${STREAM_MERGE:-0}" == "1" ]]; then
        search_args+=(--stream_merge)
    fi
    if [[ "${ROUND_ROBIN}" == "1" ]]; then
        search_args+=(--round_robin)
    fi
    if [[ "${SHARD_GROUP_SIZE}" -gt 0 ]]; then
        search_args+=(--shard_group_size "${SHARD_GROUP_SIZE}")
    fi
    if [[ "${STATIC_ASSIGN}" == "1" ]]; then
        search_args+=(--static_assign)
    fi
    if [[ -n "${SLOW_CARDS}" ]]; then
        search_args+=(--slow_cards "${SLOW_CARDS}" --slow_factor "${SLOW_FACTOR}"
                      --slow_period_ms "${SLOW_PERIOD_MS}" --slow_duty "${SLOW_DUTY}")
    fi
    if [[ "${TARGET_QPS}" != "0" ]]; then
        search_args+=(--target_qps "${TARGET_QPS}")
    fi
    if [[ "${PIPELINE}" == "1" ]]; then
        search_args+=(--pipeline)
    fi
    # Per-query per-shard TSV: slowest shard vs host merge. Needs --batch_size 1 to be per-query.
    if [[ "${SHARD_LATENCY_DUMP:-0}" == "1" ]]; then
        search_args+=(--shard_latency_dump "${RESULT_DIR}/log/per_shard_latency.tsv")
    fi
    if [[ "${MEM_REPORT}" == "1" ]]; then
        search_args+=(--mem_report)
    fi

    # The loop body keeps its original indentation ON PURPOSE: re-indenting would break the column-0
    # <<'PY' heredoc terminators below. fr_search reads NPUR_POOL_POSTINGS once, so export it.
    export NPUR_POOL_POSTINGS="${POOL_POSTINGS}"
    npur_switch PACKED_SORT NPUR_PACKED_SORT "0"
    # LSD radix instead of std::sort; implies the packed path.
    npur_switch RADIX_SORT NPUR_RADIX_SORT "0"
    # Each shard aggregates only its top ceil(topK*ratio); the merge still returns the full topK.
    # Costs a little recall (the global topK splits ~topK/N per shard). Default 1 = full.
    [[ -n "${SHARD_TOPK_RATIO:-}" ]] && export NPUR_SHARD_TOPK_RATIO="${SHARD_TOPK_RATIO}"
    # Phases across queries instead of query at a time, so the per-query syncs become a constant.
    npur_switch BATCH_AGGREGATE NPUR_BATCH_AGGREGATE "0"
    # Only does anything with BATCH_AGGREGATE=1 and BATCH_SIZE >= 2.
    npur_switch TOPK_CONCURRENT NPUR_TOPK_CONCURRENT "0"
    npur_switch POOL_SMALL_H2D NPUR_POOL_SMALL_H2D "0"
    # Needs NPUR_OVERLAP_POSTING and BATCH_FILTER; only on BatchSearchDevice (groups, or PIPELINE=1).
    npur_switch EARLY_FILTER_PREP NPUR_EARLY_FILTER_PREP "0"
    # The filter kernels' time then shows up inside AggrAndTopK_Aggregator_NPU. Needs BATCH_FILTER=1
    # and BATCH_AGGREGATE=1; only on BatchSearchDevice (groups, or PIPELINE=1).
    npur_switch SHARE_FILTER_STREAM NPUR_SHARE_FILTER_STREAM "0"
    # Changes the TOPK kernel's arguments, so the first run after pulling it needs --compile 1.
    # Only with BATCH_AGGREGATE=1.
    npur_switch TOPK_COUNTS_IN_PLACE NPUR_TOPK_COUNTS_IN_PLACE "0"
    # Needs TOPK_COUNTS_IN_PLACE=1 and a single aggregator lane (AGG_CONCURRENT unset or 1);
    # otherwise it QUIETLY STAYS OFF.
    npur_switch FUSE_AGG_TOPK NPUR_FUSE_AGG_TOPK "0"
    # Needs OVERLAP_POSTING and BITLIST_ONE_H2D; only on BatchSearchDevice. Does nothing on an index
    # where no query converts anything.
    npur_switch DEFER_CONV_SYNC NPUR_DEFER_CONV_SYNC "0"
    # Each Aggregator takes 16 of the 40 vector cores, so 2 is the natural setting. IGNORED under
    # SHARE_FILTER_STREAM, whose one stream is what orders the aggregator behind the filter.
    [[ -n "${AGG_CONCURRENT:-}" ]] && export NPUR_AGG_CONCURRENT="${AGG_CONCURRENT}"
    # WITHOUT IT every GetStream() returns nullptr and every kernel runs on the default stream, so
    # TOPK_CONCURRENT, AGG_CONCURRENT and DEFER_CONV_SYNC do nothing. It is an arm of its own in any A/B.
    npur_switch STREAM_POOL NPUR_STREAM_POOL "0"
    # Implies the early filter prepare. Needs STREAM_POOL=1 for a second stream (without it they queue
    # behind the scorer -- correct, but not overlapped), plus OVERLAP_POSTING and BATCH_FILTER.
    npur_switch OVERLAP_FILTER NPUR_OVERLAP_FILTER_KERNEL "0"
    # Output-identical (keys built index-descending, so ties come out as before); needs RADIX_SORT=1.
    npur_switch EXTRACT_FAST NPUR_EXTRACT_FAST "0"
    # Only takes effect where BatchCompute is handed prepared expressions, i.e. NPUR_OVERLAP_POSTEXPR
    # != 0 (the default).
    npur_switch BATCH_FILTER NPUR_BATCH_FILTER "0"
    npur_switch SCORER_TRIM_WRITE NPUR_SCORER_TRIM_WRITE "0"
    npur_switch TOPK_LOOP_STATS NPUR_TOPK_LOOP_STATS "0"
    # Point RECALL_REF_FILE at a SEPARATE file or the two modes overwrite each other's cache.
    if [[ "${RECALL_FP32:-0}" == "1" ]]; then
        search_args+=(--recall_fp32)
    fi
    npu_hbm_report before
    npu_hbm_sample_start
    npu_util_start
    npu_load_start
    run_latency_lines=()
    for ((run = 1; run <= REPEAT; run++)); do
    run_suffix=""
    if [[ "${REPEAT}" -gt 1 ]]; then
        run_suffix=".run${run}"
        echo "[Step] Search run ${run}/${REPEAT}..."
    fi
    search_log="${RESULT_DIR}/log/fr_search${run_suffix}.log"
    set +e
    if [[ "${PROFILE_MODE}" != "0" ]]; then
        # awk stream-aggregates the [Perf] lines so the 10k x N-stage raw output never hits the log.
        perf_file="${RESULT_DIR}/log/fr_search_perf${run_suffix}.txt"
        : > "${perf_file}"
        # RecordGuard prints the outer "BatchSearch" line LAST per query (RAII), so every [Perf] line is
        # tagged with the current q and q is bumped after each BatchSearch. q 0..4 are the warmup queries.
        perf_raw="${RESULT_DIR}/log/fr_search_perf_raw${run_suffix}.tsv"
        : > "${perf_raw}"
        # quiet=1 (--profile 2) folds the per-query recall lines into a count -- the count is still printed.
        NPUR_PERF=1 NPUR_LOG_LEVEL="${NPUR_LOG_LEVEL:-ERROR}" "${ROOT_DIR}/build/fr_search" "${search_args[@]}" 2>&1 \
            | awk -v pf="${perf_file}" -v raw="${perf_raw}" -v quiet="$([[ "${PROFILE_MODE}" == "2" ]] && echo 1 || echo 0)" '
                /^\[Perf\]/ {
                    us=$(NF-1); tag=$2; for (i=3;i<=NF-2;i++) tag=tag " " $i;
                    s[tag]+=us; c[tag]++;
                    if (raw != "") printf "%d\t%s\t%d\n", q, tag, us > raw;
                    if (tag == "BatchSearch") q++;
                    next
                }
                quiet == 1 && /^\[Result\] q[0-9]+ recall=/ { mismatch++; next }
                { print }
                END {
                    if (quiet == 1 && mismatch > 0)
                        printf "[Result] %d per-query recall lines suppressed (--profile 2); see the CPU-recall summary\n", mismatch
                    for (k in s) printf "%.1f\t%.0f\t%d\t%s\n", s[k]/c[k], s[k], c[k], k > pf
                }' \
            | tee "${search_log}"
        rc=${PIPESTATUS[0]}
        if [[ -s "${perf_file}" ]]; then
            {
                printf '\n==== per-stage timing (avg per query, sorted by avg desc) ====\n'
                printf '%12s %14s %10s  %s\n' "avg_us" "total_us" "count" "stage"
                sort -t"$(printf '\t')" -k1 -rn "${perf_file}" \
                    | awk -F'\t' '{printf "%12.1f %14.0f %10d  %s\n", $1,$2,$3,$4}'
            } | tee -a "${search_log}"
        fi
        if [[ -s "${perf_raw}" ]]; then
            # Percentiles need no query grouping, so this is correct in multi-card where q is per-shard.
            PERF_RAW="${perf_raw}" python3 - <<'PY' | tee -a "${search_log}"
import os, math
from collections import defaultdict
vals = defaultdict(list)
with open(os.environ["PERF_RAW"]) as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 3:
            continue
        vals[parts[1]].append(int(parts[2]))
def pct(a, p):
    if not a:
        return 0
    k = max(0, min(len(a) - 1, math.ceil(p / 100.0 * len(a)) - 1))
    return a[k]
rows = []
for stage, a in vals.items():
    a.sort()
    rows.append((pct(a, 99), pct(a, 50), pct(a, 90), a[-1], len(a), stage))
rows.sort(reverse=True)  # by p99 desc
print("\n==== per-stage timing percentiles (us, over all invocations, sorted by p99) ====")
print(f"{'p50':>10} {'p90':>10} {'p99':>10} {'max':>10} {'count':>9}  stage")
for p99, p50, p90, mx, cnt, stage in rows:
    print(f"{p50:>10d} {p90:>10d} {p99:>10d} {mx:>10d} {cnt:>9d}  {stage}")
PY
        fi
        # Needs the q tagging above, which assumes ONE BatchSearch in flight. Multi-card breaks it and THE
        # OUTPUT LOOKS PLAUSIBLE ANYWAY -- single-card only. The percentile table above ignores q entirely.
        if [[ "${PERF_DUMP:-0}" == "1" && -s "${perf_raw}" && "${SHARD_NUM}" -gt 1 ]]; then
            echo "[Info] PERF_DUMP tail attribution skipped: it needs single-card (${SHARD_NUM} shards here)." >&2
            echo "       The q tagging assumes serial BatchSearch; parallel shards scramble the buckets." >&2
        elif [[ "${PERF_DUMP:-0}" == "1" && -s "${perf_raw}" ]]; then
            # Per-stage avg over all timed queries vs the slowest 1%, ranked by their BatchSearch total.
            PERF_RAW="${perf_raw}" python3 - <<'PY' | tee -a "${search_log}"
import os, collections
raw = os.environ["PERF_RAW"]
q = collections.defaultdict(dict)          # q -> {stage: us}
with open(raw) as f:
    for line in f:
        qi, stage, us = line.rstrip("\n").split("\t")
        q[int(qi)][stage] = q[int(qi)].get(stage, 0) + int(us)
# Drop the 5 warmup queries (indices 0..4); keep timed queries only.
timed = [q[i] for i in sorted(q) if i >= 5 and "BatchSearch" in q[i]]
if not timed:
    raise SystemExit(0)
timed.sort(key=lambda d: d["BatchSearch"], reverse=True)
n = len(timed)
k = max(1, n // 100)                        # slowest 1%
slow = timed[:k]
stages = sorted({s for d in timed for s in d}, key=lambda s: -sum(d.get(s, 0) for d in timed))
def avg(rows, s): return sum(d.get(s, 0) for d in rows) / len(rows)
print(f"\n==== p99 TAIL per-stage: slowest {k}/{n} queries (by BatchSearch total) vs all ====")
print(f"{'avg_all_us':>12} {'avg_slow1%_us':>14} {'x':>7}  stage")
for s in stages:
    a, b = avg(timed, s), avg(slow, s)
    r = (b / a) if a > 0 else 0
    print(f"{a:12.1f} {b:14.1f} {r:7.1f}  {s}")
PY
        fi
    else
        NPUR_LOG_LEVEL="${NPUR_LOG_LEVEL:-ERROR}" "${ROOT_DIR}/build/fr_search" "${search_args[@]}" 2>&1 | tee "${search_log}"
        rc=${PIPESTATUS[0]}
    fi
    set -e
    if [[ "${rc}" -ne 0 ]]; then
        echo "[ERROR] fr_search failed (run ${run}/${REPEAT}) with exit code ${rc}. See ${search_log}" >&2
        exit "${rc}"
    fi
    lat_line="$(grep -m1 '^\[Result\] latency ms:' "${search_log}" 2>/dev/null || true)"
    [[ -n "${lat_line}" ]] && run_latency_lines+=("${lat_line}")
    echo "[Done] Search log: ${search_log}"
    done
    npu_load_stop
    npu_util_stop
    npu_hbm_sample_stop
    npu_hbm_report after

    # A single 10k-query run's p99 is not reproducible; each run's value is shown so the spread shows.
    if [[ "${REPEAT}" -gt 1 && "${#run_latency_lines[@]}" -gt 0 ]]; then
        median_src="${RESULT_DIR}/log/fr_search_latency_runs.txt"
        printf '%s\n' "${run_latency_lines[@]}" > "${median_src}"
        MEDIAN_SRC="${median_src}" BS="${BATCH_SIZE}" python3 - <<'PY' | tee "${RESULT_DIR}/log/fr_search_median.log"
import os, re, statistics
runs = []
for line in open(os.environ["MEDIAN_SRC"]):
    d = {k: float(v) for k, v in re.findall(r'([a-z0-9]+)=([0-9.]+)', line)}
    if d:
        runs.append(d)
if not runs:
    raise SystemExit(0)
order = ["avg", "p50", "p75", "p90", "p95", "p99", "max"]
keys = [k for k in order if all(k in r for r in runs)]
n = len(runs)
print(f"\n==== end-to-end latency ms across {n} runs (median per metric) ====")
print(f"{'metric':<8}" + "".join(f"{'run'+str(i+1):>10}" for i in range(n)) + f"{'median':>12}")
for k in keys:
    vals = [r[k] for r in runs]
    print(f"{k:<8}" + "".join(f"{v:>10.4f}" for v in vals) + f"{statistics.median(vals):>12.4f}")
# qps as a table row too (queries / wall-clock, printed per run by the harness),
# in .1f -- .4f would suggest precision a wall-clock measurement does not have.
if all("qps" in r for r in runs):
    vals = [r["qps"] for r in runs]
    print(f"{'qps':<8}" + "".join(f"{v:>10.1f}" for v in vals) + f"{statistics.median(vals):>12.1f}")
    print(f"\n[Result] throughput (median): {statistics.median(vals):.1f} qps")
PY
    fi
else
    echo "[Step] Skipping search."
fi

# Reached only on success (a failed fr_search exits above). Never fail the run on a notify error.
if [[ "${DO_SEARCH}" == "1" && "${NOTIFY}" == "1" ]]; then
    if [[ -z "${NOTIFY_API_KEY}" ]]; then
        echo "[Notify] NOTIFY=1 but NOTIFY_API_KEY is empty; skipping email." \
             "Put NOTIFY_API_KEY in ${ROOT_DIR}/.notify.env" >&2
    elif ! command -v curl >/dev/null 2>&1; then
        echo "[Notify] curl not found; skipping email." >&2
    else
        # Prefer the cross-run median: the last run's p99 alone would be actively misleading here.
        notify_median_log="${RESULT_DIR}/log/fr_search_median.log"
        notify_lat_src="single run"
        notify_avg="?"
        notify_p99="?"
        if [[ -s "${notify_median_log}" ]]; then
            notify_lat_src="median of ${REPEAT} runs"
            notify_avg="$(awk '$1=="avg"{print $NF}' "${notify_median_log}" 2>/dev/null || true)"
            notify_p99="$(awk '$1=="p99"{print $NF}' "${notify_median_log}" 2>/dev/null || true)"
        elif [[ -n "${lat_line:-}" ]]; then
            notify_avg="$(sed -n 's/.*avg=\([0-9.]*\).*/\1/p' <<<"${lat_line}" || true)"
            notify_p99="$(sed -n 's/.*p99=\([0-9.]*\).*/\1/p' <<<"${lat_line}" || true)"
        fi
        # The harness prints this only when verification ran, so its ABSENCE is itself a result -- report
        # it loudly, or a quiet email reads as a pass when recall was never checked.
        notify_recall="$(grep -m1 'CPU-recall over' "${search_log}" 2>/dev/null \
                         | sed -n 's/.*avg=\(.*\)/\1/p' || true)"
        [[ -z "${notify_recall}" ]] && notify_recall="NOT-VERIFIED"
        # The first 10 are printed as a sanity check even when they pass, so only recall<1 counts as bad.
        notify_bad="$(grep '^\[Result\] q[0-9]* recall=' "${search_log}" 2>/dev/null \
                      | grep -v 'recall=1\.0000' || true)"
        notify_bad_n=0
        [[ -n "${notify_bad}" ]] && notify_bad_n="$(wc -l <<<"${notify_bad}" | tr -d ' ')"
        [[ "${NUM_QUERIES}" == "0" ]] && notify_nq="all" || notify_nq="${NUM_QUERIES}"
        if [[ "${MULTICARD}" == "1" ]]; then
            notify_devices="${DEVICE_IDS} (${SHARD_NUM} shards)"
        else
            notify_devices="${DEVICE_ID} (single card)"
        fi
        # CONFIG_TAG carries only segment size and density, so without this every mail from an A/B sweep
        # of any other knob has an identical subject.
        notify_diff=""
        [[ "${NPUR_PARALLEL_SCORE_FILTER:-0}" != "0" ]] && notify_diff+=" parallel_sf=1"
        [[ "${NPUR_OVERLAP_POSTEXPR:-1}" != "1" ]] && notify_diff+=" overlap_postexpr=${NPUR_OVERLAP_POSTEXPR}"
        [[ "${NPUR_OVERLAP_POSTING:-0}" != "0" ]] && notify_diff+=" overlap_posting=1"
        [[ "${SHARD_WORKER_POOL:-1}" != "1" ]] && notify_diff+=" worker_pool=0"
        [[ "${STREAM_MERGE:-0}" != "0" ]] && notify_diff+=" stream_merge=1"
        [[ "${PACKED_SORT:-0}" != "0" ]] && notify_diff+=" packed_sort=1"
        [[ "${RADIX_SORT:-0}" != "0" ]] && notify_diff+=" radix_sort=1"
        [[ -n "${SHARD_TOPK_RATIO:-}" && "${SHARD_TOPK_RATIO}" != "1" && "${SHARD_TOPK_RATIO}" != "1.0" ]] && notify_diff+=" shard_topk=${SHARD_TOPK_RATIO}"
        [[ "${POOL_POSTINGS}" != "1" ]] && notify_diff+=" postings=0"
        [[ "${SCORER_BLOCK_DIM}" != "20" ]] && notify_diff+=" scorer=${SCORER_BLOCK_DIM}"
        [[ "${FULL_RECALL_TEXT_FILTER_BLOCK_DIM}" != "40" ]] && notify_diff+=" filter=${FULL_RECALL_TEXT_FILTER_BLOCK_DIM}"
        [[ "${NPU_TOPK_FINISH_BUFFER_RATIO}" != "1.2" ]] && notify_diff+=" fbr=${NPU_TOPK_FINISH_BUFFER_RATIO}"
        [[ -n "${notify_diff}" ]] && notify_diff=" [${notify_diff# }]"
        notify_subject="[FullRecall] ${CONFIG_TAG} ${SHARD_NUM}c p99=${notify_p99} avg=${notify_avg} recall=${notify_recall}${notify_diff}"
        notify_body="$(
            printf 'p99:      %s ms   (%s)\n' "${notify_p99}" "${notify_lat_src}"
            printf 'avg:      %s ms\n' "${notify_avg}"
            printf 'recall:   %s' "${notify_recall}"
            [[ "${notify_bad_n}" != "0" ]] && printf '   (%s queries below 1.0)' "${notify_bad_n}"
            printf '\n\n'
            printf 'config:   %s\n' "${CONFIG_TAG}"
            printf 'devices:  %s\n' "${notify_devices}"
            printf 'cores:    scorer %s (Cube) + filter %s (Vector)\n' \
                "${SCORER_BLOCK_DIM}" "${FULL_RECALL_TEXT_FILTER_BLOCK_DIM}"
            printf 'knobs:    fbr=%s topk_loop=%s worker_pool=%s parallel_sf=%s overlap_postexpr=%s postings=%s\n' \
                "${NPU_TOPK_FINISH_BUFFER_RATIO}" "${NPU_TOPK_LOOP_COUNT:-gflag-default}" \
                "${SHARD_WORKER_POOL:-1}" "${NPUR_PARALLEL_SCORE_FILTER:-0}" \
                "${NPUR_OVERLAP_POSTEXPR:-0}" "${POOL_POSTINGS}"
            printf 'queries:  %s   repeat=%s\n' "${notify_nq}" "${REPEAT}"
            if [[ -s "${notify_median_log}" ]]; then
                cat "${notify_median_log}"
            elif [[ -n "${lat_line:-}" ]]; then
                printf '\n%s\n' "${lat_line}"
            fi
            if [[ -n "${notify_bad}" ]]; then
                if [[ "${notify_bad_n}" -gt 5 ]]; then
                    printf '\nrecall mismatches (first 5 of %s):\n' "${notify_bad_n}"
                else
                    printf '\nrecall mismatches:\n'
                fi
                head -5 <<<"${notify_bad}" | sed 's/^/  /'
            fi
            if [[ "${PROFILE_MODE}" != "0" && -s "${perf_file:-}" ]]; then
                printf '\ntop stages (avg_us):\n'
                sort -t"$(printf '\t')" -k1 -rn "${perf_file}" | head -6 \
                    | awk -F'\t' '{printf "  %10.1f  %s\n", $1, $4}'
            fi
            printf '\nlog:      %s\n' "${search_log}"
        )"
        notify_payload="$(
            NS="${notify_subject}" NB="${notify_body}" python3 -c \
                'import json,os;print(json.dumps({"subject":os.environ["NS"],"text":os.environ["NB"]}))'
        )" || notify_payload=""
        if [[ -n "${notify_payload}" ]]; then
            if curl -sS -m 20 -X POST "${NOTIFY_URL}" \
                    -H "Content-Type: application/json" \
                    -H "x-api-key: ${NOTIFY_API_KEY}" \
                    -d "${notify_payload}" >/dev/null 2>&1; then
                echo "[Notify] Email sent: ${notify_subject}"
            else
                echo "[Notify] Email send failed (run itself succeeded)." >&2
            fi
        fi
    fi
fi

echo "[Done] FullRecall run completed."
