#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE:-$0}")" && pwd)"

# Local, gitignored secrets/config (e.g. NOTIFY_API_KEY). Never committed.
if [[ -f "${ROOT_DIR}/.notify.env" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/.notify.env"
fi

# SoC: honor an explicit SOC_VERSION override; otherwise auto-detect from
# npu-smi (e.g. 910B3 on one server, 910B4 on another) so the same script runs
# on both environments. Falls back to Ascend910B4 if detection is unavailable.
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

# Index-encoding knobs. Defined here (before WORK_DIR) so the default work/result
# dirs can be tagged with them: each (segment, density) combo gets its own index
# dir and never clobbers another. Explicit WORK_DIR/RESULT_DIR still override.
# 262144: halves the segment count vs 131072, which halves the per-query posting
# pointer table that PostingBitListToSet copies H2D. That stage drops from ~180us
# to ~110us avg and, more importantly, its p99 from ~1000us to ~590us -- the p99
# driver. See the block below for how this interacts with the filter core count.
DOC_NUM_PER_SEGMENT="${DOC_NUM_PER_SEGMENT:-262144}"
DENSITY_THRESHOLD="${DENSITY_THRESHOLD:-0.05}"
CONFIG_TAG="seg${DOC_NUM_PER_SEGMENT}_den${DENSITY_THRESHOLD}"
RUNS_DIR="${RUNS_DIR:-${ROOT_DIR}/runs}"

WORK_DIR="${WORK_DIR:-${RUNS_DIR}/${CONFIG_TAG}/work}"
QUERY_OUT_DIR="${QUERY_OUT_DIR:-${WORK_DIR}/queries}"
CONVERTER_IN_DIR="${CONVERTER_IN_DIR:-${WORK_DIR}/builder_input}"
INDEX_DIR="${INDEX_DIR:-${WORK_DIR}/index}"
RESULT_DIR="${RESULT_DIR:-${RUNS_DIR}/${CONFIG_TAG}/result}"

GFLAGS_LIBRARY_DIR="${GFLAGS_LIBRARY_DIR:-}"
PROTOBUF_LIBRARY_DIR="${PROTOBUF_LIBRARY_DIR:-}"
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
# Scorer/filter AI-core counts. The two kernels run on separate hardware: the MMad
# scorer on Cube (20 on 910B3), the bitmap filter on Vector (40).
#
# Filter = 40, i.e. the Vector core count. The filter's work is quantized by segment
# count -- wall time is ceil(segments/cores) segments -- so the useful ceiling is
# really min(40, segments). At DOC_NUM_PER_SEGMENT=131072 (~39 segments for 5M
# docs/shard) the sweep showed exactly that shape: 16 cores -> 377us kernel, 20 ->
# 291us, 32 -> 310us (WORSE -- still 2 segments/core, just more launch overhead),
# 40 -> 233us (1 segment/core), 48 -> 233us (flat, segments used up). At the current
# 262144 there are only ~20 segments, so 20 and 40 cores measure identically (265 vs
# 268us) -- the extra 20 cores simply idle. 40 is kept because it degrades safely:
# idle cores cost nothing, and it stays optimal if the segment count ever grows.
# Note the kernel is dominated by ~200us of fixed overhead (fit: 201us + 2.06us/KB),
# so doubling the per-core bytes only costs ~14% -- do not expect core count or
# segment size to move this stage much.
#
# Scorer = 20, i.e. the Cube core count. Sweep of the kernel avg: 884us at 8, 820
# at 12, 791 at 16, 782 at 20 (floor), 864 at 24 -- past 20 it oversubscribes Cube
# and regresses. The returns are poor because the scorer is HBM-bandwidth bound
# rather than compute bound (5x the cores buys only 1.67x), but 20 is still the
# floor and nothing competes for Cube: the filter runs on Vector, and with the
# score||filter overlap off by default the two never even run at the same time.
FULL_RECALL_TEXT_FILTER_BLOCK_DIM="${FULL_RECALL_TEXT_FILTER_BLOCK_DIM:-40}"
SCORER_BLOCK_DIM="${SCORER_BLOCK_DIM:-20}"
# NPU topk early-quit tolerance: the kernel stops narrowing once it is down to
# topK*ratio candidates and returns them all, letting the host trim to topK. It
# trades NPU iterations against host sort work, and the two nearly cancel:
#
#   ratio   TopK_NPU   std_sort_sort   sum     BatchSearch
#   <=1.0     204us        297us       501us     2461us   (early-quit disabled)
#   1.2       181          315         496       2414     <- knee
#   1.5       178          329         507       2450
#   2.0       174          345         519       2501
#
# 1.2 wins by ~36us on BatchSearch (-1.5%); recall is 100% at every value. Values
# <=1.0 are all equivalent to disabling early-quit: the check sits in the branch
# where candidates still exceed topK, so `accepted + candidates <= topK*ratio` is
# unsatisfiable and the kernel always runs the full loop_count. See kernel_topk.h.
NPU_TOPK_FINISH_BUFFER_RATIO="${NPU_TOPK_FINISH_BUFFER_RATIO:-1.2}"
NUM_QUERIES="${NUM_QUERIES:-0}"
# Queries per FullRecallSearcher batch. Default 1 = one query per BatchSearch, the
# latency-oriented path. With >1 the harness times the whole batch and divides the
# wall time by the batch size, so the reported "latency ms avg" becomes ms/query --
# i.e. inverse throughput. Two caveats at BATCH_SIZE>1: p99 degenerates (every query
# in a batch shares the batch-average value, so it is no longer a tail), and a
# query's true latency is avg*BATCH_SIZE (the whole batch's wall). Use =1 for the p99
# target, >1 to measure throughput headroom.
BATCH_SIZE="${BATCH_SIZE:-1}"
WARMUP="${WARMUP:-5}"
# ROUND_ROBIN=1: round-robin BASELINE (vs the sharded scheme). Every card loads the FULL
# corpus and queries are work-stolen onto idle cards, each searched entirely on one card
# (no cross-card merge). Pass ONE full-index dir in SHARD_INDEX_DIRS; it is replicated
# across all DEVICE_IDS below. Reports throughput (queries/wall) + single-card latency.
ROUND_ROBIN="${ROUND_ROBIN:-0}"
RECALL_QUERIES="${RECALL_QUERIES:-20}"
DEFAULT_TOPK="${DEFAULT_TOPK:-100}"
DEVICE_ID="${DEVICE_ID:-0}"
# Multi-card: comma-separated NPU device ids (e.g. "0,1"). When set, the corpus
# is split into one disjoint doc shard per card, each converted+built into its
# own index, then searched multi-card with a host-side global top-K merge. Empty
# = single-card on DEVICE_ID (unchanged).
DEVICE_IDS="${DEVICE_IDS:-}"
# Reuse pre-built shard indexes: comma-separated index dirs (one per DEVICE_IDS
# entry). When set, convert+build are skipped and these are searched directly.
SHARD_INDEX_DIRS="${SHARD_INDEX_DIRS:-}"
# Optional: cache the CPU brute-force recall ground truth (fr_search
# --recall_ref_file), computed once for all queries with RECALL_REF_THREADS cores.
RECALL_REF_FILE="${RECALL_REF_FILE:-}"
RECALL_REF_THREADS="${RECALL_REF_THREADS:-0}"
BUILD_THREADS="${BUILD_THREADS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 32)}"
CONVERT_THREADS="${CONVERT_THREADS:-${BUILD_THREADS}}"

# Derive the shard layout. MULTICARD=1 activates the N-shard search path.
# EXTERNAL_SHARDS=1 (SHARD_INDEX_DIRS set) reuses pre-built shard indexes and
# skips convert+build; otherwise the corpus is split into DOCS/SHARD_NUM slices
# and each is converted+built.
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
# Round-robin baseline: every card holds the WHOLE corpus, so replicate the single
# full-index dir across all DEVICE_IDS (the user passes SHARD_INDEX_DIRS=<full> once).
if [[ "${ROUND_ROBIN}" == "1" && "${EXTERNAL_SHARDS}" == "1" && "${#SHARD_INDEX_LIST[@]}" -eq 1 && "${SHARD_NUM}" -gt 1 ]]; then
    _full="${SHARD_INDEX_LIST[0]}"
    SHARD_INDEX_LIST=()
    for ((_i = 0; _i < SHARD_NUM; _i++)); do SHARD_INDEX_LIST+=("${_full}"); done
    SHARD_INDEX_DIRS="$(IFS=,; echo "${SHARD_INDEX_LIST[*]}")"
fi
MULTICARD=0
{ [[ "${SHARD_NUM}" -gt 0 ]] || [[ "${EXTERNAL_SHARDS}" == "1" ]]; } && MULTICARD=1

# Per-shard doc range: prints "<offset> <count>" for shard $1 over DOCS docs
# split into SHARD_NUM contiguous slices (the last shard absorbs the remainder).
shard_doc_range() {
    local i="$1"
    local base=$(( DOCS / SHARD_NUM ))
    local rem=$(( DOCS % SHARD_NUM ))
    local off=$(( i * base ))
    local cnt="${base}"
    [[ "${i}" -eq $(( SHARD_NUM - 1 )) ]] && cnt=$(( base + rem ))
    echo "${off} ${cnt}"
}

# Email notification on successful search. Opt-in: NOTIFY=1. The API key must be
# provided via NOTIFY_API_KEY (put it in the gitignored .notify.env). If NOTIFY=1
# but no key is set, the run still succeeds and just skips the email.
NOTIFY="${NOTIFY:-0}"
NOTIFY_URL="${NOTIFY_URL:-https://mail.xihe.me/api/send/notification}"
NOTIFY_API_KEY="${NOTIFY_API_KEY:-}"

DO_COMPILE=1
DO_CONVERT_QUERY=1
DO_CONVERT_DATA=1
DO_BUILD_INDEX=1
DO_SEARCH=1
PROFILE_MODE=0
# Repeat the search stage N times and report the cross-run median of the
# end-to-end latency stats. The p99 tail of a single 10k-query run is not
# reproducible run-to-run; the median de-noises it. Env-overridable, CLI wins.
REPEAT="${REPEAT:-1}"
# A/B toggle for the devicePostings pooling optimization (engine env
# NPUR_POOL_POSTINGS): 1=pooled (default), 0=raw per-query aclrtMalloc/aclrtFree
# baseline. Lets the SAME binary profile both paths -- pair with --repeat for a
# de-noised A/B. Env-overridable, CLI wins.
POOL_POSTINGS="${NPUR_POOL_POSTINGS:-1}"
# Print a detailed memory report after the search: host peak RSS + per-device NPU
# HBM used/total ([MEM] lines from fr_search --mem_report). Env-overridable, CLI wins.
MEM_REPORT="${MEM_REPORT:-0}"
# Background-sample per-device NPU AI Core usage (%) with npu-smi while the search
# runs, then report avg/max/min per device. Tooling-only (no engine rebuild).
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
  --profile <0|1|2>          1=per-stage NPU timing (NPUR_PERF): aggregate [PERF] lines into an
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
  EXTRA_LD_LIBRARY_PATH              Extra runtime library dirs, colon-separated. Current: ${EXTRA_LD_LIBRARY_PATH:-empty}
  WORK_DIR                           Intermediate output root. Current: ${WORK_DIR}
  DOCS                               Docs to index, 0=all. Current: ${DOCS}
  DOC_NUM_PER_SEGMENT                Builder segment size. Current: ${DOC_NUM_PER_SEGMENT}
  DENSITY_THRESHOLD                  Converter bitlist/bitset cutoff (density<thr -> bitlist). Lower = more bitset, less runtime conversion, bigger index. Current: ${DENSITY_THRESHOLD}
  FULL_RECALL_TEXT_FILTER_BLOCK_DIM  Runtime filter AI-core block dim. Current: ${FULL_RECALL_TEXT_FILTER_BLOCK_DIM}
  SCORER_BLOCK_DIM                   Runtime scorer (MMad) AI-core block dim. Current: ${SCORER_BLOCK_DIM}
  NUM_QUERIES                        Queries to run, 0=all converted queries. Current: ${NUM_QUERIES}
  RECALL_QUERIES                     CPU brute-force recall checks, 0=off. Current: ${RECALL_QUERIES}
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

# Run a compile/convert/build command, sending its output to a log file under
# --profile 2 instead of the console. cmake alone emits a few hundred lines and
# fr_converter/fr_builder add their own; none of it is interesting unless the step
# fails, so on failure the tail is printed and the path named. Nothing is lost either
# way, and the exit code always propagates.
run_build_tool() {
    if [[ "${PROFILE_MODE}" != "2" ]]; then
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

# Pre-built shards: reuse them, so convert+build are meaningless -> force off.
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

# ---- optional NPU AI-core utilization sampler (npu-smi, tooling-only) --------
NPU_UTIL_DIR=""
NPU_UTIL_PID=""
NPU_UTIL_DEVS=()

# Per-device HBM already in use, sampled BEFORE fr_search starts. This is the only
# unambiguous reading available: --mem-report calls aclrtGetMemInfo, which reports
# device-wide HBM and cannot tell our allocations from anyone else's, so on a shared
# card another tenant's memory lands in our "index" delta. Before our process exists,
# whatever is on the card is by definition not ours.
#
# This is not hypothetical. An 8-card run was credited with 27GB of "index" on dev4/dev5
# against ~4.6GB on the six clean cards -- 5.9x, on shards that are equal by
# construction -- and the host RSS total contradicted it. Those cards sat at 97% HBM and
# the run took a 285ms stall that showed up in every stage at once.
#
# npu-smi's second table gives per-process-per-card memory, which is the piece
# aclrtGetMemInfo structurally cannot provide, so "not-ours" here is measured rather
# than inferred. Falls back to the raw table if the parse yields nothing -- the format
# is version-specific (written against 24.1.0.3) and a wrong number is worse than none.
#
# Called before and after the search. fr_search has exited by the after call, so its
# own memory is already released and the after table cannot show what we used --
# --mem-report covers that. What the pair does show is whether the cards came back to
# where they started (a leak, or someone else's job arriving mid-run, both of which
# would explain stragglers and stalls).
npu_hbm_report() {
    local label="$1"
    command -v npu-smi >/dev/null 2>&1 || return 0
    local out parsed
    out="$(npu-smi info 2>/dev/null)" || return 0
    parsed="$(printf '%s\n' "${out}" | awk -F'|' '
        # Chip header "| 3  910B3 | OK |": $3 is the health string. The process rows
        # below look the same in $2, so key off $3 being alphabetic to tell them apart.
        $2 ~ /^ *[0-9]+ +[0-9A-Za-z]+ *$/ && $3 ~ /^ *[A-Za-z]/ { split($2,a," "); npu=a[1]; next }
        # Bus-Id row: HBM sits at the end of $4 as "21990/ 65536" or "4090 / 65536".
        $3 ~ /[0-9A-Fa-f]+:[0-9A-Fa-f]+:[0-9A-Fa-f]+\./ {
            if (match($4, /[0-9]+ *\/ *[0-9]+ *$/)) {
                hbm=substr($4, RSTART, RLENGTH); gsub(/ /,"",hbm);
                split(hbm,h,"/"); used[npu]=h[1]; tot[npu]=h[2]; order[++n]=npu
            } next }
        # Process row "| 3  0 | 594489 | fr_search | 18629 |": $3 is a pid.
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
    echo "[Step] NPU memory ${label} the search:"
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
    # Sample AI Core(%) and HBM(MB) from one `npu-smi info` call (it covers every card,
    # and both metrics live in the same table). Emits "npu aicore hbm_used" per chip.
    #
    # The id must come from the chip HEADER row ("| 3  910B3 | OK |"), not the Bus-Id
    # row: the latter's field 2 is the Chip index, which is 0 on every card, so reading
    # it there labels all cards 0 -- one card's samples get merged and the rest report
    # "no samples". These are npu-smi's PHYSICAL ids (this box exposes 3 and 7) and do
    # not have to match DEVICE_IDS, which are ACL logical ids, so report every card
    # found rather than filtering.
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
    # Kill the sampler on any exit so it never orphans (report path clears the pid).
    trap '[[ -n "${NPU_UTIL_PID}" ]] && kill "${NPU_UTIL_PID}" 2>/dev/null' EXIT
    echo "[NPUUTIL] sampling AI Core usage for devices: ${NPU_UTIL_DEVS[*]}"
}

npu_util_stop() {
    [[ -n "${NPU_UTIL_PID}" ]] || return 0
    # || true: wait on a killed job returns 143, and kill of a gone job is non-zero;
    # under `set -e` either would abort the script before the report below.
    kill "${NPU_UTIL_PID}" 2>/dev/null || true
    wait "${NPU_UTIL_PID}" 2>/dev/null || true
    NPU_UTIL_PID=""
    # Report every card npu-smi showed. The ids are physical and need not line up with
    # DEVICE_IDS (ACL logical), so filtering by DEVICE_IDS would drop real samples.
    # HBM peak is the number a spot-check by hand tends to miss: load and search have
    # different footprints, and a mid-run climb on a card means another tenant arrived.
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
    echo "       step aggregates [PERF] RecordGuard lines into an avg-per-stage table."
    [[ "${PROFILE_MODE}" == "2" ]] && \
        echo "       quiet: convert/build output -> log file, per-query recall lines -> count."
fi

echo "========================================"
echo "FullRecall new-project wrapper"
echo "Root:       ${ROOT_DIR}"
echo "CANN:       ${CANN}"
echo "SoC:        ${SOC_VERSION}"
echo "Dataset:    ${DATASET_FILE}"
if [[ "${DO_CONVERT_QUERY}" == "1" ]]; then
    echo "Tag map:    ${TAG_MAP_FILE}"
    echo "QueryData:  ${QUERYDATA_FILE}"
else
    echo "Queries:    reusing ${QUERY_OUT_DIR} (conversion skipped)"
fi
echo "RapidJSON:  ${RAPIDJSON_INCLUDE_DIR:-not found}"
echo "Work dir:   ${WORK_DIR}"
if [[ "${ROUND_ROBIN}" == "1" ]]; then
    echo "Mode:       ROUND-ROBIN baseline (${SHARD_NUM} cards, each = FULL corpus; queries work-stolen to idle cards, no merge)"
elif [[ "${MULTICARD}" == "1" ]]; then
    echo "Mode:       multi-card (${SHARD_NUM} shards on devices ${DEVICE_IDS})"
else
    echo "Mode:       single-card (device ${DEVICE_ID})"
fi
if [[ "${POOL_POSTINGS}" == "1" ]]; then
    echo "Postings:   pooled (NPUR_POOL_POSTINGS=1)"
else
    echo "Postings:   raw-malloc baseline (NPUR_POOL_POSTINGS=0)"
fi
if [[ "${NPUR_PARALLEL_SCORE_FILTER:-0}" == "1" ]]; then
    echo "Score/filter: parallel (NPUR_PARALLEL_SCORE_FILTER=1; better p50/p90, worse p99)"
elif [[ "${NPUR_OVERLAP_POSTEXPR:-1}" == "0" ]]; then
    echo "Score/filter: serial via executor (NPUR_OVERLAP_POSTEXPR=0; pre-overlap baseline)"
elif [[ "${NPUR_OVERLAP_POSTEXPR:-1}" == "2" ]]; then
    echo "Score/filter: serial, direct dispatch, no overlap (NPUR_OVERLAP_POSTEXPR=2; control arm for the default)"
else
    echo "Score/filter: serial, post-expr built inside the scorer kernel window (default)"
fi
if [[ "${NPUR_OVERLAP_POSTING:-0}" == "1" ]]; then
    echo "Posting prep: overlapped into the scorer window too (NPUR_OVERLAP_POSTING=1; PostingBitListToSet off the critical path)"
fi
if [[ "${STREAM_MERGE:-0}" == "1" ]]; then
    echo "Merge:      streaming (STREAM_MERGE=1; shards fold into a running top-K as they finish, hiding the k-way merge behind the straggler wait)"
fi
if [[ "${PACKED_SORT:-0}" == "1" || "${RADIX_SORT:-0}" == "1" ]]; then
    _hostsort="packed 8-byte keys"
    [[ "${RADIX_SORT:-0}" == "1" ]] && _hostsort="${_hostsort} + radix sort"
    echo "Host sort:  ${_hostsort} (per-shard top-K on the packed path)"
fi
if [[ -n "${SHARD_TOPK_RATIO:-}" && "${SHARD_TOPK_RATIO}" != "1" && "${SHARD_TOPK_RATIO}" != "1.0" ]]; then
    echo "Shard topK: ${SHARD_TOPK_RATIO}x per shard (SHARD_TOPK_RATIO; merge still returns the full topK; trades a little recall for smaller per-shard sort/TopK)"
fi
echo "Cores:      scorer ${SCORER_BLOCK_DIM} (Cube) + filter ${FULL_RECALL_TEXT_FILTER_BLOCK_DIM} (Vector)"
if [[ "${BATCH_SIZE}" != "1" ]]; then
    echo "Batch:      ${BATCH_SIZE} queries/batch (avg = ms/query = 1/throughput; p99 not a tail here)"
fi
if [[ -n "${RECALL_REF_FILE}" ]]; then
    echo "Recall ref: ${RECALL_REF_FILE} (cached, all queries)"
else
    echo "Recall ref: unset (spot-check ${RECALL_QUERIES} queries)"
fi
echo "========================================"

require_file "${DATASET_FILE}" "dataset.bin"

if [[ "${MULTICARD}" == "1" && "${EXTERNAL_SHARDS}" == "0" && "${DOCS}" -eq 0 ]]; then
    echo "[ERROR] multi-card build (DEVICE_IDS=${DEVICE_IDS}) needs DOCS=<total corpus size> so the" >&2
    echo "        corpus can be split into ${SHARD_NUM} equal shards. Set DOCS, or pass" >&2
    echo "        SHARD_INDEX_DIRS to search pre-built shards instead." >&2
    exit 1
fi

if [[ -f "${CANN}/bin/setenv.bash" ]]; then
    set +u
    # shellcheck disable=SC1090
    source "${CANN}/bin/setenv.bash"
    set -u
fi

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
echo "Host libs:   ${RUNTIME_LD_LIBRARY_PATH}"

if [[ "${DO_COMPILE}" == "1" ]]; then
    if [[ -z "${RAPIDJSON_INCLUDE_DIR}" || ! -f "${RAPIDJSON_INCLUDE_DIR}/rapidjson/document.h" ]]; then
        echo "[ERROR] RapidJSON headers not found." >&2
        echo "        Put RapidJSON at ${ROOT_DIR}/rapidjson-1.1.0/include," >&2
        echo "        or run: RAPIDJSON_INCLUDE_DIR=/path/to/rapidjson/include ./run.sh -c 1" >&2
        exit 1
    fi

    echo "[Step] Cleaning previous compile outputs..."
    rm -rf "${ROOT_DIR}/build" "${ROOT_DIR}/build_device"
    [[ "${PROFILE_MODE}" == "2" ]] && \
        echo "[Step] Compiling quietly (--profile 2); output -> ${RESULT_DIR}/log/build.log"

    echo "[Step] Building device kernels (serial; AscendC ExternalProject is not -j safe)..."
    run_build_tool cmake -S "${ROOT_DIR}/src/device" -B "${ROOT_DIR}/build_device" \
        -DASCEND_CANN_PACKAGE_PATH="${CANN}" \
        -DSOC_VERSION="${SOC_VERSION}"
    run_build_tool cmake --build "${ROOT_DIR}/build_device"

    echo "[Step] Building host tools..."
    run_build_tool cmake -S "${ROOT_DIR}/src" -B "${ROOT_DIR}/build" \
        -DASCEND_CANN_PACKAGE_PATH="${CANN}" \
        -DSOC_VERSION="${SOC_VERSION}" \
        -DRAPIDJSON_INCLUDE_DIR="${RAPIDJSON_INCLUDE_DIR}" \
        -DNPUR_EXTRA_RPATH_DIRS="${RUNTIME_RPATH_DIRS}"
    run_build_tool cmake --build "${ROOT_DIR}/build" -j "${BUILD_THREADS}"
else
    echo "[Step] Skipping compile due to -c 0. Reusing existing build/fr_* artifacts."
fi

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
    echo "[Step] Skipping query conversion."
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
        # one disjoint doc slice per shard; --doc_offset keeps ids globally absolute.
        # Create the shared parent first: fr_converter's mkdir is per-directory, so
        # the per-shard --out (CONVERTER_IN_DIR/shardN) needs its parent to exist.
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
    echo "[Step] Skipping dataset conversion."
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
        # Per-shard on-disk size. Shards are equal in doc count by construction, but not
        # in bytes -- posting lists follow content, and an 8-card build measured shard0
        # 29% smaller than its peers. This is also the only per-shard index figure that
        # is trustworthy on a shared card: --mem-report reads device-wide HBM
        # (aclrtGetMemInfo), so another tenant's allocations land in our numbers.
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
    echo "[Step] Skipping index build."
fi

if [[ "${DO_SEARCH}" == "1" ]]; then
    # locate index metadata; in multi-card mode assemble the shard dir list too
    shard_index_dirs=""
    tag_freq_file="${CONVERTER_IN_DIR}/tag_doc_freq.txt"
    if [[ "${EXTERNAL_SHARDS}" == "1" ]]; then
        # pre-built shards: search the given dirs directly
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
    # On-disk size of the index actually being searched. The build step prints this too,
    # but a search-only run reuses a prebuilt index and would otherwise show nothing.
    # du reads the disk, so it is right on a shared card where --mem-report (device-wide
    # HBM) is not.
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
        --effective_filter_file "${RESULT_DIR}/log/effective_filter_expressions.txt"
    )
    # tag-freq is a converter-produced diagnostic; absent when reusing prebuilt shards
    [[ -f "${tag_freq_file}" ]] && search_args+=(--converted_tag_freq_file "${tag_freq_file}")
    # index/device selection: multi-card searches N shards, single-card one index
    if [[ "${MULTICARD}" == "1" ]]; then
        search_args+=(--shard_index_dirs "${shard_index_dirs}" --device_ids "${DEVICE_IDS}")
    else
        search_args+=(--index_dir "${INDEX_DIR}" --device_id "${DEVICE_ID}")
    fi
    # optional recall ground-truth cache (computed once, reused across runs)
    if [[ -n "${RECALL_REF_FILE}" ]]; then
        search_args+=(--recall_ref_file "${RECALL_REF_FILE}" --recall_ref_threads "${RECALL_REF_THREADS}")
    fi
    if [[ "${NUM_QUERIES}" != "0" ]]; then
        search_args+=(--num_queries "${NUM_QUERIES}")
    fi
    search_args+=(--batch_size "${BATCH_SIZE}")
    # Optional aggregator / NPU-topk tuning knobs (gflags in the linked binary).
    # NPU_TOPK_LOOP_COUNT: more passes -> NPU topk converges closer to topK, so
    #   fewer over-returned candidates reach the host (cuts AggrAndTopK_fill_d2h_emplace).
    # NPU_TOPK_THRESHOLD_RATIO: matches>topK*ratio -> use NPU topk (lower = kicks in sooner).
    [[ -n "${NPU_TOPK_LOOP_COUNT:-}" ]] && \
        search_args+=(--full_recall_npu_topk_loop_count="${NPU_TOPK_LOOP_COUNT}")
    [[ -n "${NPU_TOPK_THRESHOLD_RATIO:-}" ]] && \
        search_args+=(--full_recall_npu_topk_enters_threshold_ratio="${NPU_TOPK_THRESHOLD_RATIO}")
    # NPU_TOPK_FINISH_BUFFER_RATIO: topk early-quit tolerance (wrapper default 1.2,
    # overriding the gflag default of 1.5). See the sweep in the defaults block above.
    search_args+=(--full_recall_npu_topk_finish_buffer_ratio="${NPU_TOPK_FINISH_BUFFER_RATIO}")
    [[ -n "${AGGREGATOR_BLOCK_DIM:-}" ]] && \
        search_args+=(--full_recall_aggregator_block_dim="${AGGREGATOR_BLOCK_DIM}")
    # SCORER_BLOCK_DIM: Cube cores for the vector MMad scorer kernel (wrapper default
    # 20, overriding the gflag default of 8). See the sweep in the defaults block near
    # the top. Recall unchanged (100%) at every value swept.
    search_args+=(--full_recall_scorer_block_dim="${SCORER_BLOCK_DIM}")
    # LATENCY_DUMP=1 -> per-query idx/ms/topk table for tail/outlier analysis.
    if [[ "${LATENCY_DUMP:-0}" == "1" ]]; then
        search_args+=(--latency_dump "${RESULT_DIR}/log/per_query_latency.tsv")
    fi
    # SHARD_WORKER_POOL: persistent per-shard worker threads instead of spawning one
    # per shard per query. Default ON; =0 restores the spawn-per-query path for A/B.
    #
    # This is the largest p99 win found so far. 2-shard, 3 interleaved rounds, every
    # metric better in all 3 (medians):
    #
    #     p99 6.336 -> 3.108ms (-51%)   p95 3.106 -> 2.826   avg 2.639 -> 2.473
    #
    # More telling is the spread: p99 over 9 runs was 5.07..9.98 spawning, 3.01..3.28
    # pooled. The spawn jitter was the dominant source of p99 noise all along -- what
    # earlier rounds of this work kept misattributing to "machine state".
    #
    # --shard_latency_dump shows the mechanism: merge_ms is untouched (0.132 ->
    # 0.130ms p99) and the entire win is in max_shard_ms (6.890 -> 2.851 p99, max
    # 32.1 -> 6.5ms). Spawning does not just add its own latency; the cold, freshly
    # scheduled thread makes the shard's own work slow and erratic. Pooled workers
    # are already warm and just wake on a condvar. The cost scales with shard count
    # (7 spawns per query at 8 shards vs 1 at 2), so this matters more, not less, on
    # the 8-card target config.
    if [[ "${SHARD_WORKER_POOL:-1}" != "0" ]]; then
        search_args+=(--shard_worker_pool)
    fi
    # STREAM_MERGE=1 -> fold each shard's sorted top-K into a running merge as it
    # finishes, hiding the O(K*N) host merge behind the straggler's wait. Default OFF
    # (barrier then k-way merge). Byte-identical results (deterministic tie-break).
    if [[ "${STREAM_MERGE:-0}" == "1" ]]; then
        search_args+=(--stream_merge)
    fi
    # ROUND_ROBIN=1 -> baseline dispatch: each (full-corpus) card pulls query chunks off a
    # shared cursor, no cross-card merge. See the ROUND_ROBIN note near the top.
    if [[ "${ROUND_ROBIN}" == "1" ]]; then
        search_args+=(--round_robin)
    fi
    # SHARD_LATENCY_DUMP=1 -> per-query per-shard TSV: shard0_ms..shardN_ms,
    # max_shard_ms, merge_ms, total_ms. Decomposes multi-card latency into "slowest
    # shard" vs "host merge" -- the ~0.8ms between BatchSearch and end-to-end that no
    # other timer covers. Needs --batch_size 1 to be per-query.
    if [[ "${SHARD_LATENCY_DUMP:-0}" == "1" ]]; then
        search_args+=(--shard_latency_dump "${RESULT_DIR}/log/per_shard_latency.tsv")
    fi
    # --mem-report 1 -> host peak RSS + per-device NPU HBM usage after the search.
    if [[ "${MEM_REPORT}" == "1" ]]; then
        search_args+=(--mem_report)
    fi

    # Repeat the search REPEAT times (compile/convert/build already ran once above;
    # only the search stage repeats). Per-run logs/perf files are suffixed so nothing
    # clobbers, and each run's end-to-end latency line is collected for the cross-run
    # median printed after the loop. The loop body keeps its original indentation on
    # purpose: re-indenting would break the column-0 <<'PY' heredoc terminators below.
    # Engine A/B toggle: fr_search reads NPUR_POOL_POSTINGS once at startup
    # (pooled vs raw-malloc devicePostings). Export so both invocations inherit it.
    export NPUR_POOL_POSTINGS="${POOL_POSTINGS}"
    # PACKED_SORT=1 -> the per-shard host top-K sorts 8-byte packed keys instead of the
    # 16-byte ScoreWithIndex (~-150us on that stage). Default off keeps the struct sort.
    if [[ "${PACKED_SORT:-0}" == "1" ]]; then
        export NPUR_PACKED_SORT=1
    fi
    # RADIX_SORT=1 -> sort the packed keys with LSD radix instead of std::sort (implies the
    # packed path, ~-40us on the sort stage). Off by default.
    if [[ "${RADIX_SORT:-0}" == "1" ]]; then
        export NPUR_RADIX_SORT=1
    fi
    # SHARD_TOPK_RATIO=<0..1> -> each shard aggregates only its top ceil(topK*ratio); the
    # host merge still returns the full topK from all shards. Smaller per-shard sort/TopK at
    # a little recall (global topK is split ~topK/N per shard). Default 1 (full).
    [[ -n "${SHARD_TOPK_RATIO:-}" ]] && export NPUR_SHARD_TOPK_RATIO="${SHARD_TOPK_RATIO}"
    npu_hbm_report before
    npu_util_start
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
        # Per-stage timing: fr_search's RecordGuard prints "[PERF] <stage> <us> us"
        # per query when NPUR_PERF=1. Stream-aggregate those lines with awk (so
        # the 10k-query x N-stage raw output never hits the log), pass [RESULT] and
        # everything else through, then print an avg-per-stage table sorted desc.
        perf_file="${RESULT_DIR}/log/fr_search_perf${run_suffix}.txt"
        : > "${perf_file}"
        # PERF_DUMP=1 also writes raw per-query 'q<TAB>stage<TAB>us' so we can
        # attribute the TAIL (slowest 1% of queries) per stage, not just the avg.
        # RecordGuard prints the outer "BatchSearch" line LAST per query (RAII), so
        # we tag every [PERF] line with the current query index q and bump q after
        # each BatchSearch. (q 0..4 are the 5 warmup queries; the post-process drops them.)
        # raw per-invocation 'q<TAB>stage<TAB>us' — always written. Used for the
        # per-stage percentile table below (which ignores q, so it is correct even
        # in multi-card) and, with PERF_DUMP=1, for the q-based tail attribution --
        # which is single-card only; see the guard on it below.
        perf_raw="${RESULT_DIR}/log/fr_search_perf_raw${run_suffix}.tsv"
        : > "${perf_raw}"
        # quiet=1 (--profile 2) folds the per-query recall lines into a count. The
        # harness prints one per mismatching query, which on a 10k cached run is a
        # screenful that the CPU-recall summary already totals. The count is still
        # printed, so a recall regression cannot hide -- and the raw lines survive in
        # fr_search_perf_raw / can be had back with --profile 1.
        NPUR_PERF=1 NPUR_LOG_LEVEL=ERROR "${ROOT_DIR}/build/fr_search" "${search_args[@]}" 2>&1 \
            | awk -v pf="${perf_file}" -v raw="${perf_raw}" -v quiet="$([[ "${PROFILE_MODE}" == "2" ]] && echo 1 || echo 0)" '
                /^\[PERF\]/ {
                    us=$(NF-1); tag=$2; for (i=3;i<=NF-2;i++) tag=tag " " $i;
                    s[tag]+=us; c[tag]++;
                    if (raw != "") printf "%d\t%s\t%d\n", q, tag, us > raw;
                    if (tag == "BatchSearch") q++;
                    next
                }
                quiet == 1 && /^\[RESULT\] q[0-9]+ recall=/ { mismatch++; next }
                { print }
                END {
                    if (quiet == 1 && mismatch > 0)
                        printf "[RESULT] %d per-query recall lines suppressed (--profile 2); see the CPU-recall summary\n", mismatch
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
            # Per-stage percentiles over ALL timed invocations. Percentiles need no
            # query grouping (only each stage's own duration list), so this is
            # correct in multi-card where q is per-shard, not per-query.
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
        # Tail attribution needs the q tagging above, which assumes ONE BatchSearch is
        # in flight at a time: it bumps q on each BatchSearch line and buckets every
        # [PERF] line into the current q. Multi-card breaks that -- the shards run in
        # parallel threads writing [PERF] to the same stderr, so q advances once per
        # shard rather than per query and the two shards' stages interleave into
        # arbitrary buckets. The output looks plausible and is not: a 2-shard run had
        # stages 7x FASTER on the "slow" queries, only 54% of the slow bucket's
        # BatchSearch accounted for by its own sub-stages (vs 91% overall), and a
        # different culprit each round. The percentile table above is unaffected --
        # it ignores q entirely.
        if [[ "${PERF_DUMP:-0}" == "1" && -s "${perf_raw}" && "${SHARD_NUM}" -gt 1 ]]; then
            echo "[Info] PERF_DUMP tail attribution skipped: it needs single-card (${SHARD_NUM} shards here)." >&2
            echo "       The q tagging assumes serial BatchSearch; parallel shards scramble the buckets." >&2
        elif [[ "${PERF_DUMP:-0}" == "1" && -s "${perf_raw}" ]]; then
            # Tail attribution: per-stage avg over ALL timed queries vs over the
            # slowest 1% (ranked by their BatchSearch total). Shows which stage
            # blows up on the p99 queries.
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
        NPUR_LOG_LEVEL=ERROR "${ROOT_DIR}/build/fr_search" "${search_args[@]}" 2>&1 | tee "${search_log}"
        rc=${PIPESTATUS[0]}
    fi
    set -e
    if [[ "${rc}" -ne 0 ]]; then
        echo "[ERROR] fr_search failed (run ${run}/${REPEAT}) with exit code ${rc}. See ${search_log}" >&2
        exit "${rc}"
    fi
    lat_line="$(grep -m1 '^\[RESULT\] latency ms:' "${search_log}" 2>/dev/null || true)"
    [[ -n "${lat_line}" ]] && run_latency_lines+=("${lat_line}")
    echo "[Done] Search log: ${search_log}"
    done
    npu_util_stop
    npu_hbm_report after

    # Cross-run median of the end-to-end latency stats. A single 10k-query run's p99
    # tail is not reproducible run-to-run; the median over REPEAT runs de-noises it.
    # Each run's value is shown alongside so the spread is visible.
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
# The harness now prints qps directly (queries / wall-clock). Median it across runs.
if all("qps" in r for r in runs):
    print(f"\n[RESULT] throughput (median): {statistics.median([r['qps'] for r in runs]):.1f} qps")
PY
    fi
else
    echo "[Step] Skipping search."
fi

# Opt-in email notification: only after a search that actually ran (we reach here
# only on success; a failed fr_search exits above). Never fail the run on a
# notify error.
if [[ "${DO_SEARCH}" == "1" && "${NOTIFY}" == "1" ]]; then
    if [[ -z "${NOTIFY_API_KEY}" ]]; then
        echo "[Notify] NOTIFY=1 but NOTIFY_API_KEY is empty; skipping email." \
             "Put NOTIFY_API_KEY in ${ROOT_DIR}/.notify.env" >&2
    elif ! command -v curl >/dev/null 2>&1; then
        echo "[Notify] curl not found; skipping email." >&2
    else
        # Headline latency: prefer the cross-run median -- a single run's p99 is not
        # reproducible, so the last run's number would be actively misleading here.
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
        # Recall headline. The harness prints "[RESULT] CPU-recall over N queries:
        # avg=X%" only when verification actually ran, so its absence is itself a
        # result: report it loudly instead of omitting it, or a quiet email reads as
        # a pass when recall was never checked at all.
        notify_recall="$(grep -m1 'CPU-recall over' "${search_log}" 2>/dev/null \
                         | sed -n 's/.*avg=\(.*\)/\1/p' || true)"
        [[ -z "${notify_recall}" ]] && notify_recall="NOT-VERIFIED"
        # Per-query lines are printed for every mismatch, plus the first 10 as a
        # sanity check when the ref is not cached -- so only recall<1 counts as bad.
        notify_bad="$(grep '^\[RESULT\] q[0-9]* recall=' "${search_log}" 2>/dev/null \
                      | grep -v 'recall=1\.0000' || true)"
        notify_bad_n=0
        [[ -n "${notify_bad}" ]] && notify_bad_n="$(wc -l <<<"${notify_bad}" | tr -d ' ')"
        [[ "${NUM_QUERIES}" == "0" ]] && notify_nq="all" || notify_nq="${NUM_QUERIES}"
        if [[ "${MULTICARD}" == "1" ]]; then
            notify_devices="${DEVICE_IDS} (${SHARD_NUM} shards)"
        else
            notify_devices="${DEVICE_ID} (single card)"
        fi
        # Knobs that deviate from the tuned defaults, appended to the subject. CONFIG_TAG
        # only carries segment size and density, so without this every mail from an A/B
        # sweep of any other knob has an identical subject and only the body tells them
        # apart. Empty on a default run.
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
        # Build the JSON payload with python3 so subject/body are escaped safely.
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

echo "[Done] hx_npu run completed."
