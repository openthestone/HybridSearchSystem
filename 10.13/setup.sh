#!/usr/bin/env bash
# 一次性准备：转查询 + 建索引 + 建召回参考缓存，然后把三份代码都编好。
# 产物全部落在共用的数据目录，三项验收共用；跑完三个入口就能直接跑，不用再加 COMPILE=1。
#
#   ./setup.sh                 切 2 份分片（三项验收都是每组 2 张卡，2 份就够）
#   ./setup.sh CONVERT_QUERY=0 查询已经转好过
#   ./setup.sh FORCE=1         重做全部初始化：覆盖已有索引，并重算召回缓存
#   SHARDS=4 ./setup.sh        改分片数
#   BASELINE=0 ./setup.sh      跳过未压缩基线索引（index.sh 的对照档就没得比了）
#   ./setup.sh COMPILE_ONLY=1  只重编三份代码，不碰索引和缓存（改过 C++ 后用这个）
#
# 用 cards 那份的代码建（三份由 tools/release_sync.sh 复制而来，内容相同）。
set -uo pipefail
unset PREPARE
# setup 总会在建库阶段编译 cards，避免继承的 COMPILE=1 提前触发单独编译。
unset COMPILE
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

prepare_third_party() {
    local tp="${RELEASE}/third_party" lib f base
    [[ -d "${tp}" ]] || return 0

    [[ -d "${tp}/lib64" && ! -e "${tp}/lib" ]] && ln -sfn lib64 "${tp}/lib"
    [[ -d "${tp}/lib" && ! -e "${tp}/lib64" ]] && ln -sfn lib "${tp}/lib64"

    for lib in "${tp}/lib" "${tp}/lib64"; do
        [[ -d "${lib}" && ! -L "${lib}" ]] || continue
        ldconfig -n "${lib}" 2>/dev/null
        for f in "${lib}"/*.so.*; do
            [[ -e "${f}" ]] || continue
            base="${f%%.so.*}.so"
            [[ -e "${base}" ]] || ln -sf "$(basename "${f}")" "${base}"
        done
    done

    if [[ -z "${RAPIDJSON_INCLUDE_DIR:-}" ]]; then
        for f in "${tp}/include" "${tp}/rapidjson/include" "${tp}/rapidjson-1.1.0/include"; do
            [[ -f "${f}/rapidjson/document.h" ]] && { export RAPIDJSON_INCLUDE_DIR="${f}"; break; }
        done
    fi
    echo "[依赖] ${tp}${RAPIDJSON_INCLUDE_DIR:+    RapidJSON ${RAPIDJSON_INCLUDE_DIR}}"
}

prepare cards
prepare_third_party
SHARDS="${SHARDS:-2}"

# FORCE 写成环境变量还是参数都认，统一补成参数传下去：留在环境里会被 exp.sh 当成
# "继承来的配置"报一行提示，而且后面编译那两步也不需要它。
FORCE_ALL=0
[[ "${FORCE:-0}" == "1" ]] && FORCE_ALL=1
for _a in "$@"; do [[ "${_a}" == "FORCE=1" ]] && FORCE_ALL=1; done
unset FORCE
BUILD_ARGS=("$@")
(( FORCE_ALL )) && [[ " ${BUILD_ARGS[*]-} " != *" FORCE=1 "* ]] && BUILD_ARGS+=(FORCE=1)

echo "准备交付环境"
printf '  数据目录：%s\n' "${DATA_ROOT}"
(( FORCE_ALL )) && echo "  FORCE=1：索引与召回缓存都会重做，编译本来每次都重做"
echo
# 改过 C++ 之后只要重编，没必要把十几分钟的建库再走一遍
COMPILE_ONLY="${COMPILE_ONLY:-0}"
for _a in "$@"; do [[ "${_a}" == "COMPILE_ONLY=1" ]] && COMPILE_ONLY=1; done
if (( COMPILE_ONLY )); then
    n=0
    for name in cards load index; do
        n=$((n + 1))
        echo
        printf '[%s/3] 编译 %s\n' "${n}" "${name}"
        prepare "${name}"
        ( cd "${COPY}" && tools/exp.sh compile ) || exit 1
    done
    echo
    echo "[完成] 三份代码已重新编译，索引和缓存没有改动。"
    exit 0
fi

BASELINE="${BASELINE:-1}"
BASELINE_INDEX="${DATA_ROOT}/runs/den0/work/index"
TOTAL=3
(( BASELINE )) && TOTAL=4

echo "[1/${TOTAL}] 编译 cards，转换数据，构建索引并生成召回缓存"
( cd "${COPY}" && tools/exp.sh build COMPILE=1 SHARDS="${SHARDS}" RECALL_REF=1 \
    ${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"} ) || exit 1

step=1
if (( BASELINE )); then
    step=$((step + 1))
    echo
    printf '[%s/%s] 构建未压缩基线索引（密度阈值 0，index.sh 的对照档）\n' "${step}" "${TOTAL}"
    if [[ -d "${BASELINE_INDEX}" && -n "$(ls -A "${BASELINE_INDEX}" 2>/dev/null)" ]] && (( ! FORCE_ALL )); then
        echo "  已存在，跳过：${BASELINE_INDEX}"
    else
        # CONVERT_QUERY=0 是死规矩：上一步已经转过查询了，再转一次会刷新 queries.fvecs
        # 的 mtime，把刚建好的召回缓存指纹作废。
        ( cd "${COPY}" && tools/exp.sh build SHARDS="${SHARDS}" DENSITY_THRESHOLD=0 \
            CONVERT_QUERY=0 $( (( FORCE_ALL )) && echo FORCE=1 ) ) || exit 1
    fi
fi

for name in load index; do
    step=$((step + 1))
    echo
    printf '[%s/%s] 编译 %s\n' "${step}" "${TOTAL}" "${name}"
    prepare "${name}"
    ( cd "${COPY}" && tools/exp.sh compile ) || exit 1
done

echo
echo "[完成] 索引、召回缓存和三份程序已准备好。"
echo "接下来在 release 根目录运行：./load.sh、./cards.sh 或 ./index.sh"
