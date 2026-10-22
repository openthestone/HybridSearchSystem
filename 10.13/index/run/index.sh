#!/usr/bin/env bash
# 验收项：索引加载时间、索引大小
#
#   ./run/index.sh
#
# 两档对比：密度阈值 0 的未压缩索引 vs 压缩后的索引，各自读进 NPU，比耗时。
# 走的是 fr_loadbench，不启动引擎、不建表，只做读盘 + 拷到设备两件事。
# 单卡显存 32 GiB，所以每档最多读 CAP_GIB，超出的部分不读。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

DENSITY_BASE="${DENSITY_BASE:-0}"
DENSITY_TUNED="${DENSITY_THRESHOLD:-0.00026}"
DEVICE_IDS="${DEVICE_IDS:-0,1}"
REPEAT="${REPEAT:-3}"
# 单线程读：和引擎一致（fr_search 分片内就是单线程读），报出去的才是真实耗时。
#   READ_THREADS=4 ./run/index.sh   并行读，快一倍多，但引擎没有这个能力
READ_THREADS="${READ_THREADS:-1}"
# 每卡绑一个 NUMA 节点：只稳定拷贝段、不改读盘速度，而且引擎本来就有同名开关
# （run.sh --card_cpus）。两张卡挤同一个节点会明显变慢，置空则不绑。
CARD_CPUS="${CARD_CPUS-auto}"
CAP_GIB="${CAP_GIB:-32}"              # 910B4 单卡显存，整份索引按这个封顶
HBM_BUDGET_MIB="${HBM_BUDGET_MIB:-26214}"   # 32 GiB x 0.8，体积验收线
RUNS_DIR="${RUNS_DIR:-${DATA_ROOT:-${PWD}/frnew_syn_v3}/runs}"
CARDS="$(awk -F, '{print NF}' <<< "${DEVICE_IDS}")"

BIN=build/fr_loadbench
[[ -x "${BIN}" ]] || { echo "[错误] 缺少 ${PWD}/${BIN}，先 tools/exp.sh compile" >&2; exit 1; }

dirs_of() {   # <阈值>  ->  shard0,shard1,...
    local t="$1" i out=""
    for ((i = 0; i < CARDS; i++)); do out+="${out:+,}${RUNS_DIR}/den${t}/work/index/shard${i}"; done
    echo "${out}"
}
mib_of() {    # <阈值>  ->  "总 MiB 最小分片 MiB"。分片不是严格等大，每卡的配额得按最小的来，
              # 否则总量除以卡数会超过某一片的实际大小，那一片就读不满。
    local t="$1" i sum=0 min=-1 d one
    for ((i = 0; i < CARDS; i++)); do
        d="${RUNS_DIR}/den${t}/work/index/shard${i}"
        [[ -d "${d}" ]] || { echo "0 0"; return; }
        one="$(du -sm "${d}" | cut -f1)"
        sum=$((sum + one))
        (( min < 0 || one < min )) && min="${one}"
    done
    echo "${sum} ${min}"
}

for t in "${DENSITY_BASE}" "${DENSITY_TUNED}"; do
    [[ "$(mib_of "${t}" | cut -d' ' -f1)" -gt 0 ]] && continue
    echo "[错误] 密度阈值 ${t} 的索引不全：${RUNS_DIR}/den${t}/work/index/shard0..$((CARDS - 1))" >&2
    echo "       先建出来：tools/exp.sh build SHARDS=${CARDS} CONVERT_QUERY=0 DENSITY_THRESHOLD=${t}" >&2
    exit 1
done

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/accept_index_$(date +%m%d-%H%M%S).log"

echo "验收项    索引加载时间、索引大小"
echo "对照      密度阈值 ${DENSITY_BASE}（未压缩）与 ${DENSITY_TUNED}（压缩后），各自读进 ${CARDS} 张卡"
echo "做法      不启动引擎，只做读盘 + 拷到设备；整份索引超过 ${CAP_GIB} GiB 的部分不读"
echo "口径      读线程 ${READ_THREADS}（与引擎一致）、每卡绑核「${CARD_CPUS:-不绑}」"
echo "合格标准  索引体积 ≤ ${HBM_BUDGET_MIB} MiB（32 GiB x 0.8）"
echo "日志      ${LOG}"
echo

: > "${LOG}"
rc=0
MEDIAN=""

run_den() {   # <标题> <阈值>
    local title="$1" t="$2" mib minmib per tmp
    read -r mib minmib <<< "$(mib_of "${t}")"
    # 每卡上限就是显存封顶的份额；分片比这个小的话 fr_loadbench 读多少算多少，不用在这里算
    per="$(awk -v cap="${CAP_GIB}" -v n="${CARDS}" 'BEGIN{printf "%.6g", cap / n}')"
    echo "######################## ${title} ########################" | tee -a "${LOG}"
    awk -v m="${mib}" -v cap="${CAP_GIB}" -v b="${HBM_BUDGET_MIB}" \
        'BEGIN{printf "  索引体积 %d MiB%s；对预算 %d MiB %s %d MiB\n",
               m, (m / 1024 > cap ? sprintf("，本次只读前 %g GiB", cap) : "，整份都读"),
               b, (m <= b ? "余量" : "超出"), (m <= b ? b - m : m - b)}' | tee -a "${LOG}"
    tmp="$(mktemp)"
    "${BIN}" --devices "${DEVICE_IDS}" --gib_per_card "${per}" --repeat "${REPEAT}" \
        --read_threads "${READ_THREADS}" --card_cpus "${CARD_CPUS}" \
        --dirs "$(dirs_of "${t}")" 2>&1 | tee -a "${LOG}" "${tmp}"
    (( PIPESTATUS[0] == 0 )) || rc=1
    MEDIAN="$(sed -n 's/.*中位数 \([0-9.][0-9.]*\)s.*/\1/p' "${tmp}" | tail -1)"
    rm -f "${tmp}"
}

run_den "密度阈值 ${DENSITY_BASE}（未压缩）" "${DENSITY_BASE}"
BASE_SEC="${MEDIAN}"
echo | tee -a "${LOG}"
run_den "密度阈值 ${DENSITY_TUNED}（压缩后）" "${DENSITY_TUNED}"
TUNED_SEC="${MEDIAN}"

echo | tee -a "${LOG}"
echo "######################## 对比 ########################" | tee -a "${LOG}"
if [[ -n "${BASE_SEC}" && -n "${TUNED_SEC}" ]]; then
    awk -v a="${BASE_SEC}" -v b="${TUNED_SEC}" \
        'BEGIN{d = 100 * (b / a - 1);
               printf "压缩后相对未压缩 %s %.1f%%：%.2f s -> %.2f s\n",
                      (d < 0 ? "快" : "慢"), (d < 0 ? -d : d), a, b}' | tee -a "${LOG}"
else
    echo "没解析到耗时（某一档没跑成）" | tee -a "${LOG}"
    rc=1
fi
exit "${rc}"
