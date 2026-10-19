#!/usr/bin/env bash
# 验收项：不同卡数下的延迟与吞吐（扩展性）
#
#   ./run/cards.sh
#
# 零参数。按卡数分档逐次运行，每档内扫 Batch Size 1/2/4/8，每档出一张表。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

# 卡数：优先按机器上可见的 davinci 设备数自动探测；探测不到再用 FALLBACK。
# 不探测的话，两卡机器上要跑到 4 卡那一档才撞上 aclrtSetDevice 的 107001（无效设备号），
# 前一档已经白跑了。CARDS=<n> 可以显式覆盖。
detect_cards() {
    local n
    n="$(ls -d /dev/davinci[0-9]* 2>/dev/null | wc -l | tr -d ' ')"
    [[ "${n}" -gt 0 ]] && { echo "${n}"; return 0; }
    n="$(npu-smi info -l 2>/dev/null | awk -F: '/Total *Count/{gsub(/[^0-9]/,"",$2); print $2; exit}')"
    [[ -n "${n}" && "${n}" -gt 0 ]] && { echo "${n}"; return 0; }
    echo 0
}
FALLBACK_CARDS=8        # 探测失败时用这个
CARDS="${CARDS:-$(detect_cards)}"
(( CARDS > 0 )) || CARDS="${FALLBACK_CARDS}"
CARDS_PER_GROUP=2       # 每组卡数保持不变，卡数增加等于组数增加
BATCH_SIZES="1 2 4 8"

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/accept_cards_$(date +%m%d-%H%M%S).log"

echo "验收项    不同卡数下的延迟与吞吐（扩展性）"
echo "本机卡数  ${CARDS} 张（自动探测，CARDS=<n> 可覆盖）"
echo "合格标准  BATCH_SIZE=1 的 p99 需 ≤ 2ms；其余 Batch Size 关注吞吐。"
echo "预期趋势  每组固定 ${CARDS_PER_GROUP} 张卡，卡数翻倍即组数翻倍，吞吐应随之提升，延迟应保持平稳。"
echo "读数说明  BATCH_SIZE>1 时 p99 不代表尾部延迟（同批查询共享批平均值），"
echo "          单条查询的平均耗时取 avg × BATCH_SIZE。"
echo "日志      ${LOG}"
echo

# 整个 for 循环接管道会跑在子 shell 里，rc 带不出来；所以逐条 tee -a，循环留在当前 shell
: > "${LOG}"
rc=0
for ((n = CARDS_PER_GROUP; n <= CARDS; n += CARDS_PER_GROUP)); do
    devs=""
    for ((i = 0; i < n; i++)); do devs+="${devs:+,}${i}"; done
    echo "######################## ${n} 卡 ########################" | tee -a "${LOG}"
    tools/exp.sh latency DEVICE_IDS="${devs}" BATCH_SIZES="${BATCH_SIZES}" 2>&1 | tee -a "${LOG}"
    (( PIPESTATUS[0] == 0 )) || rc=1
done
(( rc == 0 )) || echo "[Warn] 有卡数档位执行失败，详见 ${LOG}" | tee -a "${LOG}"
exit "${rc}"
