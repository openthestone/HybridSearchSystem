#!/usr/bin/env bash
# 验收项：模拟 NPU 负载波动下，动态任务分配相对固定任务分配的吞吐与延迟
#
#   ./run/load.sh
#
# 零参数。要换机器只改下面 CARDS 一行。需要 ≥4 张卡：组数必须 ≥2，否则固定分配和
# 动态分配取 chunk 的序列完全一样（schedule_groups.cpp:282），A/B 必然得出零效应。
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
CARDS_PER_GROUP=2       # 每组卡数；组数 = CARDS / 这个值

DEVS=""                 # 不用 seq -s,：BSD 的 seq 会多一个尾分隔符
for ((i = 0; i < CARDS; i++)); do DEVS+="${DEVS:+,}${i}"; done

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/accept_load_$(date +%m%d-%H%M%S).log"

echo "验收项    模拟负载波动下，动态任务分配相对固定任务分配的吞吐与延迟"
echo "本机卡数  ${CARDS} 张（自动探测，CARDS=<n> 可覆盖）"
echo "合格标准  查看 [loaded] 表的「变化 dynamic」行：吞吐提升 ≥ 15%，p99 降低 ≥ 10%。"
echo "          两项增益均需大于同表末行的「同配置波动」，否则落在测量噪声内，不成立。"
echo "对照组    [noload] 表两种分配应基本持平：无负载时调度方式本不应产生差异。"
echo "日志      ${LOG}"
echo

tools/exp.sh load DEVICE_IDS="${DEVS}" CARDS_PER_GROUP="${CARDS_PER_GROUP}" BATCH_SIZE=2 2>&1 | tee "${LOG}"
exit "${PIPESTATUS[0]}"
