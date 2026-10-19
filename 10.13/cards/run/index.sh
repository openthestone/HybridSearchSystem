#!/usr/bin/env bash
# 验收项：索引加载时间、索引大小、内存占用
#
#   ./run/index.sh
#
# 零参数。跑两次：并行加载开和关，用来对照。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/accept_index_$(date +%m%d-%H%M%S).log"

echo "验收项    索引加载时间、索引大小、内存占用"
echo "索引大小  见开头按文件后缀的拆分，其中 .posting 是密度阈值唯一影响的部分。"
echo "加载时间  见「索引加载」的墙钟耗时与各分片耗时，括号内为三次样本的极差。"
echo "内存占用  见 [MEM] 行：after_load−baseline 为索引静态占用，"
echo "          after_search−after_load 为运行时开销，host RSS peak 为主机内存峰值。"
echo "日志      ${LOG}"
echo

# 逐条 tee -a，不把整组接到一个管道上——那样退出码和变量都留在子 shell 里
: > "${LOG}"
rc=0
echo "######################## 并行加载（默认） ########################" | tee -a "${LOG}"
tools/exp.sh index MEM_REPORT=1 2>&1 | tee -a "${LOG}"
(( PIPESTATUS[0] == 0 )) || rc=1
echo "######################## 串行加载（对照） ########################" | tee -a "${LOG}"
tools/exp.sh index MEM_REPORT=1 PARALLEL_LOAD=0 2>&1 | tee -a "${LOG}"
(( PIPESTATUS[0] == 0 )) || rc=1
exit "${rc}"
