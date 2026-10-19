#!/usr/bin/env bash
# 验收项「模拟 NPU 负载波动下，动态任务分配相对固定任务分配的吞吐与延迟」
# 零参数。每份第一次跑加 COMPILE=1 先编出自己的 build/。
# PREPARE=1 只做准备和编译，不跑验收项。
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
prepare load
exec "${ENTRY}" "$@"
