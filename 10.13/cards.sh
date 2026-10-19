#!/usr/bin/env bash
# 验收项「不同卡数下的延迟与吞吐（扩展性）」
# 零参数。每份第一次跑加 COMPILE=1 先编出自己的 build/。
# PREPARE=1 只做准备和编译，不跑验收项。
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
prepare cards
exec "${ENTRY}" "$@"
