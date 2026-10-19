#!/usr/bin/env bash
# 验收项「索引加载时间、索引大小、内存占用」
# 零参数。每份第一次跑加 COMPILE=1 先编出自己的 build/。
# PREPARE=1 只做准备和编译，不跑验收项。
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
prepare index
exec "${ENTRY}" "$@"
