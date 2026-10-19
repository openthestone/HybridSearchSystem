#!/usr/bin/env bash
# 在交付包根目录只生成或重算召回参考缓存，不重建索引。
#   ./ref.sh              索引已建好、缓存尚不存在时
#   ./ref.sh FORCE=1      重算当前版本缓存并与旧文件逐字节比较
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

prepare cards
echo "召回参考缓存"
printf '  数据目录：%s\n' "${DATA_ROOT}"
echo
( cd "${COPY}" && tools/exp.sh ref "$@" )
