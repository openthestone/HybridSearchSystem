#!/usr/bin/env bash
# 一次性准备：转查询 + 建索引 + 建召回参考缓存，然后把三份代码都编好。
# 产物全部落在共用的数据目录，三项验收共用；跑完三个入口就能直接跑，不用再加 COMPILE=1。
#
#   ./setup.sh                 切 2 份分片（三项验收都是每组 2 张卡，2 份就够）
#   ./setup.sh CONVERT_QUERY=0 查询已经转好过
#   ./setup.sh FORCE=1         覆盖已有索引
#   SHARDS=4 ./setup.sh        改分片数
#
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ln -sf $ROOT/third_party/ $ROOT/cards/third_party
ln -sf $ROOT/third_party/ $ROOT/index/third_party
ln -sf $ROOT/third_party/ $ROOT/load/third_party
ln -sf $ROOT/logs/ $ROOT/cards/logs
ln -sf $ROOT/logs/ $ROOT/index/logs
ln -sf $ROOT/logs/ $ROOT/load/logs
ln -sf $ROOT/third_party/lib/ $ROOT/third_party/lib64
ln -sf $ROOT/third_party/lib/libprotobuf.so.3.21.12.0 $ROOT/third_party/lib/libprotobuf.so
ln -sf $ROOT/third_party/lib/libprotobuf-lite.so.3.21.12.0 $ROOT/third_party/lib/libprotobuf-lite.so
ln -sf $ROOT/third_party/lib/libprotoc.so.3.21.12.0 $ROOT/third_party/lib/libprotoc.so
ln -sf $ROOT/third_party/lib/libgflags.so.2.2.2 $ROOT/third_party/lib/libgflags.so
ln -sf $ROOT/third_party/lib/libgflags_nothreads.so.2.2.2 $ROOT/third_party/lib/libgflags_nothreads.so
ldconfig -n $ROOT/third_party/lib
export RAPIDJSON_INCLUDE_DIR=$ROOT/third_party/include
# 用 cards 那份的代码建（三份由 tools/release_sync.sh 复制而来，内容相同）。
set -uo pipefail
unset PREPARE
# setup 总会在建库阶段编译 cards，避免继承的 COMPILE=1 提前触发单独编译。
unset COMPILE
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

prepare cards
SHARDS="${SHARDS:-2}"

echo "准备交付环境"
printf '  数据目录：%s\n' "${DATA_ROOT}"
echo
echo "[1/3] 编译 cards，转换数据，构建索引并生成召回缓存"
( cd "${COPY}" && tools/exp.sh build COMPILE=1 SHARDS="${SHARDS}" RECALL_REF=1 "$@" ) || exit 1

step=1
for name in load index; do
    step=$((step + 1))
    echo
    printf '[%s/3] 编译 %s\n' "${step}" "${name}"
    prepare "${name}"
    ( cd "${COPY}" && tools/exp.sh compile ) || exit 1
done

echo
echo "[完成] 索引、召回缓存和三份程序已准备好。"
echo "接下来在 release 根目录运行：./load.sh、./cards.sh 或 ./index.sh"
