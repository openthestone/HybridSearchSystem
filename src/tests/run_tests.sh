#!/usr/bin/env bash
# Build + run the src/ host-side unit tests. Pure C++17 (no CANN, no protobuf),
# so these run on the local dev box as well as the server.
#
#   src/tests/run_tests.sh
#
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# include root so `src/full_recall/...` resolves into engine/ (as the CMake build does)
mkdir -p "$work/inc/src"
ln -sfn "$repo/src/engine" "$work/inc/src/full_recall"

CXX="${CXX:-c++}"
STD="-std=c++17"
STUBS="-I$repo/src/platform -I$repo/src/platform/src"
PRELUDE="-include $repo/src/platform/npur_prelude.h"
COMMON="$STD -I$repo -Wall -Wextra -Wno-unused-parameter"

rc=0
run() {  # name  <extra compile args...>
    local name="$1"; shift
    local bin="$work/$name"
    if ! $CXX $COMMON "$@" -o "$bin" 2> "$work/$name.log"; then
        echo "[BUILD-FAIL] $name"; sed 's/^/    /' "$work/$name.log"; rc=1; return
    fi
    if "$bin"; then echo "[PASS] $name"; else echo "[FAIL] $name"; rc=1; fi
}

# 1) filter parser / evaluators — pure header
run test_filter_expr "$here/test_filter_expr.cpp"

# 2) fvecs query loader — pure header
run test_query_io "$here/test_query_io.cpp"

# 3) record envelope round-trip through engine/'s REAL ReadAndDoTask
run test_record_io \
    "$here/test_record_io.cpp" \
    "$repo/src/engine/indexer/file/file_reader.cpp" \
    "$repo/src/engine/indexer/file/memory_data_manager.cpp" \
    -I"$work/inc" $STUBS $PRELUDE

echo "-----"
[ $rc -eq 0 ] && echo "ALL TESTS PASSED" || echo "SOME TESTS FAILED"
exit $rc
