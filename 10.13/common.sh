#!/usr/bin/env bash
# 三个驱动脚本共用。被 source，不单独执行。
# 数据、日志、索引都在 release 内，三份代码共用；每份只有自己的 build/。

RELEASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 已经是真目录就不动它，否则（含软链、悬空软链）指向共用的那份
link_shared() {   # <目标> <链接路径>
    [[ -d "$2" && ! -L "$2" ]] && return 0
    ln -sfn "$1" "$2" 2>/dev/null || true
}

prepare() {   # <cards|load|index>
    local name="$1" v ignored=()
    # 容器可能导出 WORK_DIR=/home/ma-user/work 等通用路径；交付入口统一按 DATA_ROOT
    # 派生路径。确需覆盖时，把赋值作为 tools/exp.sh 的场景参数传入。
    for v in WORK_DIR RUNS_DIR INDEX_DIR; do
        if [[ -n "${!v:-}" ]]; then
            ignored+=("${v}=${!v}")
            unset "${v}"
        fi
    done
    if (( ${#ignored[@]} > 0 )); then
        printf '[路径] 忽略外部设置，按数据目录生成产物路径：%s\n' "${ignored[*]}"
    fi
    COPY="${RELEASE}/${name}"
    ENTRY="${COPY}/run/${name}.sh"
    [[ -x "${ENTRY}" ]] || { echo "[错误] 缺少入口脚本：${ENTRY}" >&2; exit 1; }

    # 数据：显式给的优先，其次 release/dataset；兼容旧包的 frnew_syn_v3。
    if [[ -z "${DATA_ROOT:-}" ]]; then
        local d
        for d in "${RELEASE}/dataset" "${RELEASE}/frnew_syn_v3"; do
            [[ -f "${d}/dataset.bin" ]] && { DATA_ROOT="$(cd "${d}" && pwd)"; break; }
        done
    fi
    if [[ -z "${DATA_ROOT:-}" || ! -f "${DATA_ROOT:-}/dataset.bin" ]]; then
        cat >&2 <<EOT
[错误] 未找到 dataset.bin。
  默认位置：${RELEASE}/dataset/dataset.bin
  当前 DATA_ROOT：${DATA_ROOT:-未设置}
请把数据放入 release/dataset，或在运行前设置 DATA_ROOT=<数据目录>。
EOT
        exit 1
    fi
    DATA_ROOT="$(cd "${DATA_ROOT}" && pwd)"
    export DATA_ROOT
    export LOG_DIR="${LOG_DIR:-${RELEASE}/logs}"
    mkdir -p "${LOG_DIR}"

    # 副本里挂上软链，这样进到 release/<名字> 直接跑 tools/exp.sh 也能拿到共用的数据
    link_shared "${DATA_ROOT}" "${COPY}/frnew_syn_v3"
    link_shared "${LOG_DIR}" "${COPY}/logs"

    if [[ "${COMPILE:-0}" == "1" ]]; then
        printf '[编译] %s\n' "${name}"
        ( cd "${COPY}" && tools/exp.sh compile ) || exit 1
    fi
    unset COMPILE

    # PREPARE=1：只认数据、挂软链、按需编译，不跑验收项。之后进 release/<名字> 跑
    # tools/exp.sh 就不用再带 DATA_ROOT 了。
    if [[ "${PREPARE:-0}" == "1" ]]; then
        printf '[完成] %s 已就绪。后续实验入口：%s/tools/exp.sh\n' "${name}" "${COPY}"
        exit 0
    fi
}
