#!/usr/bin/env bash
# 开发侧：把仓库里的代码同步成 release 下的三份。这个脚本不随 release 交付，
# 交付前在本机跑一次，再把整个 release 目录打包发出去。
# 每份的 build/ build_device/ 不动，数据和日志的软链由三个驱动脚本自己维护。
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RELEASE="${REPO}/release"
COPIES=(cards load index)
PARTS=(run.sh src tools run scripts)

for p in "${PARTS[@]}"; do
    [[ -e "${REPO}/${p}" ]] || { echo "[错误] 缺少源码目录：${REPO}/${p}" >&2; exit 1; }
done
for f in common.sh setup.sh ref.sh cards.sh load.sh index.sh; do
    [[ -f "${RELEASE}/${f}" ]] || { echo "[错误] 缺少交付入口：${RELEASE}/${f}" >&2; exit 1; }
done

# release 打包出去就是根目录，所以文档里让人跑的东西都得在里面：依赖脚本和被引用的文档。
# docs 只带这三篇，其余是内部实验记录，不随交付走。
mkdir -p "${RELEASE}/docs" "${RELEASE}/scripts"
cp "${REPO}/scripts/install_deps.sh" "${RELEASE}/scripts/install_deps.sh"
for d in run-tests-v2.md run-tests.md deps-bundle.md; do
    [[ -f "${REPO}/docs/${d}" ]] || { echo "[错误] 缺少文档：${REPO}/docs/${d}" >&2; exit 1; }
    cp "${REPO}/docs/${d}" "${RELEASE}/docs/${d}"
done
echo "[同步] 公共脚本和文档"

for c in "${COPIES[@]}"; do
    [[ -x "${REPO}/run/${c}.sh" ]] || { echo "[错误] 缺少验收入口：${REPO}/run/${c}.sh" >&2; exit 1; }
    mkdir -p "${RELEASE}/${c}"
    for p in "${PARTS[@]}"; do
        rm -rf "${RELEASE}/${c}/${p}"
        cp -R "${REPO}/${p}" "${RELEASE}/${c}/${p}"
    done
    rm -rf "${RELEASE}/${c}/tools/__pycache__"
    echo "[同步] ${c} 代码"
done

echo
echo "[完成] 交付目录已更新：${RELEASE}"
echo "        原始数据放入 ${RELEASE}/dataset"
echo "        在 NPU 服务器进入交付目录后运行 ./setup.sh"
