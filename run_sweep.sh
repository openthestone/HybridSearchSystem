#!/usr/bin/env bash
#
# run_sweep.sh - 批量重复测试脚本
# 重复 5 次运行 run.sh + analyze_result.py，收集 Latency/Recall 均值，
# 列出每次结果，然后去掉最大值和最小值后计算二者的总平均值。
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="${SCRIPT_DIR}/result"

# 解析命令行参数：-c 编译模式，-v 变体
COMPILE_MODE="1"
VARIANT="s"
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c) COMPILE_MODE="$2"; shift 2 ;;
        -v) VARIANT="$2"; shift 2 ;;
        *)  echo "未知选项: $1"; exit 1 ;;
    esac
done

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}=============================================="
echo "大规模向量聚类与查询 - 批量测试脚本"
echo "测试次数: 5"
echo -e "==============================================${NC}"

declare -a latency_arr
declare -a recall_arr
tmp_out=$(mktemp)

count=0
for i in $(seq 1 5); do
    echo ""
    echo -e "${YELLOW}[${i}/5] 运行测试 ...${NC}"

    echo -e "${CYAN}  -> 执行 run.sh ...${NC}"
    if ! bash "${SCRIPT_DIR}/run.sh" -c "${COMPILE_MODE}" -v "${VARIANT}" > /dev/null 2>&1; then
        echo -e "${RED}  !! run.sh 执行失败，跳过本次${NC}"
        continue
    fi

    echo -e "${CYAN}  -> 分析结果 ...${NC}"
    if ! python3 "${SCRIPT_DIR}/analyze_result.py" > "${tmp_out}" 2>&1; then
        echo -e "${RED}  !! analyze_result.py 执行失败，跳过本次${NC}"
        continue
    fi

    # 从输出中提取两个 average 值
    # analyze_result.py 先输出 Latency 段（含 qps_by_avg_latency），再输出 Recall 段
    # 格式：  average: X.XXXX
    lat_avg=$(grep -E "^\s+average:" "${tmp_out}" | grep -v "qps_by_avg" | head -1 | awk '{print $2}')
    rec_avg=$(grep -E "^\s+average:" "${tmp_out}" | grep -v "qps_by_avg" | tail -1 | awk '{print $2}')

    if [[ -z "${lat_avg}" ]] || [[ -z "${rec_avg}" ]]; then
        echo -e "${RED}  !! 无法提取指标，跳过本次${NC}"
        continue
    fi

    latency_arr+=("${lat_avg}")
    recall_arr+=("${rec_avg}")
    ((count++)) || true

    echo -e "${GREEN}  [${i}/5] Latency avg: ${lat_avg} ms  |  Recall avg: ${rec_avg}%${NC}"
done

rm -f "${tmp_out}"

if [[ ${count} -eq 0 ]]; then
    echo -e "${RED}没有任何有效测试结果${NC}"
    exit 1
fi

echo ""
echo -e "${CYAN}=============================================="
echo "所有 ${count} 次测试结果"
echo -e "==============================================${NC}"
printf "${YELLOW}%-6s %-18s %-18s${NC}\n" "#" "Latency avg (ms)" "Recall avg (%)"
echo "----------------------------------------------"
for j in $(seq 0 $((count - 1))); do
    printf "%-6d %-18s %-18s\n" $((j + 1)) "${latency_arr[$j]}" "${recall_arr[$j]}"
done

echo ""
echo -e "${CYAN}=============================================="
echo "去掉最大值和最小值后的平均值"
echo -e "==============================================${NC}"

python3 - <<PYEOF
lat = [float(x) for x in """${latency_arr[@]}""".split()]
rec = [float(x) for x in """${recall_arr[@]}""".split()]
n = len(lat)

print(f"有效测试次数: {n}")
print()

raw_lat = sum(lat) / n
raw_rec = sum(rec) / n
print(f"{'Latency:':<12} 原始均值 = {raw_lat:.4f} ms  (min={min(lat):.4f}, max={max(lat):.4f})")
print(f"{'Recall:':<12} 原始均值 = {raw_rec:.4f} %   (min={min(rec):.4f}, max={max(rec):.4f})")

# 去掉头尾
if n > 2:
    lat_t = sorted(lat)[1:-1]
    rec_t = sorted(rec)[1:-1]
else:
    lat_t = sorted(lat)
    rec_t = sorted(rec)

lat_mean = sum(lat_t) / len(lat_t)
rec_mean = sum(rec_t) / len(rec_t)

print()
print(f"{'Latency:':<12} Trimmed 均值 ({n-2 if n>2 else n} 次) = {lat_mean:.4f} ms")
print(f"{'Recall:':<12} Trimmed 均值 ({n-2 if n>2 else n} 次) = {rec_mean:.4f} %")
PYEOF

echo ""
echo -e "${GREEN}批量测试完成${NC}"
