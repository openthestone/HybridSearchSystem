#!/usr/bin/env python3
"""
analyze_result_p99.py

分析时延超出 P99 的查询的各阶段耗时详情。
从 result/latency 目录读取全部指标文件，自动识别列名与数值。
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

# ── 路径配置 ─────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULT_DIR = ROOT_DIR / "result"
LATENCY_DIR = DEFAULT_RESULT_DIR / "latency"
LOG_DIR = DEFAULT_RESULT_DIR / "log"

# latency/ 下的指标文件，每个文件每行一个 float，对应同一条查询
LATENCY_METRICS = [
    ("bucket_level_ivf_ms", "IVF桶选择"),
    ("bucket_level_ivf_filter_eval_ms", "IVF-属性过滤"),
    ("bucket_level_ivf_centroid_score_ms", "IVF-中心打分"),
    ("bucket_level_ivf_result_pack_ms", "IVF-结果打包"),
    ("candidate_bucket_merge_ms", "候选桶合并"),
    ("npu_async_launch_ms", "NPU异步下发"),
    ("npu_submit_ms", "NPU提交(H2D+launch)"),
    ("npu_kernel_exec_ms", "NPU kernel执行"),
    ("npu_d2d_gather_ms", "D2D gather"),
    ("npu_d2h_transfer_ms", "D2H传输"),
    ("npu_sync_overhead_ms", "同步等待开销"),
    ("inbucket_attr_filter_overlapped_ms", "属性过滤(与NPU并行)"),
    ("wait_npu_flag_ms", "等待NPU完成"),
    ("result_collection_ms", "结果收集"),
    ("final_merge_ms", "最终合并"),
    ("total_end_to_end_ms", "总耗时"),
]


# ── CJK-aware formatting helpers ─────────────────────────
def _dw(s: str) -> int:
    """Display width: CJK ideographs and fullwidth chars count as 2."""
    w = 0
    for ch in s:
        cp = ord(ch)
        if (0x4E00 <= cp <= 0x9FFF or 0x3000 <= cp <= 0x303F or
                0xFF00 <= cp <= 0xFFEF or 0xF900 <= cp <= 0xFAFF):
            w += 2
        else:
            w += 1
    return w


def _rpad(s: str, width: int) -> str:
    """Right-pad string to *display* width."""
    return s + ' ' * max(0, width - _dw(s))


def _lpad(s: str, width: int) -> str:
    """Left-pad string to *display* width."""
    return ' ' * max(0, width - _dw(s)) + s


def read_float_column(path: Path) -> list[float]:
    values: list[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                values.append(float(text))
            except ValueError:
                continue
    return values


def read_process_round_count(path: Path) -> tuple[list[int], list[int]]:
    l1_counts: list[int] = []
    l2_counts: list[int] = []
    with path.open("r", encoding="utf-8") as f:
        header = True
        for line in f:
            if header:
                header = False
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                l1_counts.append(int(parts[0]))
                l2_counts.append(int(parts[1]))
    return l1_counts, l2_counts


def percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = f + 1
    if c >= len(s):
        return s[-1]
    return s[f] + (k - f) * (s[c] - s[f])


def fmt_ms(v: float) -> str:
    return f"{v:.4f}"


def main():
    # ── 1. 加载全部指标 ─────────────────────────────────
    data: dict[str, list[float]] = {}
    for key, _ in LATENCY_METRICS:
        path = LATENCY_DIR / f"{key}.txt"
        if not path.is_file():
            print(f"[Skip] {path} not found")
            continue
        data[key] = read_float_column(path)

    if "total_end_to_end_ms" not in data:
        print("[Error] total_end_to_end_ms.txt is required", file=sys.stderr)
        sys.exit(1)

    total = data["total_end_to_end_ms"]
    n = len(total)

    # 校验所有列长度一致
    for key, vals in data.items():
        if len(vals) != n:
            print(f"[Warn] {key} has {len(vals)} rows, expected {n}. Skipping.", file=sys.stderr)
            data[key] = [0.0] * n

    # 加载 process_round_count（可选）
    round_path = LOG_DIR / "process_round_count.txt"
    round_l1: list[int] = []
    round_l2: list[int] = []
    if round_path.is_file():
        round_l1, round_l2 = read_process_round_count(round_path)
        if len(round_l1) != n:
            print(f"[Warn] process_round_count has {len(round_l1)} rows, expected {n}.", file=sys.stderr)
            round_l1 = []
            round_l2 = []

    # ── 2. 计算全局百分位 ───────────────────────────────
    print("=" * 72)
    print("  全局时延分布 (total_end_to_end_ms)")
    print("=" * 72)
    print(f"  查询总数       : {n}")
    for p in [50, 90, 95, 99, 99.5, 99.9, 100]:
        print(f"  P{p:<6}        : {fmt_ms(percentile(total, p))} ms")
    print(f"  平均值         : {fmt_ms(statistics.mean(total))} ms")
    print(f"  标准差         : {fmt_ms(statistics.stdev(total))} ms")

    # ── 3. 筛选 P99+ 查询 ───────────────────────────────
    p99_val = percentile(total, 99)
    tail_indices = [i for i, v in enumerate(total) if v >= p99_val]
    tail_count = len(tail_indices)

    if tail_count == 0:
        print("\n[Info] 没有查询时延达到或超过 P99，无需分析。")
        return

    print()
    print("=" * 72)
    print(f"  P99+ 查询分析  (阈值: {fmt_ms(p99_val)} ms, 共 {tail_count} 条)")
    print("=" * 72)

    # ── 4. P99+ 各阶段统计 ──────────────────────────────
    stage_keys = [k for k, _ in LATENCY_METRICS if k != "total_end_to_end_ms" and k in data]
    stage_names = {k: name for k, name in LATENCY_METRICS}

    NUM_W = 12
    RATIO_W = 8
    LABEL_W = max(_dw(name) for name in stage_names.values()) + 2
    LABEL_W = max(LABEL_W, _dw("总耗时") + 2, _dw("阶段") + 2)
    SEP1_W = LABEL_W + 2 + NUM_W + 2 + NUM_W + 2 + NUM_W + 2 + RATIO_W

    print()
    print(f"  {_rpad('阶段', LABEL_W)}  {_lpad('平均(ms)', NUM_W)}  {_lpad('中位数(ms)', NUM_W)}  {_lpad('最大(ms)', NUM_W)}  {_lpad('占比', RATIO_W)}")
    print("  " + "-" * SEP1_W)

    tail_total = [total[i] for i in tail_indices]
    tail_total_mean = statistics.mean(tail_total)

    for key in stage_keys:
        vals = [data[key][i] for i in tail_indices]
        mean_v = statistics.mean(vals)
        median_v = statistics.median(vals)
        max_v = max(vals)
        ratio = (mean_v / tail_total_mean * 100) if tail_total_mean > 0 else 0
        label = stage_names.get(key, key)
        print(f"  {_rpad(label, LABEL_W)}  {_lpad(fmt_ms(mean_v), NUM_W)}  {_lpad(fmt_ms(median_v), NUM_W)}  {_lpad(fmt_ms(max_v), NUM_W)}  {_lpad(f'{ratio:.2f}%', RATIO_W)}")

    # 总耗时行
    tail_mean = statistics.mean(tail_total)
    tail_median = statistics.median(tail_total)
    tail_max = max(tail_total)
    tail_min = min(tail_total)
    print("  " + "-" * SEP1_W)
    print(f"  {_rpad('总耗时', LABEL_W)}  {_lpad(fmt_ms(tail_mean), NUM_W)}  {_lpad(fmt_ms(tail_median), NUM_W)}  {_lpad(fmt_ms(tail_max), NUM_W)}  {_lpad('100.00%', RATIO_W)}")

    # ── 5. P99+ vs 全体对比 ─────────────────────────────
    print()
    print("=" * 72)
    print("  P99+ vs 全体对比")
    print("=" * 72)

    all_total_mean = statistics.mean(total)
    all_total_median = statistics.median(total)

    T2_LABELS = ["总耗时-平均(ms)", "总耗时-中位数(ms)", "总耗时-最大(ms)"] + list(stage_names.values())
    T2_LABEL_W = max(_dw(s) for s in T2_LABELS) + 2
    T2_SEP_W = T2_LABEL_W + 2 + NUM_W + 2 + NUM_W + 2 + RATIO_W

    print(f"  {_rpad('指标', T2_LABEL_W)}  {_lpad('全体', NUM_W)}  {_lpad('P99+', NUM_W)}  {_lpad('倍率', RATIO_W)}")
    print("  " + "-" * T2_SEP_W)
    print(f"  {_rpad('总耗时-平均(ms)', T2_LABEL_W)}  {_lpad(fmt_ms(all_total_mean), NUM_W)}  {_lpad(fmt_ms(tail_mean), NUM_W)}  {_lpad(f'{tail_mean/all_total_mean:.2f}x', RATIO_W)}")
    print(f"  {_rpad('总耗时-中位数(ms)', T2_LABEL_W)}  {_lpad(fmt_ms(all_total_median), NUM_W)}  {_lpad(fmt_ms(tail_median), NUM_W)}  {_lpad(f'{tail_median/all_total_median:.2f}x', RATIO_W)}")
    print(f"  {_rpad('总耗时-最大(ms)', T2_LABEL_W)}  {_lpad(fmt_ms(max(total)), NUM_W)}  {_lpad(fmt_ms(tail_max), NUM_W)}  {_lpad('—', RATIO_W)}")

    for key in stage_keys:
        all_vals = data[key]
        all_mean = statistics.mean(all_vals)
        tail_vals = [data[key][i] for i in tail_indices]
        t_mean = statistics.mean(tail_vals)
        ratio = t_mean / all_mean if all_mean > 0 else 0
        label = stage_names.get(key, key)
        print(f"  {_rpad(label, T2_LABEL_W)}  {_lpad(fmt_ms(all_mean), NUM_W)}  {_lpad(fmt_ms(t_mean), NUM_W)}  {_lpad(f'{ratio:.2f}x', RATIO_W)}")

    # ── 6. 扩搜轮次分布（如有） ─────────────────────────
    if round_l1:
        print()
        print("=" * 72)
        print("  P99+ 扩搜轮次分布")
        print("=" * 72)

        tail_l1 = [round_l1[i] for i in tail_indices]
        tail_l2 = [round_l2[i] for i in tail_indices]

        from collections import Counter
        l1_dist = Counter(tail_l1)
        l2_dist = Counter(tail_l2)

        print("  L1 轮次分布: ", end="")
        for k in sorted(l1_dist.keys()):
            print(f"{k}轮={l1_dist[k]}条({l1_dist[k]/tail_count*100:.1f}%)  ", end="")
        print()

        print("  L2 批次分布: ", end="")
        for k in sorted(l2_dist.keys()):
            print(f"{k}批={l2_dist[k]}条({l2_dist[k]/tail_count*100:.1f}%)  ", end="")
        print()

        all_l1_mean = statistics.mean(round_l1)
        tail_l1_mean = statistics.mean(tail_l1)
        all_l2_mean = statistics.mean(round_l2)
        tail_l2_mean = statistics.mean(tail_l2)
        print(f"  L1 平均轮次: 全体={all_l1_mean:.2f}  P99+={tail_l1_mean:.2f}  ({tail_l1_mean/all_l1_mean:.2f}x)")
        print(f"  L2 平均批次: 全体={all_l2_mean:.2f}  P99+={tail_l2_mean:.2f}  ({tail_l2_mean/all_l2_mean:.2f}x)")

    # ── 7. P99+ 内部时延分布细分 ────────────────────────
    print()
    print("=" * 72)
    print("  P99+ 内部时延分布细分")
    print("=" * 72)

    PCT_W = 10
    TOT_W = 14
    STAGE_W = 10
    T3_SEP_W = PCT_W + 2 + TOT_W + len(stage_keys) * (2 + STAGE_W)

    print(f"  {_rpad('百分位', PCT_W)}  {_lpad('总耗时(ms)', TOT_W)}", end="")
    for key in stage_keys:
        label = stage_names.get(key, key)
        short = label[:6]
        print(f"  {_lpad(short, STAGE_W)}", end="")
    print()
    print("  " + "-" * T3_SEP_W)

    for p in [0, 25, 50, 75, 90, 100]:
        rank_idx = min(int(len(tail_indices) * p / 100.0), tail_count - 1)
        sorted_tail = sorted(tail_indices, key=lambda i: total[i])
        qi = sorted_tail[rank_idx]

        label = f"P99+P{p}" if p > 0 else "MIN"
        print(f"  {_rpad(label, PCT_W)}  {_lpad(fmt_ms(total[qi]), TOT_W)}", end="")
        for key in stage_keys:
            print(f"  {_lpad(fmt_ms(data[key][qi]), STAGE_W)}", end="")
        print()

    print()
    print(f"  注: P99+ 集合共 {tail_count} 条查询，占全体 {tail_count/n*100:.1f}%")


if __name__ == "__main__":
    main()
