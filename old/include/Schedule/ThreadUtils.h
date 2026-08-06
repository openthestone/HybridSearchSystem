#pragma once
#include <pthread.h>
#include <sched.h>
#include <iostream>
#include <vector>

// NUMA-aware physical core remapping.
// When populated, logical_core_id is translated to the physical core at
// g_numa_aware_cores[logical_core_id] before binding.
// This allows the group/rank math (GetGroupId, IsLeader, …) to keep using
// a dense 0-based logical index while the actual CPU pinning targets the
// NUMA-local cores of each NPU device.
inline std::vector<int> g_numa_aware_cores;

// 函数：将当前线程绑定到指定的物理核心
// core_id 是逻辑索引 (0, 1, …, cpu_core_count-1)；
// 如果 g_numa_aware_cores 非空，则查找对应的物理核编号。
inline void BindThreadToCore(int core_id) {
// 仅在 Linux 环境下生效（因为使用了 pthread_setaffinity_np）
#if defined(__linux__)
    int physical_core = core_id;
    if (!g_numa_aware_cores.empty() && core_id >= 0 && static_cast<size_t>(core_id) < g_numa_aware_cores.size()) {
        physical_core = g_numa_aware_cores[core_id];
    }

    // 1. 定义一个 CPU 集合掩码
    cpu_set_t cpuset;

    // 2. 清空集合
    CPU_ZERO(&cpuset);

    // 3. 将目标 physical_core 加入集合
    CPU_SET(physical_core, &cpuset);

    // 4. 调用系统 API 设置亲和性
    int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    // 错误检查
    if (rc != 0) {
        std::cerr << "Error calling pthread_setaffinity_np (core_id=" << core_id << " -> physical=" << physical_core
                  << "): " << rc << "\n";
    }
#else
    // 非 Linux 系统（如 Windows 开发环境）不做任何操作，防止编译报错
    (void)core_id;
#endif
}