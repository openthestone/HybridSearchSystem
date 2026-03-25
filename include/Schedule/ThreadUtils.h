#pragma once
#include <pthread.h>
#include <sched.h>
#include <iostream>

// 函数：将当前线程绑定到指定的物理核心 core_id
inline void BindThreadToCore(int core_id)
{
// 仅在 Linux 环境下生效（因为使用了 pthread_setaffinity_np）
#if defined(__linux__)

    // 1. 定义一个 CPU 集合掩码
    cpu_set_t cpuset;

    // 2. 清空集合
    CPU_ZERO(&cpuset);

    // 3. 将目标 core_id 加入集合
    // 这意味着告诉 OS：这个线程只能用 core_id 这个核
    CPU_SET(core_id, &cpuset);

    // 4. 调用系统 API 设置亲和性
    // pthread_self(): 获取当前线程句柄
    // sizeof(cpu_set_t): 集合大小
    // &cpuset: 具体的绑定规则
    int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    // 错误检查
    if (rc != 0)
    {
        std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
    }
#else
    // 非 Linux 系统（如 Windows 开发环境）不做任何操作，防止编译报错
    (void)core_id;
#endif
}