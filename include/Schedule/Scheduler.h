//全局查询任务管理队列和结果管理队列


#pragma once
#include <folly/MPMCQueue.h> // 引入 Folly 队列
#include "Query/Query.h"

class Scheduler
{
public:
    // 初始化队列容量
    // 默认容量覆盖常见批量查询场景；可按需显式覆盖。
    Scheduler(size_t queue_capacity = 50000)
        : task_queue_(queue_capacity), 
          result_queue_(queue_capacity) 
    {}

    // --- 任务队列 (Main -> Worker) ---

    // 尝试写入任务
    // 如果队列满了，根据你的业务逻辑，可以选择：
    // 1. blockingWrite(q) -> 阻塞直到有空间 (推荐用于生产者-消费者模型)
    // 2. write(q) -> 返回 false，丢弃或稍后重试
    void Push(Query *q)
    {
        task_queue_.blockingWrite(q);
    }

    // 尝试获取任务 (非阻塞)
    // 成功返回指针，失败(空)返回 nullptr
    Query *Pop()
    {
        Query *q = nullptr;
        if (task_queue_.read(q)) {
            return q;
        }
        return nullptr;
    }

    // --- 结果队列 (Worker -> Main) ---

    void PushResult(Query *q)
    {
        result_queue_.blockingWrite(q);
    }

    Query *PopResult()
    {
        Query *q = nullptr;
        if (result_queue_.read(q)) {
            return q;
        }
        return nullptr;
    }

    // 获取估算的当前任务数量
    size_t GetTaskQueueSize()
    {
        return task_queue_.size();
    }

private:
    // 移除 std::mutex，MPMCQueue 内部实现了无锁(基于原子操作)或细粒度锁机制
    folly::MPMCQueue<Query *> task_queue_;
    folly::MPMCQueue<Query *> result_queue_;
};
