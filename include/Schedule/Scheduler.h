//全局查询任务管理队列和结果管理队列


#pragma once
#include <folly/MPMCQueue.h>
#include "Query/Query.h"

class Scheduler
{
public:
    Scheduler(size_t queue_capacity = 50000)
        : task_queue_(queue_capacity)
        , result_queue_(queue_capacity)
    {
    }

    // --- 任务队列 (Main -> Worker) ---

    void Push(Query *q)
    {
        task_queue_.blockingWrite(q);
    }

    Query *Pop()
    {
        Query *q = nullptr;
        if (task_queue_.read(q)) {
            return q;
        }
        return nullptr;
    }

    // --- 结果队列 (Worker -> Main) ---
    // MPMC：多个 worker leader 同时 PushResult
    // SPSC：单个 main thread PopResult（blockingRead 阻塞等待）

    void PushResult(Query *q)
    {
        result_queue_.blockingWrite(q);
    }

    Query *PopResult()
    {
        Query *q = nullptr;
        result_queue_.blockingRead(q);
        return q;
    }

    size_t GetTaskQueueSize()
    {
        return task_queue_.size();
    }

private:
    folly::MPMCQueue<Query *> task_queue_;
    folly::MPMCQueue<Query *> result_queue_;
};
