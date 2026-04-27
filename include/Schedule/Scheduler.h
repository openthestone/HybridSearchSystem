//全局查询任务管理队列和结果管理队列


#pragma once
#include <folly/MPMCQueue.h> // 引入 Folly 队列（仅用于 task_queue_）
#include <queue>
#include <mutex>
#include <semaphore.h>
#include "Query/Query.h"

class Scheduler
{
public:
    // 初始化队列容量
    // task_queue_ 用 Folly MPMCQueue（SPMC，生产者-多消费者）
    // result_queue_ 用 std::queue + semaphore（main thread 阻塞等待）
    Scheduler(size_t queue_capacity = 50000)
        : task_queue_(queue_capacity)
    {
        sem_init(&result_sem_, 0, 0);  // 匿名信号量，初始值 0
    }

    ~Scheduler() {
        sem_destroy(&result_sem_);
    }

    // --- 任务队列 (Main -> Worker) ---

    // 尝试写入任务（阻塞直到有空间）
    void Push(Query *q)
    {
        task_queue_.blockingWrite(q);
    }

    // 尝试获取任务（非阻塞）
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
    // MPMC：多个 worker leader 同时 PushResult
    // SPSC：单个 main thread PopResult（通过 semaphore 阻塞等待）

    void PushResult(Query *q)
    {
        {
            std::lock_guard<std::mutex> lk(result_mtx_);
            result_queue_.push(q);
        }
        sem_post(&result_sem_);  // 通知 main thread：结果已就绪
    }

    // 阻塞等待直到有结果返回（永返回有效指针）
    // 调用方无需检查 nullptr
    Query *PopResult()
    {
        sem_wait(&result_sem_);  // 阻塞，直到有 PushResult 唤醒
        std::lock_guard<std::mutex> lk(result_mtx_);
        Query *q = result_queue_.front();
        result_queue_.pop();
        return q;
    }

    // 获取估算的当前任务数量
    size_t GetTaskQueueSize()
    {
        return task_queue_.size();
    }

private:
    // 任务队列：Folly MPMCQueue（Main → Workers，SPMC）
    folly::MPMCQueue<Query *> task_queue_;

    // 结果队列：std::queue + mutex + semaphore（Workers → Main，MPMC）
    std::mutex result_mtx_;
    std::queue<Query *> result_queue_;
    sem_t result_sem_;  // 计数信号量，初始值 0
};
