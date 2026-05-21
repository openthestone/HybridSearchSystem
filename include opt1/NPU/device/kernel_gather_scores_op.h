#pragma once
#include "kernel_operator.h"
#include "device_common.h"

// Gather task descriptor: one per L2 bucket to extract
// Must match the host-side definition in npuAPI.cpp
#ifndef __HOST_GATHER_TASK_DATA__
#define __HOST_GATHER_TASK_DATA__
struct GatherTaskData
{
    uint64_t src_offset; // byte offset in source buffer (dev_l0_cache)
    uint64_t dst_offset; // byte offset in destination buffer (d_result_ws), 32B-aligned
    uint32_t count;      // number of float elements to copy (actual valid count)
    uint32_t count_aligned; // count rounded up to kGatherAlignFloats (for DataCopy)
};
#endif

// AscendC DataCopy for float32 requires count to be a multiple of 8 (32 bytes)
constexpr uint32_t kGatherAlignFloats = 8;
constexpr uint32_t kGatherChunkFloats = 2048;

class KernelGatherScoresOp
{
public:
    __aicore__ inline KernelGatherScoresOp() {}

    __aicore__ inline void Init(GM_ADDR srcBuffer, GM_ADDR dstBuffer,
                                GM_ADDR taskBuffer, uint32_t totalTasks)
    {
        srcGM.SetGlobalBuffer((__gm__ float *)srcBuffer);
        dstGM.SetGlobalBuffer((__gm__ float *)dstBuffer);

        uint32_t taskId = AscendC::GetBlockIdx();
        if (taskId >= totalTasks)
        {
            valid_ = false;
            return;
        }
        valid_ = true;

        __gm__ GatherTaskData *taskPtr = (__gm__ GatherTaskData *)taskBuffer;
        __gm__ GatherTaskData *task = &taskPtr[taskId];

        srcOffset_ = task->src_offset / sizeof(float);
        dstOffset_ = task->dst_offset / sizeof(float);
        count_ = task->count;
        countAligned_ = task->count_aligned;
    }

    __aicore__ inline void Process()
    {
        if (!valid_ || count_ == 0)
            return;

        pipe.InitBuffer(copyQueue_, 1, kGatherChunkFloats * sizeof(float));

        uint32_t remaining = countAligned_;
        uint32_t srcIdx = srcOffset_;
        uint32_t dstIdx = dstOffset_;

        while (remaining > 0)
        {
            uint32_t thisChunk = (remaining > kGatherChunkFloats) ? kGatherChunkFloats : remaining;

            AscendC::LocalTensor<float> localBuf = copyQueue_.AllocTensor<float>();

            AscendC::DataCopy(localBuf, srcGM[srcIdx], thisChunk);
            copyQueue_.EnQue(localBuf);

            AscendC::LocalTensor<float> outBuf = copyQueue_.DeQue<float>();
            AscendC::DataCopy(dstGM[dstIdx], outBuf, thisChunk);
            copyQueue_.FreeTensor(outBuf);

            srcIdx += thisChunk;
            dstIdx += thisChunk;
            remaining -= thisChunk;
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> copyQueue_;

    AscendC::GlobalTensor<float> srcGM;
    AscendC::GlobalTensor<float> dstGM;

    uint32_t srcOffset_ = 0;
    uint32_t dstOffset_ = 0;
    uint32_t count_ = 0;
    uint32_t countAligned_ = 0;
    bool valid_ = false;
};
