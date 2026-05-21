#include "kernel_gather_scores_op.h"

extern "C" __global__ __aicore__ void kernel_gather_scores(GM_ADDR srcBuffer, GM_ADDR dstBuffer,
                                                            GM_ADDR taskBuffer, uint32_t totalTasks)
{
    KernelGatherScoresOp op;
    op.Init(srcBuffer, dstBuffer, taskBuffer, totalTasks);
    op.Process();
}
