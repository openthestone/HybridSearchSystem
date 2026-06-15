#include "kernel_vector_mmad_op.h"

extern "C" __global__ __aicore__ void kernel_vector_mmad(GM_ADDR queryMatrix, GM_ADDR taskBuffer,
                                                         GM_ADDR baseDocAddr, GM_ADDR baseResultAddr, uint32_t k)
{
    KernelVectorMmadOp op;

    // MatrixB (Query) -> queryMatrix
    // MatrixA (Docs)  -> baseDocAddr
    // MatrixC (Res)   -> baseResultAddr
    op.Init(queryMatrix, taskBuffer, baseDocAddr, baseResultAddr, k);
    op.Process();
}
