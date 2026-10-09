#include "kernel_vector_mmad_op.h"

extern "C" __global__ __aicore__ void kernel_vector_mmad(GM_ADDR queryMatrix, GM_ADDR docMatrix, GM_ADDR resultMatrix,
                                                         uint16_t m, uint16_t mOut, uint32_t n, uint16_t k,
                                                         uint16_t b1N, uint16_t b2N, uint32_t zNSegLen,
                                                         uint32_t blockLength) {
    AscendC::TPipe pipe;
    KernelVectorMmadOp op;
    op.Init(queryMatrix, docMatrix, resultMatrix, m, mOut, n, k, zNSegLen, b1N, b2N, blockLength, &pipe);
    op.Process();
}
