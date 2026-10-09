#include "kernel_operator.h"
#include "kernel_topk.h"

extern "C" __global__ __aicore__ void TOPK(GM_ADDR docVectors, GM_ADDR docLocationVectors, GM_ADDR docVectorsNext,
                                           GM_ADDR docLocationVectorsNext, GM_ADDR docResultCountInDevice,
                                           GM_ADDR docNumber, uint32_t loopCount, uint32_t blockDimAggregator,
                                           uint32_t blockNumberShardAggregator, uint32_t topK, uint32_t topKN,
                                           uint32_t docNumberStride) {
    KernelTOPK topk;
    topk.Init(docVectors, docLocationVectors, docVectorsNext, docLocationVectorsNext, docResultCountInDevice, docNumber,
              loopCount, blockDimAggregator, blockNumberShardAggregator, topK, topKN, docNumberStride);
    topk.Process();
}
