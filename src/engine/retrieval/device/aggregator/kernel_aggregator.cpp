#include "kernel_operator.h"
#include "kernel_aggregator.h"

extern "C" __global__ __aicore__ void Aggregator(GM_ADDR filterResult, GM_ADDR docVectorScores,
                                                 GM_ADDR docLocationInDeviceConst, GM_ADDR docLocationInDeviceResult,
                                                 GM_ADDR effectiveCountDevice, uint32_t docNumber, uint32_t blockNumber,
                                                 uint32_t blockLength, uint32_t lastBlockLength) {
    AscendC::TPipe pipe;
    KernelAggregator aggregator;
    aggregator.Init(&pipe, filterResult, docVectorScores, docLocationInDeviceConst, docLocationInDeviceResult,
                    effectiveCountDevice, docNumber, blockNumber, blockLength, lastBlockLength);
    aggregator.Process();
}
