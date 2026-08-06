#pragma once

#include <cstdint>
#include <vector>
#include "device_common_external.h"
#include "kernel_operator.h"

enum FilterOpType : uint32_t {
    CONJUNCTION = 0,
    AND = 1,
    OR = 2,
    NOT = 3,
};

const uint8_t BUFFER_NUM = 2;  // number of buffer blocks to allocate
const uint32_t HEADER_BYTE_SIZE = 8;
const uint32_t DUMP_DEFAULT_LENGTH = 64;
constexpr uint32_t BLOCK_SIZE = 32;
const uint16_t MAX_UINT16_VALUE = 65535;

const uint32_t FILTER_AND_PARM_NUM = 2;
const uint32_t FILTER_OR_PARM_NUM = 2;
const uint32_t FILTER_NOT_PARM_NUM = 2;

// NOTE: dstLocal and srcGlobal start addresses must be 32-byte aligned; use the official DataCopyPad otherwise
template <typename typeT>
__aicore__ inline void DataCopyPadCustom_GM2UB(const AscendC::LocalTensor<typeT>& dstLocal,
                                               const AscendC::GlobalTensor<typeT>& srcGlobal, const uint32_t calCount) {
    if (calCount < BLOCK_SIZE / sizeof(typeT)) {  // less than 32B: assign directly
        for (uint32_t i = 0; i < calCount; i++) {
            dstLocal.SetValue(i, srcGlobal.GetValue(i));
        }
    } else {  // more than 32B: copy the 32B-multiple part first, then assign the unaligned remainder
        uint32_t padDataCount = calCount - (calCount % (BLOCK_SIZE / sizeof(typeT)));
        AscendC::DataCopy(dstLocal, srcGlobal, padDataCount);
        for (uint32_t i = 0; i < (calCount % (BLOCK_SIZE / sizeof(typeT))); i++) {
            dstLocal[padDataCount].SetValue(i, srcGlobal[padDataCount].GetValue(i));
        }
    }
}

// NOTE: dstLocal and srcGlobal start addresses must be 32-byte aligned; use the official DataCopyPad otherwise
template <typename typeT>
__aicore__ inline void DataCopyPadCustom_UB2GM(const AscendC::GlobalTensor<typeT>& dstGlobal,
                                               const AscendC::LocalTensor<typeT>& srcLocal, const uint32_t calCount) {
    if (calCount < BLOCK_SIZE / sizeof(typeT)) {
        for (uint32_t i = 0; i < calCount; i++) {
            typeT localValue = srcLocal.GetValue(i);
            auto cursor = dstGlobal.address_ + i;
            *cursor = localValue;
        }
    } else {
        uint32_t padDataCount = calCount - (calCount % (BLOCK_SIZE / sizeof(typeT)));
        AscendC::DataCopy(dstGlobal, srcLocal, padDataCount);
        for (uint32_t i = 0; i < (calCount % (BLOCK_SIZE / sizeof(typeT))); i++) {
            typeT localValue = srcLocal[padDataCount].GetValue(i);
            auto cursor = dstGlobal[padDataCount].address_ + i;
            *cursor = localValue;
        }
    }
}

// log-level print switches
#define DEBUG_LOG_ENABLE false
#define ERROR_LOG_ENABLE false
#define DUMP_TENSOR_ENABLE false

// The extra macro nesting here is to fully disable log printing. By design, as long
// as an AscendC::printf call exists in the code, the kernel still prints kernel info
// when it runs through the function containing that AscendC::printf -- even if the
// surrounding condition makes the printf itself unreachable -- e.g.:
// opType=PostingBitListToSet, DumpHead: AIC-0, CoreType=MIX, block dim=8, total_block_num=8, block_remain_len=1048456,
// block_initial_space=1048576, rsv=0, magic=5aa5bccd
// CANN Version: 8.0.0, TimeStamp: 20241231000821303
// So when logging is disabled we use macros to remove the AscendC::printf calls entirely.
#if DEBUG_LOG_ENABLE == true
#define DEBUG_LOG(fmt, args...) AscendC::printf(fmt, ##args)
#else
#define DEBUG_LOG(fmt, args...)
#endif

#if DUMP_TENSOR_ENABLE == true
#define DEBUG_DUMP_TENSOR(tensor, desc, dumpSize) AscendC::DumpTensor(tensor, desc, dumpSize)
#else
#define DEBUG_DUMP_TENSOR(tensor, desc, dumpSize)
#endif

#if ERROR_LOG_ENABLE == true
#define ERROR_LOG(fmt, args...) AscendC::printf(fmt, ##args)
#else
#define ERROR_LOG(fmt, args...)
#endif
