#include "../device_common.h"
#include "filter_and_op.h"
#include "filter_or_op.h"
#include "filter_not_op.h"
#include "filter_conjunction_op.h"
#include "bitlist_to_set.h"

/*
 * postings: pointer to an array of pointers; each points to a bitlist to process.
 * byteSizePerSegment: byte length after bitlist->bitset conversion; the caller must ensure it is a multiple of 32.
 * maxPostingLength: theoretical max bitlist posting length, controlled by the offline side.
 * resultPostings: pointer to an array of pointers; each points to where a converted bitset is stored.
 * formerNum: number of "full" cores (those assigned the larger data amount).
 * tailNum: number of "tail" cores (those assigned the smaller data amount).
 * formerLength: data amount computed by a full core.
 * tailLength: data amount computed by a tail core.
 */
extern "C" __global__ __aicore__ void PostingBitListToSet(GM_ADDR postings, uint32_t byteSizePerSegment,
                                                          uint32_t maxPostingLength, GM_ADDR resultPostings,
                                                          uint32_t formerNum, uint32_t formerLength, uint32_t tailNum,
                                                          uint32_t tailLength, uint32_t clearSparse, uint32_t ablate,
                                                          uint32_t useScatter) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);  // set the default kernel type to pure AIV
    int64_t blockIdx = AscendC::GetBlockIdx();
    int64_t blockNum = AscendC::GetBlockNum();
    DEBUG_LOG("======blockIdx%d, blockNum:%d\n", blockIdx, blockNum);
    uint32_t beginIdx = 0;
    uint32_t endIdx = 0;
    if (blockIdx < formerNum) {
        // full core
        beginIdx = blockIdx * formerLength;
        endIdx = beginIdx + formerLength;
    } else {
        // tail core
        beginIdx = formerNum * formerLength + tailLength * (blockIdx - formerNum);
        endIdx = beginIdx + tailLength;
    }
    AscendC::TPipe pipe;
    BitlistToSet converter;
    converter.Init(&pipe, postings, byteSizePerSegment, maxPostingLength, resultPostings, beginIdx, endIdx, clearSparse,
                   ablate, useScatter);
    converter.Process();
}

/*
 * postings: pointer to an array of pointers; each points to a token's per-segment bitset.
 * postOrderExpr: pointer to the post-order operator-expression array (uint32_t elements).
 * resultStack: start address of the intermediate result stack, opNum*segmentLength*segmentNum uint16_t values.
 * context: parameters needed by the filter computation (see the struct).
 * result: pointer to the start of the filter output.
 */
extern "C" __global__ __aicore__ void BitmapTextFilter(GM_ADDR postings, GM_ADDR postOrderExpr, GM_ADDR resultStack,
                                                       TextFilterContextData context, GM_ADDR result) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);  // set the default kernel type to pure AIV
    int64_t blockIdx = AscendC::GetBlockIdx();
    int64_t blockNum = AscendC::GetBlockNum();
    uint32_t segIdx = blockIdx;
    const uint32_t& postingNum = context.postingNum;
    const uint32_t& segmentNum = context.segmentNum;
    const uint32_t& segmentLength = context.segmentLength;
    const uint32_t& tileNum = context.tileNum;
    const uint32_t& exprLen = context.exprLen;
    const uint32_t& opNum = context.opNum;
    const uint32_t& filterFlags = context.flags;
    uint32_t segmentPostingSize = postingNum * sizeof(uint16_t*);
    uint32_t maxResultStackSize = opNum * segmentLength;  // number of u16 elements
    AscendC::GlobalTensor<uint32_t> postOrderExprGm;
    postOrderExprGm.SetGlobalBuffer((__gm__ uint32_t*)postOrderExpr, exprLen);

    while (segIdx < segmentNum) {
        // NOTE: the stored values are actually uint16_t*; convert when using.
        // __gm__ uint8_t* cannot be cast to __gm__ uint16_t**.
        AscendC::GlobalTensor<uint64_t> postingsGm;
        postingsGm.SetGlobalBuffer((__gm__ uint64_t*)postings + postingNum * segIdx, postingNum);
        uint64_t postingsGmOffset = 0;  // offset index of the next posting to use
        uint32_t stackIdx = 0;
        AscendC::GlobalTensor<uint16_t> curResultStack;
        curResultStack.SetGlobalBuffer((__gm__ uint16_t*)resultStack + maxResultStackSize * segIdx, maxResultStackSize);
        DEBUG_LOG("===curResultStack size:%d\n", curResultStack.GetSize());
        AscendC::GlobalTensor<uint16_t> resultGm;
        resultGm.SetGlobalBuffer((__gm__ uint16_t*)result + segmentLength * segIdx, segmentLength);
        uint32_t opCnt = 0;
        for (uint32_t exprIdx = 0; exprIdx < exprLen; exprIdx++) {
            uint32_t opType = postOrderExprGm.GetValue(exprIdx);
            // whether this is the last op; if so, the result is copied directly to result instead of the stack
            bool isfinalOp = (opCnt == (opNum - 1));
            AscendC::TPipe pipe;
            if (opType == FilterOpType::CONJUNCTION) {
                FilterConjunctionOp conjunctionOp;
                exprIdx +=
                    conjunctionOp.Init(&pipe, postingsGm, postingsGmOffset, segmentLength, postOrderExprGm[exprIdx + 1],
                                       tileNum, curResultStack, stackIdx, isfinalOp, resultGm);
                conjunctionOp.Process();
            } else if (opType == FilterOpType::AND) {
                FilterAndOp andOp;
                exprIdx += andOp.Init(&pipe, postingsGm, postingsGmOffset, segmentLength, postOrderExprGm[exprIdx + 1],
                                      tileNum, curResultStack, stackIdx, isfinalOp, resultGm);
                andOp.Process();
            } else if (opType == FilterOpType::OR) {
                FilterOrOp orOp;
                exprIdx += orOp.Init(&pipe, postingsGm, postingsGmOffset, segmentLength, postOrderExprGm[exprIdx + 1],
                                     tileNum, curResultStack, stackIdx, isfinalOp, resultGm, filterFlags);
                orOp.Process();
            } else if (opType == FilterOpType::NOT) {
                FilterNotOp notOp;
                exprIdx += notOp.Init(&pipe, postingsGm, postingsGmOffset, segmentLength, postOrderExprGm[exprIdx + 1],
                                      tileNum, curResultStack, stackIdx, isfinalOp, resultGm);
                notOp.Process();
            } else {
                ERROR_LOG("[ERROR] invalid opType:%d", opType);
                break;
            }
            opCnt += 1;
        }
        DEBUG_LOG("[INFO] after Filter, res: seg[%d]:\n", segIdx);
        DEBUG_DUMP_TENSOR(resultGm, 999, segmentLength);  // 999 is the log dump marker
        segIdx += blockNum;
    }
}
