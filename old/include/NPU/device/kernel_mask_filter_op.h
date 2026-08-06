#pragma once
#include "kernel_operator.h"
#include "device_common.h"

// Per-bucket task descriptor for mask filter kernel
// Must match host-side definition in npuAPI.cpp
struct MaskFilterTask {
    uint64_t scores_offset;  // byte offset into d_result_ws (Fixpipe output)
    uint64_t mask_offset;    // byte offset into mask buffer
    uint64_t output_offset;  // byte offset into scores output buffer
    uint64_t index_offset;   // byte offset into local-index output buffer
    uint32_t doc_num;        // number of documents in this bucket
    uint32_t mask_words;     // ceil(doc_num / 64)
};

// Single-block kernel that loops over all buckets internally.
// Task buffer layout: [uint64_t task_count][MaskFilterTask × N]
// All GlobalTensors bound to full buffer ranges in Init() — no raw GM_ADDR members.

class KernelMaskFilterOp {
   public:
    __aicore__ inline KernelMaskFilterOp() {}

    __aicore__ inline void Init(GM_ADDR scoresBuf, GM_ADDR maskBuf, GM_ADDR outputBuf, GM_ADDR indexBuf,
                                GM_ADDR countBuf, GM_ADDR taskBuf) {
        // Read task count from header using raw __gm__ access (valid in Init)
        __gm__ uint64_t* rawTask = (__gm__ uint64_t*)taskBuf;
        taskCount_ = static_cast<uint32_t>(rawTask[0]);

        // Bind GlobalTensors to full buffer ranges.
        // Sizes are upper bounds; actual access controlled by task offsets.
        // MaskFilterTask = 40 bytes = 5 × uint64_t
        // Buffer: [1 u64 header] + [N × 5 u64 tasks]
        taskRawGm_.SetGlobalBuffer((__gm__ uint64_t*)taskBuf, 1 + static_cast<uint64_t>(taskCount_) * 5);
        scoresGm_.SetGlobalBuffer((__gm__ float*)scoresBuf, kMaxFloatElements);
        maskGm_.SetGlobalBuffer((__gm__ uint64_t*)maskBuf, kMaxU64Elements);
        outputGm_.SetGlobalBuffer((__gm__ float*)outputBuf, kMaxFloatElements);
        indexGm_.SetGlobalBuffer((__gm__ uint32_t*)indexBuf, kMaxU32Elements);
        countGm_.SetGlobalBuffer((__gm__ uint32_t*)countBuf, kMaxCountElements);

        pipe_.InitBuffer(dummyQue_, 1, 256);
    }

    __aicore__ inline void Process() {
        auto dummy = dummyQue_.AllocTensor<float>();
        dummyQue_.FreeTensor(dummy);

        for (uint32_t t = 0; t < taskCount_; ++t) {
            // Read task descriptor fields from taskRawGm_
            // Layout per task: [scores_offset(u64)][mask_offset(u64)][output_offset(u64)]
            //                  [index_offset(u64)][doc_num(u32)|mask_words(u32)]
            uint64_t base = 1 + static_cast<uint64_t>(t) * 5;
            uint64_t scores_offset_bytes = taskRawGm_.GetValue(base);
            uint64_t mask_offset_bytes = taskRawGm_.GetValue(base + 1);
            uint64_t output_offset_bytes = taskRawGm_.GetValue(base + 2);
            uint64_t index_offset_bytes = taskRawGm_.GetValue(base + 3);
            uint64_t packed = taskRawGm_.GetValue(base + 4);

            uint32_t doc_num = static_cast<uint32_t>(packed & 0xFFFFFFFFULL);
            uint32_t mask_words = static_cast<uint32_t>(packed >> 32);

            uint32_t scores_idx = static_cast<uint32_t>(scores_offset_bytes / sizeof(float));
            uint32_t mask_idx = static_cast<uint32_t>(mask_offset_bytes / sizeof(uint64_t));
            uint32_t output_base = static_cast<uint32_t>(output_offset_bytes / sizeof(float));
            uint32_t index_base = static_cast<uint32_t>(index_offset_bytes / sizeof(uint32_t));

            uint32_t match_count = 0;

            for (uint32_t w = 0; w < mask_words; ++w) {
                uint64_t mask_word = maskGm_.GetValue(mask_idx + w);
                uint32_t base_id = w * 64;

                while (mask_word != 0) {
                    uint32_t bit = __builtin_ctzll(mask_word);
                    uint32_t doc_id = base_id + bit;

                    if (doc_id < doc_num) {
                        float score = scoresGm_.GetValue(scores_idx + doc_id);
                        outputGm_.SetValue(output_base + match_count, score);
                        indexGm_.SetValue(index_base + match_count, doc_id);
                        match_count++;
                    }
                    mask_word &= mask_word - 1;
                }
            }

            countGm_.SetValue(t, match_count);
        }
    }

   private:
    // Upper bounds for GlobalTensor binding (not actual allocation sizes)
    static constexpr uint32_t kMaxFloatElements = 16 * 1024 * 1024;  // 64 MB
    static constexpr uint32_t kMaxU64Elements = 4 * 1024 * 1024;     // 32 MB
    static constexpr uint32_t kMaxU32Elements = 16 * 1024 * 1024;    // 64 MB
    static constexpr uint32_t kMaxCountElements = 65536;

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> dummyQue_;
    AscendC::GlobalTensor<float> scoresGm_;
    AscendC::GlobalTensor<uint64_t> maskGm_;
    AscendC::GlobalTensor<float> outputGm_;
    AscendC::GlobalTensor<uint32_t> indexGm_;
    AscendC::GlobalTensor<uint32_t> countGm_;
    AscendC::GlobalTensor<uint64_t> taskRawGm_;
    uint32_t taskCount_ = 0;
};
