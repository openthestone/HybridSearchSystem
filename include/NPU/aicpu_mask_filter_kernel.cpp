/**
 * aicpu_mask_filter_kernel.cpp
 *
 * AI CPU kernel for MaskFilter op — replaces AI Core AscendC kernel.
 * Pure C++ bit-scan logic, identical semantics to kernel_mask_filter_op.
 *
 * IO layout (6 inputs, 0 outputs):
 *   io_addrs[0] = scores buffer   (d_result_ws — FP32 scores from NPU scoring)
 *   io_addrs[1] = mask buffer     (d_mf_mask — uint64 bitmaps)
 *   io_addrs[2] = output buffer   (d_mf_output — FP32 compacted scores, kernel writes)
 *   io_addrs[3] = index buffer    (d_mf_index — uint32 compacted indices, kernel writes)
 *   io_addrs[4] = count buffer    (d_mf_count — uint32 per-task counts, kernel writes)
 *   io_addrs[5] = task buffer     (d_mf_task — [uint64 N][MaskFilterTask × N])
 *
 * Task buffer layout:
 *   [0]           uint64_t task_count
 *   [1..N*5+1]    MaskFilterTask entries (40 bytes each, 5 × uint64_t)
 *
 * MaskFilterTask (40 bytes):
 *   scores_offset  (uint64_t) byte offset into scores buffer
 *   mask_offset    (uint64_t) byte offset into mask buffer
 *   output_offset  (uint64_t) byte offset into output buffer
 *   index_offset   (uint64_t) byte offset into index buffer
 *   doc_num        (uint32_t) number of docs in bucket
 *   mask_words     (uint32_t) ceil(doc_num / 64)
 */

#include <cstdint>

#pragma pack(push, 1)
struct AicpuParamHeadMF {
    uint32_t length;
    uint32_t ioAddrNum;
    uint32_t extInfoLength;
    uint64_t extInfoAddr;
};
#pragma pack(pop)

struct MaskFilterTask {
    uint64_t scores_offset;
    uint64_t mask_offset;
    uint64_t output_offset;
    uint64_t index_offset;
    uint32_t doc_num;
    uint32_t mask_words;
};

extern "C" {

uint32_t RunCpuKernel(void *param) {
    auto *head = reinterpret_cast<AicpuParamHeadMF *>(param);
    auto *io_addrs = reinterpret_cast<uint64_t *>(
        reinterpret_cast<char *>(param) + sizeof(AicpuParamHeadMF));

    // 6 IO addresses
    auto *scores = reinterpret_cast<float *>(io_addrs[0]);
    auto *mask   = reinterpret_cast<uint64_t *>(io_addrs[1]);
    auto *output = reinterpret_cast<float *>(io_addrs[2]);
    auto *index  = reinterpret_cast<uint32_t *>(io_addrs[3]);
    auto *count  = reinterpret_cast<uint32_t *>(io_addrs[4]);
    auto *task_raw = reinterpret_cast<uint64_t *>(io_addrs[5]);

    uint32_t N = static_cast<uint32_t>(task_raw[0]);
    auto *tasks = reinterpret_cast<MaskFilterTask *>(task_raw + 1);

    for (uint32_t t = 0; t < N; ++t) {
        auto &task = tasks[t];

        uint32_t scores_idx = static_cast<uint32_t>(task.scores_offset / sizeof(float));
        uint32_t mask_idx   = static_cast<uint32_t>(task.mask_offset / sizeof(uint64_t));
        uint32_t output_base = static_cast<uint32_t>(task.output_offset / sizeof(float));
        uint32_t index_base  = static_cast<uint32_t>(task.index_offset / sizeof(uint32_t));

        uint32_t match_count = 0;

        for (uint32_t w = 0; w < task.mask_words; ++w) {
            uint64_t mask_word = mask[mask_idx + w];
            uint32_t base_id = w * 64;

            while (mask_word != 0) {
                uint32_t bit = __builtin_ctzll(mask_word);
                uint32_t doc_id = base_id + bit;

                if (doc_id < task.doc_num) {
                    float score = scores[scores_idx + doc_id];
                    output[output_base + match_count] = score;
                    index[index_base + match_count] = doc_id;
                    match_count++;
                }
                mask_word &= mask_word - 1;
            }
        }

        count[t] = match_count;
    }

    return 0;
}

uint32_t CustSetCpuKernelContext(void *param) {
    (void)param;
    return 0;
}

}
