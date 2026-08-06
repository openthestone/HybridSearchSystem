// npu/kernel_score.cpp — AscendC MMad kernel: query · docs → scores.
//
// Adapted from sks_hw `kernel_vector_mmad_op.cpp`. Differences:
//   - No BatchTaskData: each block computes its slice via GetBlockIdx()
//   - Single query (MatrixB), all docs (MatrixA) → scores (MatrixC column 0)
//   - DOCS_PER_BLOCK = 1024 (UB-safe at k=64)
//
// Per-block work:
//   doc_offset = blockIdx * DOCS_PER_BLOCK
//   m = min(DOCS_PER_BLOCK, total_docs - doc_offset)
//   Compute [m × k] × [k × 16] → [m × 16] (Cube MMad), Fixpipe column 0 → m floats.
//
// Inputs:
//   queryFP16[k]          row-major
//   docsFP16[total*k]     row-major (M × K)
// Output:
//   scoresF32[total]      dot(query, docs[i])
//
// Launch:
//   blockDim = ceil(total / DOCS_PER_BLOCK)

#include "kernel_operator.h"

constexpr uint32_t SCORE_DOCS_PER_BLOCK = 256;
constexpr uint32_t CUBE_BLOCK = 16;
constexpr uint32_t CUBE_BLOCK_SIZE = 16 * 16;

class KernelScoreOp {
   public:
    __aicore__ inline KernelScoreOp() {}

    __aicore__ inline void Init(GM_ADDR queryGM, GM_ADDR docsGM, GM_ADDR scoresGM, uint32_t total_docs,
                                uint32_t vector_dim, uint32_t block_offset_blocks) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

        uint32_t block_id = AscendC::GetBlockIdx() + block_offset_blocks;
        uint32_t doc_offset = block_id * SCORE_DOCS_PER_BLOCK;

        uint32_t m = 0;
        if (doc_offset < total_docs) {
            uint32_t remaining = total_docs - doc_offset;
            m = remaining < SCORE_DOCS_PER_BLOCK ? remaining : SCORE_DOCS_PER_BLOCK;
        }
        m_ = m;
        k_ = vector_dim;
        n_ = 16;

        uint32_t m_aligned = (m_ + CUBE_BLOCK - 1) / CUBE_BLOCK * CUBE_BLOCK;
        uint32_t k_aligned = (k_ + CUBE_BLOCK - 1) / CUBE_BLOCK * CUBE_BLOCK;
        aSize_ = m_aligned * k_aligned;
        bSize_ = k_aligned * n_;
        cSize_ = m_aligned * n_;

        queryGM_.SetGlobalBuffer((__gm__ half*)(queryGM));
        docsGM_.SetGlobalBuffer((__gm__ half*)(docsGM + (uint64_t)doc_offset * k_ * sizeof(half)));
        scoresGM_.SetGlobalBuffer((__gm__ float*)(scoresGM + (uint64_t)doc_offset * sizeof(float)));
    }

    __aicore__ inline void Process() {
        if (m_ == 0)
            return;
        pipe.InitBuffer(inQueueA1, 1, aSize_ * sizeof(half));
        pipe.InitBuffer(inQueueA2, 1, aSize_ * sizeof(half));
        pipe.InitBuffer(inQueueB1, 1, bSize_ * sizeof(half));
        pipe.InitBuffer(inQueueB2, 1, bSize_ * sizeof(half));
        pipe.InitBuffer(outQueueCO1, 1, cSize_ * sizeof(float));

        CopyIn();
        SplitA();
        SplitB();
        Compute();
        CopyOut();
    }

   private:
    __aicore__ inline uint32_t CeilCubeBlock(uint32_t len) {
        return (len + CUBE_BLOCK - 1) / CUBE_BLOCK;
    }

    __aicore__ inline void CopyIn() {
        AscendC::LocalTensor<half> a1Local = inQueueA1.AllocTensor<half>();
        AscendC::LocalTensor<half> b1Local = inQueueB1.AllocTensor<half>();

        AscendC::Nd2NzParams nd2nzA1Params;
        nd2nzA1Params.ndNum = 1;
        nd2nzA1Params.nValue = m_;
        nd2nzA1Params.dValue = k_;
        nd2nzA1Params.srcNdMatrixStride = 0;
        nd2nzA1Params.srcDValue = k_;
        nd2nzA1Params.dstNzC0Stride = CeilCubeBlock(m_) * CUBE_BLOCK;
        nd2nzA1Params.dstNzNStride = 1;
        nd2nzA1Params.dstNzMatrixStride = 0;
        AscendC::DataCopy(a1Local, docsGM_, nd2nzA1Params);

        AscendC::Nd2NzParams nd2nzB1Params;
        nd2nzB1Params.ndNum = 1;
        nd2nzB1Params.nValue = k_;
        nd2nzB1Params.dValue = n_;
        nd2nzB1Params.srcNdMatrixStride = 0;
        nd2nzB1Params.srcDValue = n_;
        nd2nzB1Params.dstNzC0Stride = CeilCubeBlock(k_) * CUBE_BLOCK;
        nd2nzB1Params.dstNzNStride = 1;
        nd2nzB1Params.dstNzMatrixStride = 0;
        AscendC::DataCopy(b1Local, queryGM_, nd2nzB1Params);

        inQueueA1.EnQue(a1Local);
        inQueueB1.EnQue(b1Local);
    }

    __aicore__ inline void SplitA() {
        AscendC::LocalTensor<half> a1Local = inQueueA1.DeQue<half>();
        AscendC::LocalTensor<half> a2Local = inQueueA2.AllocTensor<half>();

        uint32_t dstOffset = CeilCubeBlock(k_) * CUBE_BLOCK_SIZE;
        uint32_t srcOffset = CUBE_BLOCK_SIZE;

        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = CeilCubeBlock(k_);
        loadDataParams.srcStride = CeilCubeBlock(m_);
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = false;

        for (uint32_t i = 0; i < CeilCubeBlock(m_); ++i) {
            AscendC::LoadData(a2Local[i * dstOffset], a1Local[i * srcOffset], loadDataParams);
        }

        inQueueA2.EnQue<half>(a2Local);
        inQueueA1.FreeTensor(a1Local);
    }

    __aicore__ inline void SplitB() {
        AscendC::LocalTensor<half> b1Local = inQueueB1.DeQue<half>();
        AscendC::LocalTensor<half> b2Local = inQueueB2.AllocTensor<half>();

        uint32_t dstOffset = CeilCubeBlock(n_) * CUBE_BLOCK_SIZE;
        uint32_t srcOffset = CUBE_BLOCK_SIZE;

        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = CeilCubeBlock(n_);
        loadDataParams.srcStride = CeilCubeBlock(k_);
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = true;

        for (uint32_t i = 0; i < CeilCubeBlock(k_); ++i) {
            AscendC::LoadData(b2Local[i * dstOffset], b1Local[i * srcOffset], loadDataParams);
        }

        inQueueB1.FreeTensor(b1Local);
        inQueueB2.EnQue<half>(b2Local);
    }

    __aicore__ inline void Compute() {
        AscendC::LocalTensor<half> a2Local = inQueueA2.DeQue<half>();
        AscendC::LocalTensor<half> b2Local = inQueueB2.DeQue<half>();
        AscendC::LocalTensor<float> c1Local = outQueueCO1.AllocTensor<float>();

        AscendC::MmadParams mmadParams;
        mmadParams.m = m_;
        mmadParams.n = n_;
        mmadParams.k = k_;

        AscendC::Mmad(c1Local, a2Local, b2Local, mmadParams);

        outQueueCO1.EnQue<float>(c1Local);
        inQueueA2.FreeTensor(a2Local);
        inQueueB2.FreeTensor(b2Local);
    }

    __aicore__ inline void CopyOut() {
        AscendC::LocalTensor<float> c1Local = outQueueCO1.DeQue<float>();

        AscendC::FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = 1;
        fixpipeParams.mSize = m_;
        fixpipeParams.srcStride = m_;
        fixpipeParams.dstStride = 1;
        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = 0;

        AscendC::Fixpipe(scoresGM_, c1Local, fixpipeParams);
        outQueueCO1.FreeTensor(c1Local);
    }

   private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::A1, 1> inQueueA1;
    AscendC::TQue<AscendC::TPosition::A2, 1> inQueueA2;
    AscendC::TQue<AscendC::TPosition::B1, 1> inQueueB1;
    AscendC::TQue<AscendC::TPosition::B2, 1> inQueueB2;
    AscendC::TQue<AscendC::TPosition::CO1, 1> outQueueCO1;

    AscendC::GlobalTensor<half> queryGM_;
    AscendC::GlobalTensor<half> docsGM_;
    AscendC::GlobalTensor<float> scoresGM_;

    uint32_t m_, k_, n_;
    uint32_t aSize_, bSize_, cSize_;
};

extern "C" __global__ __aicore__ void kernel_score(GM_ADDR queryFP16, GM_ADDR docsFP16, GM_ADDR scoresF32,
                                                   uint32_t total_docs, uint32_t vector_dim,
                                                   uint32_t block_offset_blocks) {
    KernelScoreOp op;
    op.Init(queryFP16, docsFP16, scoresF32, total_docs, vector_dim, block_offset_blocks);
    op.Process();
}
