#pragma once

#include "kernel_operator.h"
#include "device_common.h"

class KernelVectorMmadOp {
   public:
    __aicore__ inline KernelVectorMmadOp() {}

    /*
     * queryMatrix: query
     * docMatrix: doc
     * resultMatrix: result matrix
     * mtrM: left-matrix height, i.e. number of queries
     * mtrN: right-matrix width, i.e. total doc count; must be a multiple of b2N
     * mtrK: left-matrix width / right-matrix height, i.e. the dimension size
     * nZnSegLen: offline data tiling size, matching split_doc_num_zn in the index meta config
     * b1N: docs per L2->L1 copy; must be a multiple of b2N
     * b2N: docs per L1->L0B copy; ideally nZnSegLen should be configured to match it
     * blockLength: number of right-matrix columns a full core computes; must be a multiple of b1N
     * pipeIn: framework object that manages global memory and synchronization
     */
    __aicore__ inline void Init(GM_ADDR queryMatrix, GM_ADDR docMatrix, GM_ADDR resultMatrix, uint16_t mtrM,
                                uint32_t mtrN, uint16_t mtrK, uint16_t nZnSegLen, uint16_t b1N, uint16_t b2N,
                                uint32_t blockLength, AscendC::TPipe* pipeIn) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
        int64_t blockIdx = AscendC::GetBlockIdx();
        int64_t blockNum = AscendC::GetBlockNum();
        if (blockIdx == blockNum - 1) {
            // tail core
            m_matrixN = mtrN - blockLength * blockIdx;
        } else {
            m_matrixN = blockLength;
        }
        DEBUG_LOG("[INFO] blockIdx[%d], queryMatrix=0x%x,  docMatrix=0x%x, resultMatrix=0x%x\n", blockIdx,
                  (uint64_t)queryMatrix, (uint64_t)docMatrix, (uint64_t)resultMatrix);
        m_docNum = mtrN;
        m_matrixM = mtrM;
        m_matrixK = mtrK;
        m_nZnSegLen = nZnSegLen;
        m_b1N = b1N;
        m_b2N = b2N;
        if (m_b2N > m_nZnSegLen) {
            m_b2N = m_nZnSegLen;
        }
        DEBUG_LOG("[INFO] m_docNum=%d, m_matrixM=%d, m_matrixK=%d, m_nZnSegLen=%d, m_b1N=%d, m_b2N=%d\n", m_docNum,
                  m_matrixM, m_matrixK, m_nZnSegLen, m_b1N, m_b2N);

        m_aSize = m_matrixM * m_matrixK;
        m_bSize = m_matrixK * m_matrixN;
        m_b1Size = m_matrixK * m_b1N;
        m_b2Size = m_matrixK * m_b2N;
        m_cSize = m_matrixM * m_matrixN;
        m_mUnitNum = m_matrixM / m_mmadUnit;
        m_kUnitNum = m_matrixK / m_mmadUnit;
        DEBUG_LOG("[INFO] m_aSize=%d, m_bSize=%d, m_b1Size=%d, m_b2Size=%d, m_cSize=%d, m_mUnitNum=%d, m_kUnitNum=%d\n",
                  m_aSize, m_bSize, m_b1Size, m_b2Size, m_cSize, m_mUnitNum, m_kUnitNum);

        m_nB1BlockNum = (m_matrixN - 1) / m_b1N + 1;  // number of doc L2->L1 copy blocks
        m_nB2BlockNum = m_b1N / m_b2N;                // L1->L0B copy blocks per one L2->L1 copy
        m_nUnitPerB1Block = m_b1N / m_mmadUnit;
        m_nUnitPerB2Block = m_b2N / m_mmadUnit;

        m_c1Size = m_b2N * m_matrixM;
        m_c2Size = m_b1N * m_matrixM;
        m_unitSize = m_mmadUnit * m_mmadUnit;
        DEBUG_LOG(
            "[INFO] m_nB1BlockNum=%d, m_nB2BlockNum=%d, m_nUnitPerB1Block=%d, m_nUnitPerB2Block=%d, m_c1Size=%d, "
            "m_c2Size=%d, m_unitSize=%d\n",
            m_nB1BlockNum, m_nB2BlockNum, m_nUnitPerB1Block, m_nUnitPerB2Block, m_c1Size, m_c2Size, m_unitSize);

        m_aGM.SetGlobalBuffer((__gm__ half*)queryMatrix, m_aSize * sizeof(half));
        m_bGM.SetGlobalBuffer((__gm__ half*)docMatrix + m_matrixK * blockLength * blockIdx, m_bSize * sizeof(half));
        m_cGM.SetGlobalBuffer((__gm__ float*)resultMatrix + blockLength * blockIdx, m_cSize * sizeof(float));

        m_pipe = pipeIn;
        m_pipe->InitBuffer(m_inQueueA1, 1, m_aSize * sizeof(half));            // L2->L1
        m_pipe->InitBuffer(m_inQueueA2, 1, m_aSize * sizeof(half));            // L1->L0A
        m_pipe->InitBuffer(m_inQueueB1, 3, m_b1N * m_matrixK * sizeof(half));  // L2->L1, 3 blocks (triple buffer)
        m_pipe->InitBuffer(m_inQueueB2, 2, m_b2Size * sizeof(half));           // L1->L0B, 2 blocks (double buffer)
        m_pipe->InitBuffer(m_outQueueCO1, 2, m_c1Size * sizeof(float));        // L0C->UB, 2 blocks (double buffer)
        m_copyAInParams = {1,         m_matrixM, m_matrixK, static_cast<uint16_t>(m_matrixM * m_matrixK),
                           m_matrixK, m_matrixM, 1,         static_cast<uint16_t>(m_matrixM * m_matrixK)};
        m_mmadParams.m = m_matrixM;
        m_mmadParams.n = m_b2N;
        m_mmadParams.k = m_matrixK;
        m_aggrC2CopyParams.blockCount = 1;
        m_aggrC2CopyParams.blockLen = m_mUnitNum * m_nUnitPerB2Block;
        m_aggrC2EnhancedParams.blockMode = AscendC::BlockMode::BLOCK_MODE_MATRIX;
    }
    __aicore__ inline void Process() {
        CopyInA();
        SplitA();

        AscendC::LocalTensor<half> a2Local = m_inQueueA2.DeQue<half>();
        // loop 1: number of L2->L1 copy blocks
        for (uint32_t b1Idx = 0; b1Idx < m_nB1BlockNum; ++b1Idx) {
            CopyInB(b1Idx);
            AscendC::LocalTensor<half> b1Local = m_inQueueB1.DeQue<half>();
            if (b1Idx == (m_nB1BlockNum - 1)) {
                // tail block
                m_nB2BlockNum = (m_matrixN - m_b1N * b1Idx) / m_b2N;
            }
            // loop 2: number of L1->L0B copy blocks
            for (uint16_t b2Idx = 0; b2Idx < m_nB2BlockNum; ++b2Idx) {
                SplitB(b1Local, b2Idx);
                Compute(a2Local);
                CopyOut(b1Idx, b2Idx);
            }
            m_inQueueB1.FreeTensor(b1Local);
        }

        m_inQueueA2.FreeTensor(a2Local);
    }

   private:
    __aicore__ inline void CopyInA() {
        AscendC::LocalTensor<half> a1Local = m_inQueueA1.AllocTensor<half>();
        // nd to nz
        AscendC::DataCopy(a1Local, m_aGM, m_copyAInParams);
        m_inQueueA1.EnQue(a1Local);
    }

    __aicore__ inline void SplitA() {
        uint32_t srcOffset = 0;
        uint32_t dstOffset = 0;
        AscendC::LocalTensor<half> a1Local = m_inQueueA1.DeQue<half>();
        AscendC::LocalTensor<half> a2Local = m_inQueueA2.AllocTensor<half>();
        // transform nz to zz
        for (uint16_t i = 0; i < m_mUnitNum; ++i) {
            AscendC::LoadData2DParams loadADataParams;
            loadADataParams.repeatTimes = m_kUnitNum;
            loadADataParams.srcStride = m_mUnitNum;
            loadADataParams.ifTranspose = false;

            AscendC::LoadData(a2Local[dstOffset], a1Local[srcOffset], loadADataParams);

            srcOffset += m_unitSize;
            dstOffset += m_matrixK * m_mmadUnit;
        }

        m_inQueueA2.EnQue<half>(a2Local);
        m_inQueueA1.FreeTensor(a1Local);
    }

    __aicore__ inline void CopyInB(uint32_t b1Idx) {
        // the offline side already tiled the data; just copy it
        AscendC::LocalTensor<half> b1Local = m_inQueueB1.AllocTensor<half>();
        if (m_b1N >= m_nZnSegLen) {
            uint32_t doneSize = b1Idx * m_b1Size;
            uint32_t restSize = m_bSize - doneSize;
            AscendC::DataCopy(b1Local, m_bGM[doneSize], (restSize >= m_b1Size) ? m_b1Size : restSize);
        } else {
            AscendC::DataCopyParams copyBParams(m_kUnitNum, m_b1N, m_matrixN - m_b1N, 0);
            AscendC::DataCopy(b1Local, m_bGM[b1Idx * m_nUnitPerB1Block * m_unitSize], copyBParams);
        }
        m_inQueueB1.EnQue(b1Local);
    }

    __aicore__ inline void SplitB(const AscendC::LocalTensor<half>& b1Local, uint16_t b2Idx) {
        AscendC::LocalTensor<half> b2Local = m_inQueueB2.AllocTensor<half>();
        AscendC::LoadData2DParams loadDataParams;
        if (m_b2N >= m_nZnSegLen) {
            uint32_t b1StartPos = b2Idx * m_b2Size;
            loadDataParams.repeatTimes = m_nUnitPerB2Block * m_kUnitNum;
            loadDataParams.srcStride = 1;
            AscendC::LoadData(b2Local, b1Local[b1StartPos], loadDataParams);
        } else {
            uint32_t b1StartPos = b2Idx * m_nUnitPerB2Block * m_unitSize;
            loadDataParams.repeatTimes = m_nUnitPerB2Block;
            loadDataParams.srcStride = 1;
            for (uint16_t i = 0; i < m_kUnitNum; i++) {
                AscendC::LoadData(b2Local[i * m_b2Size / m_kUnitNum], b1Local[b1StartPos + i * m_b1Size / m_kUnitNum],
                                  loadDataParams);
            }
        }
        m_inQueueB2.EnQue<half>(b2Local);
    }
    __aicore__ inline void Compute(const AscendC::LocalTensor<half>& a2Local) {
        AscendC::LocalTensor<half> b2Local = m_inQueueB2.DeQue<half>();
        AscendC::LocalTensor<float> c1Local = m_outQueueCO1.AllocTensor<float>();
        AscendC::Mmad(c1Local, a2Local, b2Local, m_mmadParams);
        m_outQueueCO1.EnQue<float>(c1Local);
        m_inQueueB2.FreeTensor(b2Local);
    }

    __aicore__ inline void CopyOut(uint32_t b1Idx, uint32_t b2Idx) {
        uint32_t doneN = b1Idx * m_b1N + b2Idx * m_b2N;
        AscendC::LocalTensor<float> c1Local = m_outQueueCO1.DeQue<float>();
        AscendC::FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = m_b2N;
        fixpipeParams.mSize = m_matrixM;
        fixpipeParams.srcStride = m_matrixM;
        fixpipeParams.dstStride = m_docNum;  // number of elements per row of the destination ND matrix (up to ~2.5M)

        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = 0;
        AscendC::Fixpipe(m_cGM[doneN], c1Local, fixpipeParams);
        m_outQueueCO1.FreeTensor(c1Local);
    }

   private:
    AscendC::TPipe* m_pipe{};

    AscendC::TQue<AscendC::QuePosition::A1, 1> m_inQueueA1;
    AscendC::TQue<AscendC::QuePosition::A2, 1> m_inQueueA2;
    AscendC::TQue<AscendC::QuePosition::B1, 1> m_inQueueB1;
    AscendC::TQue<AscendC::QuePosition::B2, 1> m_inQueueB2;
    // dst queue
    AscendC::TQue<AscendC::QuePosition::CO1, 1> m_outQueueCO1;

    AscendC::GlobalTensor<half> m_aGM;
    AscendC::GlobalTensor<half> m_bGM;
    AscendC::GlobalTensor<float> m_cGM;

    uint16_t m_matrixM{};
    uint32_t m_matrixN{};
    uint16_t m_matrixK{};
    uint32_t m_docNum{};
    uint16_t m_nZnSegLen{};
    uint16_t m_b1N = 2048;
    uint16_t m_b2N = 512;
    uint16_t m_mmadUnit = 16;

    uint32_t m_aSize{}, m_bSize{}, m_b1Size{}, m_b2Size{}, m_cSize{}, m_c1Size{}, m_c2Size{}, m_unitSize{};
    uint16_t m_mUnitNum{}, m_kUnitNum{}, m_nB2BlockNum{}, m_nUnitPerB1Block{}, m_nUnitPerB2Block{};
    uint32_t m_nB1BlockNum{};

    AscendC::Nd2NzParams m_copyAInParams;
    AscendC::MmadParams m_mmadParams;
    AscendC::DataCopyParams m_aggrC2CopyParams;
    AscendC::DataCopyEnhancedParams m_aggrC2EnhancedParams;
};
