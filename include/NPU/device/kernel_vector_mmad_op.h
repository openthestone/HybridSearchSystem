#pragma once
#include "kernel_operator.h"
#include "device_common.h"

/*
    MatrixA是桶的数据矩阵（ceil(doc_num / 16) * 64 ）
    MatrixB是查询矩阵（64*16）
    MatrixC是结果矩阵（ceil(doc_num / 16) * 16）
    三者都是HBM上的指针
*/

constexpr uint32_t CUBE_BLOCK = 16;
constexpr uint32_t CUBE_BLOCK_SIZE = 16 * 16;

class KernelVectorMmadOp
{
public:
    __aicore__ inline KernelVectorMmadOp() {}

    __aicore__ inline void Init(GM_ADDR MatrixB, GM_ADDR taskBuffer,
                                GM_ADDR MatrixA, GM_ADDR MatrixC)
    {
        // 1. 获取当前核的任务信息
        uint32_t taskId = AscendC::GetBlockIdx();
        __gm__ BatchTaskData *taskPtr = (__gm__ BatchTaskData *)taskBuffer;
        __gm__ BatchTaskData *currentTaskGm = &taskPtr[taskId];

        // 2. 解析任务参数
        uint32_t task_offset_A = currentTaskGm->offset_A;
        uint32_t task_offset_C = currentTaskGm->offset_C;
        
        // 维度定义
        // m: 当前桶的文档数量
        // k: 向量维度 (64)
        // n: 查询向量填充后的维度 (16)
        this->m = currentTaskGm->m; 
        this->k = 64;
        this->n = 16;

        // 计算 Buffer 大小 (需对齐到 16 以防止 DataCopy 越界)
        uint32_t m_aligned = (m + CUBE_BLOCK - 1) / CUBE_BLOCK * CUBE_BLOCK;
        uint32_t k_aligned = (k + CUBE_BLOCK - 1) / CUBE_BLOCK * CUBE_BLOCK;
        uint32_t n_aligned = (n + CUBE_BLOCK - 1) / CUBE_BLOCK * CUBE_BLOCK;

        // 这里完成对桶数据矩阵的行 padding逻辑
        // 后续分配UB内存都是按照aSize来分配的
        this->aSize = m_aligned * k_aligned;
        this->bSize = k_aligned * n_aligned;
        this->cSize = m_aligned * n_aligned;

        // 3. 设置 GM 指针
        // 严格遵循参考代码：KERNEL_TYPE_AIC_ONLY
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
        
        aGM.SetGlobalBuffer((__gm__ half *)(MatrixA + (uint64_t)task_offset_A));
        bGM.SetGlobalBuffer((__gm__ half *)MatrixB);
        cGM.SetGlobalBuffer((__gm__ float *)(MatrixC + (uint64_t)task_offset_C));

        // 4. 初始化 Pipe 和 Queue
        // 严格遵循参考代码：深度为 1 (单缓冲)
        pipe.InitBuffer(inQueueA1, 1, aSize * sizeof(half));
        pipe.InitBuffer(inQueueA2, 1, aSize * sizeof(half));
        pipe.InitBuffer(inQueueB1, 1, bSize * sizeof(half));
        pipe.InitBuffer(inQueueB2, 1, bSize * sizeof(half));
        pipe.InitBuffer(outQueueCO1, 1, cSize * sizeof(float));
    }

    __aicore__ inline void Process()
    {
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

    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<half> a1Local = inQueueA1.AllocTensor<half>();
        AscendC::LocalTensor<half> b1Local = inQueueB1.AllocTensor<half>();

        // Matrix A: Doc Vectors [m, 64] -> ND2NZ
        AscendC::Nd2NzParams nd2nzA1Params;
        nd2nzA1Params.ndNum = 1;
        nd2nzA1Params.nValue = m;
        nd2nzA1Params.dValue = k;
        nd2nzA1Params.srcNdMatrixStride = 0;
        nd2nzA1Params.srcDValue = k;
        nd2nzA1Params.dstNzC0Stride = CeilCubeBlock(m) * CUBE_BLOCK;
        nd2nzA1Params.dstNzNStride = 1;
        nd2nzA1Params.dstNzMatrixStride = 0;
        AscendC::DataCopy(a1Local, aGM, nd2nzA1Params);

        // Matrix B: Query Vector [64, 16] -> ND2NZ
        AscendC::Nd2NzParams nd2nzB1Params;
        nd2nzB1Params.ndNum = 1;
        nd2nzB1Params.nValue = k;
        nd2nzB1Params.dValue = n;
        nd2nzB1Params.srcNdMatrixStride = 0;
        nd2nzB1Params.srcDValue = n;
        nd2nzB1Params.dstNzC0Stride = CeilCubeBlock(k) * CUBE_BLOCK;
        nd2nzB1Params.dstNzNStride = 1;
        nd2nzB1Params.dstNzMatrixStride = 0;
        AscendC::DataCopy(b1Local, bGM, nd2nzB1Params);

        inQueueA1.EnQue(a1Local);
        inQueueB1.EnQue(b1Local);
    }

    __aicore__ inline void SplitA()
    {
        AscendC::LocalTensor<half> a1Local = inQueueA1.DeQue<half>();
        AscendC::LocalTensor<half> a2Local = inQueueA2.AllocTensor<half>();

        uint32_t dstOffset = CeilCubeBlock(k) * CUBE_BLOCK_SIZE;
        uint32_t srcOffset = CUBE_BLOCK_SIZE;

        // Nz -> Zz (L1 -> L0A)
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = CeilCubeBlock(k);
        loadDataParams.srcStride = CeilCubeBlock(m);
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = false;

        for (int i = 0; i < CeilCubeBlock(m); ++i) {
            AscendC::LoadData(a2Local[i * dstOffset], a1Local[i * srcOffset], loadDataParams);
        }

        inQueueA2.EnQue<half>(a2Local);
        inQueueA1.FreeTensor(a1Local);
    }

    __aicore__ inline void SplitB()
    {
        AscendC::LocalTensor<half> b1Local = inQueueB1.DeQue<half>();
        AscendC::LocalTensor<half> b2Local = inQueueB2.AllocTensor<half>();

        uint32_t dstOffset = CeilCubeBlock(n) * CUBE_BLOCK_SIZE;
        uint32_t srcOffset = CUBE_BLOCK_SIZE;

        // Nz -> Zn (L1 -> L0B)
        // 注意：参考代码中 ifTranspose = true，这对右矩阵（B）是必须的
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = CeilCubeBlock(n);
        loadDataParams.srcStride = CeilCubeBlock(k);
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = true; 

        for (int i = 0; i < CeilCubeBlock(k); ++i) {
            AscendC::LoadData(b2Local[i * dstOffset], b1Local[i * srcOffset], loadDataParams);
        }

        inQueueB1.FreeTensor(b1Local);
        inQueueB2.EnQue<half>(b2Local);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<half> a2Local = inQueueA2.DeQue<half>();
        AscendC::LocalTensor<half> b2Local = inQueueB2.DeQue<half>();
        AscendC::LocalTensor<float> c1Local = outQueueCO1.AllocTensor<float>();

        AscendC::MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;

        AscendC::Mmad(c1Local, a2Local, b2Local, mmadParams);

        outQueueCO1.EnQue<float>(c1Local);
        inQueueA2.FreeTensor(a2Local);
        inQueueB2.FreeTensor(b2Local);
    }

    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<float> c1Local = outQueueCO1.DeQue<float>();
        
        AscendC::FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = n;
        fixpipeParams.mSize = m;
        fixpipeParams.srcStride = m;
        fixpipeParams.dstStride = n;

        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = 0;

        AscendC::Fixpipe(cGM, c1Local, fixpipeParams);
        outQueueCO1.FreeTensor(c1Local);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::A1, 1> inQueueA1;
    AscendC::TQue<AscendC::TPosition::A2, 1> inQueueA2;
    AscendC::TQue<AscendC::TPosition::B1, 1> inQueueB1;
    AscendC::TQue<AscendC::TPosition::B2, 1> inQueueB2;
    AscendC::TQue<AscendC::TPosition::CO1, 1> outQueueCO1;

    AscendC::GlobalTensor<half> aGM;
    AscendC::GlobalTensor<half> bGM;
    AscendC::GlobalTensor<float> cGM;
    
    // 动态维度
    uint32_t m, k, n;
    uint32_t aSize, bSize, cSize;
};