#include "vector_scorer_mmad.h"
#include <cstdlib>
#include "src/utils/logger.h"
#include "src/utils/env_switch.h"
#include "src/full_recall/core/constant_definition.h"
#include "src/full_recall/core/log_definition.h"
#include "configuration/develop_configuration.h"
#include "src/utils/performance_recorder.h"
#include "src/full_recall/retrieval/searcher/runtime/stream_manager.h"
#include "ascend_device/aclrtlaunch_kernel_vector_mmad.h"

#ifdef ASCENDC_CPU_DEBUG
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void kernel_vector_mmad(GM_ADDR queryMatrix, GM_ADDR docMatrix, GM_ADDR resultMatrix,
                                                         uint16_t m, uint16_t mOut, uint32_t n, uint16_t k,
                                                         uint16_t b1N, uint16_t b2N, uint32_t zNSegLen,
                                                         uint32_t blockLength)
#endif

    namespace NpuRetrieval {

    bool VectorScorerMmad::BatchCompute(const std::vector<uint16_t>& queryVectors, uint32_t queryDimension,
                                        const std::string& fieldName, GmBlock& resultChunk) {
        RecordGuard guard{"VectorScorerMmad BatchCompute"};
        ScorerLaunch launch;
        const bool launched = BatchComputeLaunch(queryVectors, queryDimension, fieldName, resultChunk, launch);
        // Sync unconditionally: on the failure path it is what releases anything the launch took.
        const bool synced = BatchComputeSync(launch);
        return launched && synced;
    }

    bool VectorScorerMmad::BatchComputeLaunch(const std::vector<uint16_t>& queryVectors, uint32_t queryDimension,
                                              const std::string& fieldName, GmBlock& resultChunk,
                                              ScorerLaunch& launch) {
        RecordGuard guard{"VectorScorerMmad Launch"};
        if (m_dataTable == nullptr) {
            LOG_ERROR("m_dataTable is nullptr");
            return false;
        }
        uint32_t docNum = m_dataTable->GetDocNum();
        VectorFieldData* fieldData{};
        if (!m_dataTable->GetVectorFieldData(fieldName, fieldData) || fieldData == nullptr) {
            LOG_ERROR("get vector field data failed, fieldName:" << fieldName);
            return false;
        }
        uint32_t splitDocNumZn = m_dataTable->GetSplitDocNumZn();
        uint32_t dataDimension = fieldData->GetDimension();
        if (docNum == 0) {
            LOG_ERROR("doc num is 0 of field: " << fieldName);
            return false;
        }
        if (dataDimension != queryDimension) {
            LOG_ERROR("Dimension check failed. dataDimension=" << dataDimension
                                                               << ", queryDimension=" << queryDimension);
            return false;
        }
        auto docDataSize = fieldData->GetLength();
        uint64_t expectedSize = static_cast<uint64_t>(docNum) * dataDimension * sizeof(int16_t);
        if (docDataSize < expectedSize) {
            LOG_ERROR("Check doc data size failed. docNum=" << docNum << ", docDataSize=" << docDataSize
                                                            << ", expectedSize=" << expectedSize
                                                            << ", dataDimension=" << dataDimension);
            return false;
        }

        uint16_t b1N = GetB1N(dataDimension);
        uint16_t b2N = splitDocNumZn;
        if (b1N == 0 || (b1N % b2N) != 0) {
            LOG_ERROR("invalid params, b1N=" << b1N << " b2N=" << b2N);
            return false;
        }
        uint32_t extendDocNum = m_dataTable->GetScoreExtendDocNum();
        uint32_t blockDocNum = extendDocNum / FLAGS_full_recall_scorer_block_dim;  // docs assigned to each core
        blockDocNum = ((blockDocNum + b1N - 1) / b1N) * b1N;                       // round up to a multiple of b1N
        uint32_t queryNum = queryVectors.size() / queryDimension;
        uint32_t blockDim = (extendDocNum + blockDocNum - 1) / blockDocNum;  // number of cores actually used
        LOG_DEBUG("deviceId:" << m_deviceId << " extendDocNum:" << extendDocNum << " blockDocNum:" << blockDocNum
                              << " blockDim:" << blockDim);
        if (queryNum == 0) {
            LOG_ERROR("query num is 0, queryVectors size " << queryVectors.size() << ", query dimension "
                                                           << queryDimension);
            return false;
        }
        uint32_t extendQueryNum = ((queryNum - 1) / 16 + 1) * 16;  // pad the query count up to a multiple of 16
        // NPUR_SCORER_TRIM_WRITE=1 (default off): write back only queryNum rows of the C matrix
        // instead of all extendQueryNum. The MMAD unit is 16x16 so the compute keeps the padded M
        // either way; only the padded rows' Fixpipe write-back goes, which nothing reads. Also
        // shrinks the result pool block by the same factor. Off = byte-identical to the old path.
        static const bool trimWrite = npur_env::On("NPUR_SCORER_TRIM_WRITE");
        const uint32_t resultRows = trimWrite ? queryNum : extendQueryNum;
        size_t resultInDeviceSize = extendDocNum * resultRows * sizeof(float);

        size_t queryAllSize = extendQueryNum * queryDimension * sizeof(uint16_t);  // uint16_t represent half
        bool state = true;
#ifdef ASCENDC_CPU_DEBUG
        uint16_t* queryInDevice = (uint16_t*)AscendC::GmAlloc(queryAllSize);
        for (uint32_t i = 0; i < queryVectors.size(); i++) {
            queryInDevice[i] = queryVectors[i];
        }
        resultChunk.data = (uint16_t*)AscendC::GmAlloc(resultInDeviceSize);
        kernel_vector_mmad(queryInDevice, fieldData->GetFieldDataInDevice(), resultChunk.data,
                           static_cast<uint16_t>(extendQueryNum), static_cast<uint16_t>(resultRows), extendDocNum,
                           static_cast<uint16_t>(queryDimension), b1N, b2N, splitDocNumZn, blockDocNum);
        AscendC::GmFree((void*)queryInDevice);
#else
    uint8_t* queryInDevice = nullptr;
    GmBlock queryChunk{GmPoolName::SCORER_QUERY_POOL, nullptr, 0};
    // Whichever way it was taken, this is the one way it goes back.
    auto releaseQuery = [&]() {
        if (queryChunk.data != nullptr) {
            GmMemoryManager::GetByDeviceId(m_deviceId)->FreeBlock(queryChunk);
            queryChunk.data = nullptr;  // FreeBlock does not clear it; a second call must be a no-op
        } else if (queryInDevice != nullptr) {
            CHECK_ACL_ONLY_LOG(aclrtFree(queryInDevice));
        }
        queryInDevice = nullptr;
    };
    aclError ret = ACL_SUCCESS;
    if (PoolSmallH2dEnabled()) {
        GmMemoryManager::GetByDeviceId(m_deviceId)
            ->AllocateBlock(GmPoolName::SCORER_QUERY_POOL, static_cast<uint32_t>(queryAllSize), queryChunk);
        if (queryChunk.data == nullptr) {
            LOG_ERROR("queryInDevice pool allocate fail, deviceId is:" << m_deviceId);
            return false;
        }
        queryInDevice = reinterpret_cast<uint8_t*>(queryChunk.data);
    } else {
        ret = aclrtMalloc((void**)&queryInDevice, queryAllSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_ERROR("queryInDevice aclrtMalloc fail, deviceId is:" << m_deviceId << " error code is:" << ret);
            return false;
        }
    }
    ret = aclrtMemcpy(queryInDevice, queryAllSize, (void*)queryVectors.data(), queryVectors.size() * sizeof(uint16_t),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("queryInDevice aclrtMemcpy fail, deviceId is:" << m_deviceId << " error code is:" << ret);
        releaseQuery();
        return false;
    }
    GmMemoryManager::GetByDeviceId(m_deviceId)
        ->AllocateBlock(GmPoolName::VECTOR_SCORE_RESULT_POOL, resultInDeviceSize, resultChunk);
    if (resultChunk.data == nullptr) {
        LOG_ERROR("GmMemoryManager allocate failed.");
        releaseQuery();
        return false;
    }
    aclrtStream stream = StreamManager::GetInstance()->GetStream(m_deviceId);
    {
        RecordGuard guard{"VectorScorerMmad Kernel Launch"};
        ACLRT_LAUNCH_KERNEL(kernel_vector_mmad)
        (blockDim, stream, queryInDevice, fieldData->GetFieldDataInDevice(),
         reinterpret_cast<uint8_t*>(resultChunk.data), static_cast<uint16_t>(extendQueryNum),
         static_cast<uint16_t>(resultRows), extendDocNum, static_cast<uint16_t>(queryDimension), b1N, b2N,
         splitDocNumZn, blockDocNum);
    }
    // The kernel is now running. The stream to wait on and the query buffer it reads travel to
    // Sync, which is the only place allowed to release them.
    launch.stream = stream;
    launch.queryInDevice = queryInDevice;
    launch.queryChunk = queryChunk;
    launch.launched = true;
#endif
        LOG_DEBUG("finish vector scoring by mmad, query num: "
                  << queryNum << ", extend query num: " << extendQueryNum << ", query dimension: " << queryDimension
                  << ", fieldName: " << fieldName << ", doc num: " << docNum << ", extend doc num: " << extendDocNum);
        return state;
    }

    bool VectorScorerMmad::BatchComputeSync(ScorerLaunch & launch) {
        if (!launch.launched) {
            return true;
        }
        RecordGuard guard{"VectorScorerMmad Sync"};
        bool state = true;
#ifndef ASCENDC_CPU_DEBUG
        {
            RecordGuard guard{"VectorScorerMmad Kernel Sync"};
            aclError ret = aclrtSynchronizeStream(launch.stream);
            if (ret != ACL_SUCCESS) {
                LOG_ERROR("aclrtSynchronizeStream fail,  error code is:" << ret);
                state = false;
            }
        }
        StreamManager::GetInstance()->FreeStream(m_deviceId, launch.stream);
        if (launch.queryChunk.data != nullptr) {
            GmMemoryManager::GetByDeviceId(m_deviceId)->FreeBlock(launch.queryChunk);
        } else {
            CHECK_ACL_ONLY_LOG(aclrtFree(launch.queryInDevice));
        }
#endif
        launch = ScorerLaunch{};
        return state;
    }

    uint16_t VectorScorerMmad::GetB1N(const uint32_t dimension) {
        // Currently only 910B4 is supported; total L1 size is 512K.
        // The B matrix copied into L1 each time is dimension*b1N*{buff_num}*sizeof(FP16).
        switch (dimension) {
            // 32-dim: 32*2048*3*2=393216
            case 32:
                return 2048;
            // 64-dim: 64*1024*3*2=393216
            case 64:
                return 1024;
            // 128-dim: 128*512*3*2=393216
            case 128:
                return 512;
            default:
                LOG_WARN("unsupported dimension:" << dimension);
                return 0;
        }
    }
}  // namespace NpuRetrieval
