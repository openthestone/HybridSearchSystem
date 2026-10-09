#include <gflags/gflags.h>
#include "builder_impl.h"
#include <filesystem>
#include <mutex>
#include "src/full_recall/indexer/builder/segment/idmapping_builder.h"
#include "src/full_recall/indexer/builder/segment/vector_builder.h"
#include "src/full_recall/indexer/builder/segment/inverted_builder.h"
#include "src/full_recall/indexer/builder/meta_builder.h"
#include "stream_input.h"
#include "src/full_recall/indexer/file/memory_data_manager.h"
#include "src/full_recall/indexer/schema/builder_schema.h"
#include "src/utils/file_manager.h"
#include "src/utils/logger.h"
#include "src/workflow/async/executor.h"

DECLARE_bool(build_from_memory);

namespace NpuRetrieval {
namespace {

bool EnsureDir(const std::string& dir) {
    std::error_code ec;
    if (std::filesystem::exists(dir, ec)) {
        return true;
    }
    if (!AddDirRecursively(dir)) {
        LOG_ERROR("create output dir fail. path:" << dir);
        return false;
    }
    return true;
}

bool ResolveIoPaths(const std::string& inputDir, const std::string& outputDir, std::string& realInputDir,
                    std::string& realOutputDir) {
    if (!GetRealFilePath(inputDir, realInputDir)) {
        LOG_ERROR("resolve input path failed:" << inputDir);
        return false;
    }
    std::error_code ec;
    if (!std::filesystem::exists(realInputDir, ec)) {
        LOG_ERROR("input path not exists. file path:" << realInputDir);
        return false;
    }
    if (!GetRealFilePath(outputDir, realOutputDir)) {
        LOG_ERROR("resolve output path failed:" << outputDir);
        return false;
    }
    return EnsureDir(realOutputDir);
}

// Pre-create every per-field output subdirectory so the parallel per-segment builds never race.
bool PrepareOutputDirs(BuilderSchema& schema, const std::string& outputDir) {
    if (!EnsureDir(outputDir + "/" + ID_MAPPING_DIR)) {
        return false;
    }
    for (const auto& field : schema.GetInvertedFiled()) {
        if (!EnsureDir(outputDir + "/" + field.first)) {
            return false;
        }
    }
    for (const auto& field : schema.GetEmbeddingFiled()) {
        if (!EnsureDir(outputDir + "/" + field.first)) {
            return false;
        }
    }
    return true;
}

// docid mapping first: it assigns the ldocids the vector and posting builders depend on.
bool BuildSegment(uint32_t segIdx, BuilderSchema& schema, const std::string& dataDir, const std::string& outputDir,
                  uint32_t& segDocNum) {
    LOG_INFO("build segment:" << segIdx);
    const uint32_t segmentNum = schema.GetSegmentNum();
    const uint32_t docNumPerSeg = schema.GetDocNumPerSegment();

    IDMappingBuilder idMappingBuilder;
    if (!idMappingBuilder.Build(schema, dataDir, segIdx, outputDir)) {
        return false;
    }
    const std::unordered_map<uint64_t, uint32_t>& globalToLocal = idMappingBuilder.GetIdMapping();
    segDocNum = static_cast<uint32_t>(globalToLocal.size());

    // Every segment except the tail must be exactly full.
    if (segIdx < segmentNum - 1 && segDocNum != docNumPerSeg) {
        LOG_ERROR("segment doc num mismatch, segmentId:" << segIdx << " real docnum:" << segDocNum
                                                         << " doc num per seg in schema is:" << docNumPerSeg);
        return false;
    }

    VectorBuilder vectorBuilder;
    if (!vectorBuilder.Build(schema, dataDir, segIdx, globalToLocal, outputDir)) {
        return false;
    }
    InvertedBuilder invertedBuilder;
    if (!invertedBuilder.Build(schema, dataDir, segIdx, globalToLocal, outputDir)) {
        return false;
    }
    return true;
}

}  // namespace

/*
 * dataDir: root path of the input shard (or socket list in streaming mode)
 * indexOutputDir: root path of the output shard
 */
bool Build(const std::string& schema, uint32_t shardId, const std::string& dataDir, const std::string& indexOutputDir) {
    const int bthreadRet = bthread_setconcurrency(20);  // cap concurrent bthreads at 20
    if (bthreadRet != 0) {
        LOG_ERROR("Failed to set bthread concurrency, ret:" << bthreadRet);
        return false;
    }
    LOG_INFO("build start, shard is:" << shardId);

    BuilderSchema builderSchema;
    if (!builderSchema.Initialize(schema)) {
        LOG_ERROR("init schema fail");
        return false;
    }

    std::string realInputDir;
    std::string realOutputDir;
    if (FLAGS_build_from_memory) {
        if (!GetRealFilePath(indexOutputDir, realOutputDir) || !EnsureDir(realOutputDir)) {
            LOG_ERROR("prepare output directory failed: " << indexOutputDir);
            return false;
        }
    } else if (!ResolveIoPaths(dataDir, indexOutputDir, realInputDir, realOutputDir)) {
        LOG_ERROR("resolve io paths fail");
        return false;
    }

    if (!PrepareOutputDirs(builderSchema, realOutputDir)) {
        LOG_ERROR("PrepareOutputDirs failed");
        return false;
    }
    if (!LoadStreamingDataIfNeeded(dataDir)) {
        return false;
    }

    const std::string buildInputDir = FLAGS_build_from_memory ? "" : realInputDir;
    const uint32_t segmentNum = builderSchema.GetSegmentNum();

    std::shared_ptr<Executor> executor = CreateExecutor();
    LogContext logContext;
    auto asyncContext = executor->CreateExecuteContext(logContext);
    std::mutex docNumMutex;
    uint32_t totalDocNum = 0;
    for (uint32_t segIdx = 0; segIdx < segmentNum; segIdx++) {
        asyncContext->AddTask([&, segIdx]() -> ErrorCode::ResultType {
            uint32_t segDocNum = 0;
            if (!BuildSegment(segIdx, builderSchema, buildInputDir, realOutputDir, segDocNum)) {
                LOG_ERROR("segment[" << segIdx << "] build failed");
                return ErrorCode::ResultType::FAIL;
            }
            std::lock_guard<std::mutex> lock(docNumMutex);
            totalDocNum += segDocNum;
            return ErrorCode::ResultType::SUCCESS;
        });
    }

    bool ok = true;
    asyncContext->Wait([&ok](ErrorCode::ResultType ret) {
        if (ret != ErrorCode::ResultType::SUCCESS) {
            ok = false;
        }
    });

    if (FLAGS_build_from_memory) {
        MemoryDataManager::Instance()->Clear();
    }
    if (!ok) {
        LOG_ERROR("build failed");
        return false;
    }
    if (totalDocNum != builderSchema.GetDocNum()) {
        LOG_ERROR("doc num not equal, real is:" << totalDocNum
                                                << " doc num in schema is:" << builderSchema.GetDocNum());
        return false;
    }

    MetaBuilder metaBuilder;
    if (!metaBuilder.Build(builderSchema, realOutputDir)) {
        return false;
    }
    LOG_INFO("build end");
    return true;
}
}  // namespace NpuRetrieval
