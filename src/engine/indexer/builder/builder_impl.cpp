#include <gflags/gflags.h>
#include "builder_impl.h"
#include <atomic>
#include <chrono>
#include <filesystem>
#include <mutex>
#include <sstream>
#include <thread>
#include "huawei_platform/huawei_secure_c/include/securec.h"
#include "src/full_recall/indexer/builder/idmapping_builder.h"
#include "src/full_recall/indexer/builder/inverted_builder.h"
#include "src/full_recall/indexer/builder/meta_builder.h"
#include "src/full_recall/indexer/builder/vector_builder.h"
#include "src/full_recall/indexer/file/file_reader.h"
#include "src/full_recall/indexer/file/memory_data_manager.h"
#include "src/full_recall/indexer/schema/builder_schema.h"
#include "src/utils/file_manager.h"
#include "src/utils/logger.h"
#include "src/workflow/async/executor.h"
#include "utils/rapidjson_util.h"
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

DECLARE_bool(build_from_memory);

namespace NpuRetrieval {
namespace {
// JSON keys carried in each streaming-build packet header. These are a wire
// contract with the sender, so the string values must not change.
const std::string kFieldTypeKey = "field_type";
const std::string kFieldNameKey = "field_name";
const std::string kFileIndexKey = "file_index";
const std::string kDataSizeKey = "data_size";
const std::string kEndMarker = "end";
constexpr int kSocketBufferSize = 32 * 1024 * 1024;
constexpr int kConnectRetries = 10;

// Create dir (recursively) if it is not already present.
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

// Pre-create every per-field output subdirectory up front so that the parallel
// per-segment builds never race to create the same directory.
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

// Build one segment end to end: docid mapping first (it assigns the ldocids the
// vector/posting builders depend on), then the vector and posting indexes.
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

    // Every segment except the tail must be exactly full; only the last one may
    // hold fewer than docNumPerSeg documents.
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

// Read exactly `size` bytes from the socket, looping over short reads.
bool ReadFully(int socketFd, void* buffer, size_t size) {
    uint8_t* cursor = static_cast<uint8_t*>(buffer);
    size_t received = 0;
    while (received < size) {
        ssize_t n = recv(socketFd, cursor + received, size - received, MSG_WAITALL);
        if (n <= 0) {
            if (n == 0) {
                LOG_ERROR("Connection closed by peer");
            } else {
                LOG_ERROR("received failed: " << strerror(errno));
            }
            return false;
        }
        received += static_cast<size_t>(n);
    }
    return true;
}

// One packet = [u32 header size][header json][payload]. The header json names
// the field the payload belongs to and its byte length.
bool ReadPacket(int socketFd, std::string& fieldType, std::string& fieldName, uint32_t& fileIndex,
                std::vector<uint8_t>& payload) {
    uint32_t headerSize = 0;
    if (!ReadFully(socketFd, &headerSize, sizeof(headerSize))) {
        return false;
    }
    std::vector<char> headerBytes(headerSize);
    if (!ReadFully(socketFd, headerBytes.data(), headerSize)) {
        return false;
    }
    rapidjson::Document header;
    if (!ReadJsonFromString(std::string(headerBytes.begin(), headerBytes.end()), header)) {
        LOG_ERROR("Failed to parse header JSON");
        return false;
    }
    uint32_t dataSize = 0;
    if (!GetStringFromValueObj(header, fieldType, kFieldTypeKey) ||
        !GetStringFromValueObj(header, fieldName, kFieldNameKey) ||
        !GetUintFromValueObj(header, fileIndex, kFileIndexKey) ||
        !GetUintFromValueObj(header, dataSize, kDataSizeKey)) {
        LOG_ERROR("incomplete packet header");
        return false;
    }
    payload.resize(dataSize);
    return ReadFully(socketFd, payload.data(), dataSize);
}

// Connect to one IPC socket and drain packets into MemoryDataManager until the
// peer sends an end marker or closes. Errors are surfaced through `hasError`.
void ReceiveFromOneSocket(const std::string& socketPath, size_t socketIndex, std::atomic<bool>& hasError,
                          std::mutex& logMutex) {
    int clientSocket = socket(AF_UNIX, SOCK_STREAM, 0);
    if (clientSocket == -1) {
        std::lock_guard<std::mutex> lock(logMutex);
        LOG_ERROR("Socket " << socketIndex << " creation failed: " << strerror(errno));
        hasError = true;
        return;
    }
    int bufferSize = kSocketBufferSize;
    setsockopt(clientSocket, SOL_SOCKET, SO_RCVBUF, &bufferSize, sizeof(bufferSize));
    setsockopt(clientSocket, SOL_SOCKET, SO_SNDBUF, &bufferSize, sizeof(bufferSize));

    struct sockaddr_un serverAddr;
    memset_s(&serverAddr, sizeof(serverAddr), 0, sizeof(serverAddr));
    serverAddr.sun_family = AF_UNIX;
    strncpy_s(serverAddr.sun_path, sizeof(serverAddr.sun_path), socketPath.c_str(), sizeof(serverAddr.sun_path) - 1);

    bool connected = false;
    for (int retry = 0; retry < kConnectRetries; ++retry) {
        if (connect(clientSocket, reinterpret_cast<struct sockaddr*>(&serverAddr), sizeof(serverAddr)) == 0) {
            connected = true;
            std::lock_guard<std::mutex> lock(logMutex);
            LOG_INFO("Socket " << socketIndex << " connected successfully");
            break;
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    if (!connected) {
        {
            std::lock_guard<std::mutex> lock(logMutex);
            LOG_ERROR("Socket " << socketIndex << " connection failed: " << strerror(errno));
        }
        close(clientSocket);
        hasError = true;
        return;
    }

    try {
        uint64_t packetCount = 0;
        while (!hasError) {
            std::string fieldType;
            std::string fieldName;
            uint32_t fileIndex = 0;
            std::vector<uint8_t> payload;
            if (!ReadPacket(clientSocket, fieldType, fieldName, fileIndex, payload)) {
                std::lock_guard<std::mutex> lock(logMutex);
                LOG_INFO("Socket " << socketIndex << " reception ended, connection closed");
                break;
            }
            if (fieldType == kEndMarker) {
                std::lock_guard<std::mutex> lock(logMutex);
                LOG_INFO("Socket " << socketIndex << " received end marker");
                break;
            }
            MemoryDataManager::Instance()->AddFieldData(fieldName, fileIndex, payload);
            ++packetCount;
        }
        {
            std::lock_guard<std::mutex> lock(logMutex);
            LOG_INFO("Socket " << socketIndex << " completed, total packets: " << packetCount);
        }
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(logMutex);
        LOG_ERROR("Socket " << socketIndex << " exception: " << e.what());
        hasError = true;
    }
    close(clientSocket);
}

// Fan out one receiver thread per socket and join them all.
bool ReceiveFromSockets(const std::vector<std::string>& socketPaths) {
    LOG_INFO("Starting multi-socket streaming data reception");
    std::atomic<bool> hasError{false};
    std::mutex logMutex;
    std::vector<std::thread> workers;
    workers.reserve(socketPaths.size());
    for (size_t i = 0; i < socketPaths.size(); ++i) {
        workers.emplace_back(ReceiveFromOneSocket, std::cref(socketPaths[i]), i, std::ref(hasError),
                             std::ref(logMutex));
    }
    for (auto& worker : workers) {
        worker.join();
    }
    if (hasError) {
        LOG_ERROR("Some error occurred while receive segment data");
        return false;
    }
    LOG_INFO("All segment data reception completed successfully");
    return true;
}

// In streaming mode `dataDir` is a comma-separated list of socket paths; pull
// the whole dataset into memory before the segment builds start.
bool LoadStreamingDataIfNeeded(const std::string& dataDir) {
    if (!FLAGS_build_from_memory) {
        return true;
    }
    std::vector<std::string> socketPaths;
    std::istringstream ss(dataDir);
    std::string path;
    while (std::getline(ss, path, ',')) {
        socketPaths.push_back(path);
    }
    LOG_INFO("Starting memory data reception from " << socketPaths.size() << " sockets");
    if (!ReceiveFromSockets(socketPaths)) {
        LOG_ERROR("Failed to receive streaming data");
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

    // Resolve the output root (and, in file mode, validate the input root).
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
