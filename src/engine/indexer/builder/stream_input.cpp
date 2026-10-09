#include "stream_input.h"

#include <gflags/gflags.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "huawei_platform/huawei_secure_c/include/securec.h"
#include "src/full_recall/indexer/file/memory_data_manager.h"
#include "src/utils/logger.h"
#include "utils/rapidjson_util.h"

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

// Drain packets into MemoryDataManager until the peer sends an end marker or closes.
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
}  // namespace

// In streaming mode `dataDir` is a comma-separated list of socket paths.
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
}  // namespace NpuRetrieval
