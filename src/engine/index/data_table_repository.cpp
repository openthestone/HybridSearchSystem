#include "data_table_repository.h"
#include <unordered_set>
#include <iomanip>
#include "src/utils/logger.h"
#include "src/utils/string_util.h"
#include "src/utils/random_number.h"
#include "src/configuration/develop_configuration.h"
#include "src/common/error_code.h"
#include "src/workflow/async/executor.h"
#include "src/common/npuretrieval_constants.h"
#include "src/utils/alarm.h"
#ifdef ASCEND_C_ENABLED
#include "src/full_recall/retrieval/searcher/runtime/stream_manager.h"
#endif

namespace NpuRetrieval {

bool DataTableRepository::CreateTable(const std::string& deviceIdsStr, const std::string& indexLoadCardsStr,
                                      const std::string& tableName, const std::string& version,
                                      const std::string& dataDir) {
    std::vector<std::vector<int32_t>> indexLoadCards;
    auto tableInfo = std::make_shared<DataTableInfo>();
    if (!indexLoadCardsStr.empty()) {
        if (!ParseIndexLoadCards(indexLoadCardsStr, indexLoadCards) || indexLoadCards.empty()) {
            LOG_ERROR("invalid indexLoadCards:" << indexLoadCardsStr);
            return false;
        }
    } else {
        if (!ParseDeviceIds(deviceIdsStr, indexLoadCards) || indexLoadCards.empty()) {
            LOG_ERROR("No valid device_ids parsed from: " << deviceIdsStr);
            return false;
        }
    }
    tableInfo->shardNums = indexLoadCards[0].size();

    const auto& shardNum = tableInfo->shardNums;

    if (CheckTableLoaded(tableName, version)) {
        LOG_INFO("table:" << tableName << " version:" << version << " has been loaded previously.");
        return true;
    }
    const std::string tableKey = MakeTableKey(tableName, version);
    const size_t replicaCount = indexLoadCards.size();
    std::vector<std::vector<std::shared_ptr<DataTable>>> pending(
        shardNum, std::vector<std::shared_ptr<DataTable>>(replicaCount, nullptr));
    std::shared_ptr<Executor> executor = CreateExecutor();
    std::shared_ptr<LogContext> logContext = std::make_shared<LogContext>();
    auto asyncContext = executor->CreateExecuteContext(*logContext);

    // load every (shard, replica) DataTable concurrently
    for (int32_t n = 0; n < shardNum; ++n) {
        for (size_t i = 0; i < replicaCount; ++i) {
            int32_t deviceId = indexLoadCards[i][n];
#ifdef ASCEND_C_ENABLED
            if (!StreamManager::GetInstance()->Init(deviceId, FLAGS_full_recall_stream_init_size)) {
                LOG_ERROR("StreamManager::GetInstance()->Init failed, skip load dataTables, device id: " << deviceId);
                continue;
            }
#endif
            std::ostringstream oss;
            oss << dataDir << "/" << std::setfill('0') << std::setw(INDEX_SHARD_PREFIX_NUMS) << n;
            std::string shardDir = oss.str();
            LOG_INFO("Preparing to create DataTable: device=" << deviceId << ", shard=" << n << ", replica=" << i
                                                              << ", path=" << shardDir);
            auto loadTask = [this, deviceId, &tableKey, shardDir, &pending, n, i]() -> ErrorCode::ResultType {
                LOG_INFO("Creating DataTable: device=" << deviceId << ", shard=" << n << ", replica=" << i);
                std::shared_ptr<DataTable> dataTable = std::make_shared<DataTable>();
                if (!dataTable->LoadData(deviceId, shardDir)) {
                    LOG_ERROR("Failed to create DataTable: device=" << deviceId << ", shard=" << n
                                                                    << ", path=" << shardDir);
                    return ErrorCode::ResultType::FAIL;
                }
                pending[n][i] = dataTable;
                LOG_INFO("Successfully created DataTable: device=" << deviceId << ", shard=" << n
                                                                   << ",tableName=" << tableKey);
                return ErrorCode::ResultType::SUCCESS;
            };
            asyncContext->AddTask(loadTask);
        }
    }
    asyncContext->Wait();

    size_t minReplicasPerShard = SIZE_MAX;
    if (!CollectShardReplicas(pending, shardNum, replicaCount, minReplicasPerShard, tableInfo->dataTables)) {
        LOG_ERROR("Memory usage exceeded, create DataTable failed, table name is: " << tableName);
        return false;
    }
    if (!ValidateReplicaRate(replicaCount, minReplicasPerShard)) {
        LOG_ERROR("Failed to load index: " << tableName);
        return false;
    }
    {
        std::unique_lock<std::shared_mutex> lock(m_mutex);
        m_tables[tableKey] = std::move(tableInfo);
    }
    return true;
}

bool DataTableRepository::CollectShardReplicas(
    const std::vector<std::vector<std::shared_ptr<DataTable>>>& pendingTables, const int32_t shardNum,
    const size_t replicaCount, size_t& minReplicasPerShard,
    std::vector<std::vector<std::shared_ptr<DataTable>>>& dataTables) {
    for (int32_t n = 0; n < shardNum; ++n) {
        std::vector<std::shared_ptr<DataTable>> shardTables;
        shardTables.reserve(replicaCount);
        size_t loaded = 0;
        for (size_t i = 0; i < replicaCount; ++i) {
            if (!CheckDeviceMemory(pendingTables[n][i])) {
                return false;
            }
            if (pendingTables[n][i] != nullptr) {
                shardTables.emplace_back(pendingTables[n][i]);
                loaded++;
            }
        }
        if (loaded < minReplicasPerShard) {
            minReplicasPerShard = loaded;
        }
        if (!shardTables.empty()) {
            dataTables.emplace_back(std::move(shardTables));
        }
        LOG_INFO("Shard " << n << " has " << loaded << "/" << replicaCount << " replicas loaded");
    }
    return true;
}

bool DataTableRepository::ValidateReplicaRate(const size_t replicaCount, size_t& minReplicasPerShard) {
    const double successRate =
        replicaCount > 0 ? static_cast<double>(minReplicasPerShard) / static_cast<double>(replicaCount) : 0.0;
    if (minReplicasPerShard == 0) {
        LOG_ERROR("No devices initialized successfully for dataTables, total devices num: " << replicaCount);
        return false;
    }
    if (minReplicasPerShard >= replicaCount) {
        LOG_INFO("All devices initialized successfully for dataTables, total devices num: " << replicaCount);
        return true;
    }
    if (successRate < FLAGS_full_recall_device_load_rate) {
        LOG_ERROR("Failed to load enough devices, success rate: " << successRate << ", required: "
                                                                  << FLAGS_full_recall_device_load_rate);
        return false;
    }
    return true;
}

bool DataTableRepository::RemoveTable(const std::string& tableName, const std::string& version) {
    if (!CheckTableLoaded(tableName, version)) {
        LOG_INFO("table:" << tableName << " version:" << version << " has not been loaded previously.");
        return true;
    }
    {
        const std::string tableKey = MakeTableKey(tableName, version);
        std::unique_lock<std::shared_mutex> lock(m_mutex);
        m_tables.erase(tableKey);
    }
    return true;
}

bool DataTableRepository::RemoveUnusedTable(
    const std::unordered_map<std::string, std::vector<std::string>>& localTableNameMap) {
    std::unordered_set<std::string> localTables;
    localTables.reserve(localTableNameMap.size());
    for (const auto& [tableName, versions] : localTableNameMap) {
        for (const std::string& version : versions) {
            const std::string tableKey = MakeTableKey(tableName, version);
            localTables.emplace(tableKey);
        }
    }
    {
        std::unique_lock<std::shared_mutex> lock(m_mutex);
        for (auto iter = m_tables.begin(); iter != m_tables.end();) {
            if (localTables.find(iter->first) == localTables.end()) {
                LOG_INFO("table:" << iter->first << " will be removed.");
                iter = m_tables.erase(iter);
            } else {
                ++iter;
            }
        }
    }
    return true;
}

std::shared_ptr<DataTable> DataTableRepository::GetDataTable(const std::string& tableName, const std::string& version,
                                                             uint32_t shardId) {
    const std::string tableKey = MakeTableKey(tableName, version);
    return GetDataTable(tableKey, shardId);
}

std::shared_ptr<DataTable> DataTableRepository::GetDataTable(const std::string& tableKey, uint32_t shardId) {
    std::shared_lock<std::shared_mutex> lock(m_mutex);
    auto iter = m_tables.find(tableKey);
    if (iter == m_tables.end()) {
        return nullptr;
    }
    const std::vector<std::vector<std::shared_ptr<DataTable>>>& dataTables = iter->second->dataTables;
    if (dataTables.empty()) {
        LOG_ERROR("No DataTable available for index: " << tableKey);
        return nullptr;
    }
    if (shardId >= dataTables.size()) {
        LOG_ERROR("shardId is illegal indexName: " << tableKey << ", shardId:" << shardId << ", dataTables size "
                                                   << dataTables.size());
        return nullptr;
    }
    const auto& replicas = dataTables[shardId];
    if (replicas.empty()) {
        LOG_ERROR("No replicas for shard " << shardId << " of indexName: " << tableKey);
        return nullptr;
    }
    if (replicas.size() == 1) {
        return replicas[0];
    }
    uint64_t counter = iter->second->queryCounter->fetch_add(1);
    const uint32_t replicaIdx = counter % replicas.size();
    LOG_DEBUG("Selected replica " << replicaIdx << "/" << replicas.size() << " for shard " << shardId);
    return replicas[replicaIdx];
}

bool DataTableRepository::CheckTableLoaded(const std::string& tableName, const std::string& version) {
    const std::string tableKey = MakeTableKey(tableName, version);
    std::shared_lock<std::shared_mutex> lock(m_mutex);
    auto ptr = m_tables.find(tableKey);
    if (ptr != m_tables.end() && (!ptr->second->dataTables.empty())) {
        return true;
    }
    return false;
}

bool DataTableRepository::CheckDeviceMemory(const std::shared_ptr<DataTable>& dataTablePtr) {
    if (dataTablePtr == nullptr) {
        return true;
    }
#ifdef ASCEND_C_ENABLED
    int32_t deviceId = dataTablePtr->GetDeviceId();
    auto ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("aclrtSetDevice fail, deviceId is:" << deviceId << " error code is:" << ret);
        return false;
    }
    size_t freeMem = 0;
    size_t totalMem = 0;
    aclrtMemAttr attr = ACL_HBM_MEM;
    ret = aclrtGetMemInfo(attr, &freeMem, &totalMem);
    if (ret == ACL_SUCCESS) {
        size_t usedMem = totalMem - freeMem;
        double usageRate = 0.0;
        if (totalMem > 0) {
            usageRate = static_cast<double>(usedMem) / static_cast<double>(totalMem);
        }
        double totalMemMB = static_cast<double>(totalMem) / 1024 / 1024;
        double usedMemMB = static_cast<double>(usedMem) / 1024 / 1024;
        LOG_INFO("device id " << deviceId << ", usedMemMB is: " << usedMemMB << ", totalMemMB is: " << totalMemMB);
        if (usageRate > NPU_MEMORY_WARNING_THRESHOLD) {
            std::stringstream ss;
            uint32_t precisionBit = 2;
            ss << "device id is: " << deviceId << " memory usage is high: " << std::fixed
               << std::setprecision(precisionBit) << usageRate << " threshold is:" << NPU_MEMORY_WARNING_THRESHOLD;
            std::string alarmMsg = ss.str();
            LOG_ERROR("NPU device memory exceed threshold, alarm msg is: " << alarmMsg);
            SendAlarmHcm(AlarmLevel::MAJOR, "NPU device memory exceed threshold", alarmMsg);
            CHECK_ACL_ONLY_LOG(aclrtResetDevice(deviceId));
            return false;
        }
    } else {
        LOG_ERROR("aclrtGetMemInfo fail, deviceId is:" << deviceId << " error code is:" << ret);
        CHECK_ACL_ONLY_LOG(aclrtResetDevice(deviceId));
        return false;
    }
    CHECK_ACL_ONLY_LOG(aclrtResetDevice(deviceId));
    return true;
#endif
    return true;
}
}  // namespace NpuRetrieval
