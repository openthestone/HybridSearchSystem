#pragma once
#include <shared_mutex>
#include "src/common/uncopyable.h"
#ifdef ASCEND_C_ENABLED
#include "src/full_recall/index/data_table.h"
#else
#include "src/full_recall/index/mock_data_table.h"
#endif
namespace NpuRetrieval {

struct DataTableInfo {
    int32_t shardNums{1};
    std::vector<std::vector<std::shared_ptr<DataTable>>> dataTables;
    // query counter, used for round-robin replica selection across cards
    std::shared_ptr<std::atomic<uint64_t>> queryCounter = std::make_shared<std::atomic<uint64_t>>(0);
};

class DataTableRepository {
   public:
    static DataTableRepository* Instance() {
        static DataTableRepository value;
        return &value;
    }

    bool CreateTable(const std::string& deviceIdsStr, const std::string& indexLoadCardsStr,
                     const std::string& tableName, const std::string& version, const std::string& dataDir);
    bool RemoveTable(const std::string& tableName, const std::string& version);
    bool RemoveUnusedTable(const std::unordered_map<std::string, std::vector<std::string>>& localTableNameMap);
    bool CheckTableLoaded(const std::string& tableName, const std::string& version);

    std::shared_ptr<DataTable> GetDataTable(const std::string& tableName, const std::string& version, uint32_t shardId);
    std::shared_ptr<DataTable> GetDataTable(const std::string& tableKey, uint32_t shardId);
    bool CheckDeviceMemory(const std::shared_ptr<DataTable>& dataTablePtr);
    NPURETRIEVAL_DECLARE_UNCOPYABLE(DataTableRepository);

   private:
    DataTableRepository() {};
    ~DataTableRepository() = default;

    std::string MakeTableKey(const std::string& indexName, const std::string& version) {
        return indexName + "-" + version;
    }

    bool CollectShardReplicas(const std::vector<std::vector<std::shared_ptr<DataTable>>>& pendingTables,
                              const int32_t shardNum, const size_t totalDevices, size_t& minReplicasPerShard,
                              std::vector<std::vector<std::shared_ptr<DataTable>>>& dataTables);
    bool ValidateReplicaRate(const size_t totalDevices, size_t& minReplicasPerShard);

   private:
    // key: tableName-tableVersion
    std::unordered_map<std::string, std::shared_ptr<DataTableInfo>> m_tables{};
    // guards m_tables
    std::shared_mutex m_mutex;
};
}  // namespace NpuRetrieval
