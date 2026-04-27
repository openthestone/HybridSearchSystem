#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "utils/RunSupport.h"
#include "Clustering/run_clustering.h"
#include "DataBaseCPU/DataBaseCPU.h"
#include "utils/DataReader.h"
#include "Query/Query.h"
#include "Schedule/Scheduler.h"
#include "Schedule/WorkerGroup.h"
#include "Schedule/ThreadUtils.h"

namespace fs = RunSupport::fs;

namespace
{

// ---- NUMA-aware core mapping ----
// Auto-detect NPU NUMA affinity and build a core list that places each
// device's worker groups on CPU cores that are local to that device's NUMA
// node.  Falls back to sequential cores 0..N-1 when detection fails.
std::vector<int> BuildNumaAwareCoreMap()
{
    // Gather local CPU core lists per logical device.
    // Ascend logical device IDs map to physical /dev/davinci<id> in ascending order.
    struct DeviceNumaInfo
    {
        int logical_dev_id;
        int numa_node;
        std::vector<int> local_cores;
    };

    std::vector<DeviceNumaInfo> dev_infos;
    dev_infos.reserve(static_cast<size_t>(npu_device_count));

    // Enumerate available davinci devices to get physical IDs.
    // Physical IDs are sorted so that logical device 0 = smallest physical ID, etc.
    std::vector<int> physical_ids;
    for (int pid = 0; pid < 256; ++pid)
    {
        std::string dev = "/dev/davinci" + std::to_string(pid);
        if (access(dev.c_str(), F_OK) == 0)
        {
            physical_ids.push_back(pid);
        }
    }

    if (physical_ids.size() < static_cast<size_t>(npu_device_count))
    {
        std::cerr << "[NUMA] Cannot enumerate enough davinci devices, falling back to default cores.\n";
        return {};
    }

    // Sort to get consistent logical -> physical mapping.
    std::sort(physical_ids.begin(), physical_ids.end());

    for (int d = npu_device_id_start; d < npu_device_id_start + npu_device_count; ++d)
    {
        DeviceNumaInfo info;
        info.logical_dev_id = d;
        int physical_id = physical_ids[static_cast<size_t>(d)];

        // Find PCI bus via npu-smi.
        std::string cmd = "npu-smi info -t board -i " + std::to_string(physical_id) + " 2>/dev/null";
        FILE *pipe = popen(cmd.c_str(), "r");
        std::string pci_bus;
        if (pipe)
        {
            char buf[512];
            while (fgets(buf, sizeof(buf), pipe))
            {
                std::string line(buf);
                auto pos = line.find("PCIe Bus Info");
                if (pos != std::string::npos)
                {
                    // Format: "  PCIe Bus Info                  : 0000:C2:00.0"
                    // Find the separator " : " after the label.
                    auto sep = line.find(" : ", pos);
                    if (sep != std::string::npos)
                    {
                        pci_bus = line.substr(sep + 3);
                        // Trim trailing whitespace / newline
                        while (!pci_bus.empty() && (pci_bus.back() == ' ' || pci_bus.back() == '\t' || pci_bus.back() == '\n' || pci_bus.back() == '\r'))
                            pci_bus.pop_back();
                        // sysfs paths use lowercase hex
                        std::transform(pci_bus.begin(), pci_bus.end(), pci_bus.begin(),
                                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                    }
                    break;
                }
            }
            pclose(pipe);
        }

        if (pci_bus.empty())
        {
            std::cerr << "[NUMA] Cannot find PCI bus for NPU " << physical_id << ", falling back.\n";
            return {};
        }

        // Read NUMA node.
        std::string numa_path = "/sys/bus/pci/devices/" + pci_bus + "/numa_node";
        std::ifstream nf(numa_path);
        if (!nf.is_open() || !(nf >> info.numa_node) || info.numa_node < 0)
        {
            std::cerr << "[NUMA] Cannot read NUMA node for " << pci_bus << ", falling back.\n";
            return {};
        }

        // Read local CPU list (format: "start-end" or "start,end,...").
        std::string cpu_path = "/sys/bus/pci/devices/" + pci_bus + "/local_cpulist";
        std::ifstream cf(cpu_path);
        std::string cpu_list_str;
        if (!cf.is_open() || !std::getline(cf, cpu_list_str) || cpu_list_str.empty())
        {
            std::cerr << "[NUMA] Cannot read local CPUs for " << pci_bus << ", falling back.\n";
            return {};
        }

        // Parse cpulist (e.g. "144-167" or "0-23,48-71").
        {
            std::string token;
            std::istringstream ss(cpu_list_str);
            while (std::getline(ss, token, ','))
            {
                auto dash = token.find('-');
                if (dash != std::string::npos)
                {
                    int start = std::stoi(token.substr(0, dash));
                    int end = std::stoi(token.substr(dash + 1));
                    for (int c = start; c <= end; ++c)
                        info.local_cores.push_back(c);
                }
                else
                {
                    info.local_cores.push_back(std::stoi(token));
                }
            }
        }

        std::cout << "[NUMA] Device " << d << " (physical NPU " << physical_id
                  << ", PCI " << pci_bus << ") -> NUMA " << info.numa_node
                  << ", local_cores=" << info.local_cores.front() << "-" << info.local_cores.back()
                  << " (" << info.local_cores.size() << " cores)\n";
        dev_infos.push_back(std::move(info));
    }

    // Build the core map: for each device, allocate groups_per_device groups,
    // each consuming cores_per_group cores from the device's local core list.
    const int cores_needed_per_device = groups_per_device * cores_per_group;
    std::vector<int> result;
    result.reserve(static_cast<size_t>(group_count * cores_per_group));

    for (const auto &di : dev_infos)
    {
        if (static_cast<int>(di.local_cores.size()) < cores_needed_per_device)
        {
            std::cerr << "[NUMA] Device " << di.logical_dev_id << " has only "
                      << di.local_cores.size() << " local cores, need " << cores_needed_per_device
                      << ". Falling back.\n";
            return {};
        }
        for (int i = 0; i < cores_needed_per_device; ++i)
        {
            result.push_back(di.local_cores[static_cast<size_t>(i)]);
        }
    }

    return result;
}


bool LoadPreparedQueriesFromFvec(const std::vector<std::string> &query_paths,
                                 int expected_dim,
                                 int default_top_k,
                                 std::vector<DataReader::PreparedQuery> &queries,
                                 std::string &loaded_path)
{
    queries.clear();
    loaded_path.clear();

    if (expected_dim < 64 || default_top_k <= 0)
    {
        return false;
    }

    auto append_query = [&](const float *raw, int source_dim, size_t line_no) {
        DataReader::PreparedQuery q;
        q.query_vec.assign(static_cast<size_t>(expected_dim), 0.0f);
        const int copy_dim = std::min<int>(64, std::min<int>(expected_dim, source_dim));
        for (int d = 0; d < copy_dim; ++d)
        {
            q.query_vec[static_cast<size_t>(d)] = raw[static_cast<size_t>(d)];
        }
        q.filter_expr = "";
        q.top_k = default_top_k;
        q.line_no = line_no;
        queries.push_back(std::move(q));
    };

    for (const auto &path : query_paths)
    {
        std::ifstream in(path, std::ios::binary);
        if (!in)
        {
            continue;
        }

        in.seekg(0, std::ios::end);
        const std::streamoff file_size = in.tellg();
        in.seekg(0, std::ios::beg);
        if (file_size < static_cast<std::streamoff>(sizeof(int32_t) * 2))
        {
            continue;
        }

        int32_t first = 0;
        int32_t second = 0;
        in.read(reinterpret_cast<char *>(&first), sizeof(int32_t));
        in.read(reinterpret_cast<char *>(&second), sizeof(int32_t));
        if (!in)
        {
            continue;
        }

        const std::streamoff headered_bytes = static_cast<std::streamoff>(sizeof(int32_t) * 2) +
                                              static_cast<std::streamoff>(first) * static_cast<std::streamoff>(second) *
                                                  static_cast<std::streamoff>(sizeof(float));

        if (first > 0 && second >= 64 && file_size == headered_bytes)
        {
            const int32_t rows = first;
            const int32_t source_dim = second;
            std::vector<float> raw(static_cast<size_t>(source_dim));
            queries.clear();
            queries.reserve(static_cast<size_t>(rows));
            for (int32_t i = 0; i < rows; ++i)
            {
                in.read(reinterpret_cast<char *>(raw.data()),
                        static_cast<std::streamsize>(raw.size() * sizeof(float)));
                if (!in)
                {
                    queries.clear();
                    break;
                }
                append_query(raw.data(), source_dim, static_cast<size_t>(i + 1));
            }
            if (!queries.empty())
            {
                loaded_path = path;
                return true;
            }
            continue;
        }

        in.clear();
        in.seekg(0, std::ios::beg);
        queries.clear();
        size_t line_no = 0;
        while (true)
        {
            int32_t dim = 0;
            in.read(reinterpret_cast<char *>(&dim), sizeof(int32_t));
            if (!in)
            {
                break;
            }
            if (dim < 64)
            {
                queries.clear();
                break;
            }

            std::vector<float> raw(static_cast<size_t>(dim));
            in.read(reinterpret_cast<char *>(raw.data()),
                    static_cast<std::streamsize>(raw.size() * sizeof(float)));
            if (!in)
            {
                queries.clear();
                break;
            }

            ++line_no;
            append_query(raw.data(), dim, line_no);
        }

        if (!queries.empty())
        {
            loaded_path = path;
            return true;
        }
    }

    return false;
}

bool CollectParallelLatencyMetrics(const std::vector<std::unique_ptr<Query>> &queries,
                                   std::vector<double> &total_end_to_end_ms)
{
    total_end_to_end_ms.resize(queries.size());
    for (size_t i = 0; i < queries.size(); ++i)
    {
        const Query &query = *queries[i];
        if (query.start_time_for_parallel_record == std::chrono::steady_clock::time_point{})
        {
            std::cerr << "[Parallel] Missing start_time_for_parallel_record for query line "
                      << query.query_id << "\n";
            return false;
        }
        if (query.end_time_for_parallel_record == std::chrono::steady_clock::time_point{})
        {
            std::cerr << "[Parallel] Missing end_time_for_parallel_record for query line "
                      << query.query_id << "\n";
            return false;
        }
        if (query.end_time_for_parallel_record < query.start_time_for_parallel_record)
        {
            std::cerr << "[Parallel] Invalid parallel timing order for query line "
                      << query.query_id << "\n";
            return false;
        }

        total_end_to_end_ms[i] =
            std::chrono::duration<double, std::milli>(query.end_time_for_parallel_record -
                                                      query.start_time_for_parallel_record)
                .count();
    }

    return true;
}

bool CollectParallelRecallMetrics(const InputDataset &dataset,
                                  const std::vector<DataReader::PreparedQuery> &prepared_queries,
                                  std::vector<std::unique_ptr<Query>> &queries,
                                  DataReader::GroundTruthCache *ground_truth_cache,
                                  std::vector<double> &recall_rate_percent)
{
    if (queries.size() != prepared_queries.size())
    {
        std::cerr << "[Parallel] Query count mismatch when collecting recall metrics. prepared="
                  << prepared_queries.size() << ", runtime=" << queries.size() << "\n";
        return false;
    }

    recall_rate_percent.resize(queries.size());
    for (size_t i = 0; i < queries.size(); ++i)
    {
        Query &query = *queries[i];
        const auto &prepared = prepared_queries[i];

        std::vector<DataReader::GroundTruthResultItem> cached_ground_truth;
        if (ground_truth_cache &&
            ground_truth_cache->Lookup(prepared.query_vec,
                                       prepared.filter_expr,
                                       prepared.top_k,
                                       cached_ground_truth))
        {
            query.ground_truth_results = RunSupport::ConvertCacheItemsToQueryResults(cached_ground_truth);
        }
        else
        {
            query.brute_force_search(dataset);
            if (ground_truth_cache &&
                !ground_truth_cache->Store(prepared.query_vec,
                                           prepared.filter_expr,
                                           prepared.top_k,
                                           RunSupport::ConvertQueryResultsToCacheItems(query.ground_truth_results)))
            {
                std::cerr << "[Warn] Failed to append ground-truth cache entry for query line "
                          << prepared.line_no << ".\n";
            }
        }

        query.calculate_recall();
        recall_rate_percent[i] = static_cast<double>(query.recall_rate) * 100.0;
    }

    return true;
}

bool FlushParallelOutputs(const fs::path &result_root,
                          const std::vector<double> &total_end_to_end_ms,
                          const std::vector<double> &recall_rate_percent,
                          double overall_qps)
{
    const fs::path recall_dir = result_root / "recall";
    const fs::path latency_dir = result_root / "latency";
    const fs::path log_dir = result_root / "log";
    const std::vector<double> qps_values{overall_qps};
    return RunSupport::EnsureOutputDirectory(recall_dir) &&
           RunSupport::EnsureOutputDirectory(latency_dir) &&
           RunSupport::EnsureOutputDirectory(log_dir) &&
           RunSupport::WriteMetricFile(recall_dir, "recall_rate_percent.txt", recall_rate_percent) &&
           RunSupport::WriteMetricFile(latency_dir, "total_end_to_end_ms.txt", total_end_to_end_ms) &&
           RunSupport::WriteMetricFile(log_dir, "QPS.txt", qps_values);
}

void PrintParallelProgress(size_t completed_count,
                           size_t total_count)
{
    std::cout << "[Progress] " << completed_count << "/" << total_count << std::endl;
}
} // namespace

int main()
{
    constexpr const char *kDatasetFile = "../../dataset_HW.bin";
    constexpr int kDefaultTopK = 100;
    constexpr int QueryNum = 10000;
    const std::vector<std::string> kQueryPaths = {
        "datasets/hw_queries.fvecs",
        "../datasets/hw_queries.fvecs",
        "../../datasets/hw_queries.fvecs",
        "../../../datasets/hw_queries.fvecs"
    };

    const fs::path config_path = RunSupport::ResolveConfigPath();
    std::cout << "[System] Loading config from " << config_path << "...\n";
    if (!LoadParams(config_path.string(), ResourceConfigProfile::Parallel))
    {
        std::cerr << "[Error] Failed to load config: " << config_path << "\n";
        return -1;
    }
    const fs::path query_result_root_dir =
        query_result_root.empty() ? fs::path(".") : fs::path(query_result_root);
    const fs::path memory_log_file = query_result_root_dir / "log" / "memory_logs.txt";

    DataReader::DatasetBuffers dataset_buffers;
    if (!DataReader::LoadDatasetCache(kDatasetFile, dataset_buffers))
    {
        std::cerr << "[Fatal] Failed to load dataset cache: " << kDatasetFile << "\n";
        return -1;
    }
    std::cout << "[Loader] Dataset loaded. Docs=" << total_doc_num
              << ", Dim=" << vector_dim
              << ", Tags=" << total_tag_num
              << ", Buckets(Level1)=" << total_bucket_num_level_1 << "\n";

    std::vector<DataReader::PreparedQuery> prepared_queries;
    std::string loaded_query_path;
    if (!LoadPreparedQueriesFromFvec(kQueryPaths,
                                     vector_dim,
                                     kDefaultTopK,
                                     prepared_queries,
                                     loaded_query_path))
    {
        std::cerr << "[Fatal] Failed to read queries from deep1B_queries.fvecs\n";
        return -1;
    }
    std::cout << "[Loader] Queries loaded from " << loaded_query_path << "\n";

    if (prepared_queries.empty())
    {
        std::cout << "[Info] No valid query found in " << loaded_query_path << ". Exit with code 0.\n";
        return 0;
    }

    std::cout << "[Loader] Prepared query count: " << prepared_queries.size() << "\n";

    // Step 1: Ensure queries vector has exactly QueryNum items
    if (prepared_queries.size() > QueryNum) {
        prepared_queries.resize(QueryNum);
        std::cout << "[Loader] Truncated queries: " << prepared_queries.size() << "\n";
    } else if (prepared_queries.size() < QueryNum) {
        const size_t original_count = prepared_queries.size();
        prepared_queries.reserve(QueryNum);
        for (size_t i = original_count; i < QueryNum; ++i) {
            prepared_queries.push_back(prepared_queries[i % original_count]);
        }
        std::cout << "[Loader] Expanded queries: " << original_count << " -> " << prepared_queries.size() << "\n";
    }

    // Step 2: Load filter expressions: read 10 from file, replicate to QueryNum
    {
        std::vector<std::string> filter_exprs_10;
        const fs::path filter_expr_path = config_path.has_parent_path()
                                              ? (config_path.parent_path() / "filter_expr_example.txt")
                                              : fs::path("filter_expr_example.txt");
        {
            std::ifstream fexpr_file(filter_expr_path);
            std::string line;
            while (std::getline(fexpr_file, line)) {
                if (!line.empty()) {
                    filter_exprs_10.push_back(line);
                }
            }
        }
        if (filter_exprs_10.empty()) {
            std::cerr << "[Warn] filter_expr_example.txt not found or empty at " << filter_expr_path
                      << ". Using empty filters for all queries.\n";
        } else if (filter_exprs_10.size() < 10) {
            std::cerr << "[Warn] filter_expr_example.txt has fewer than 10 lines. Using available expressions.\n";
        }

        std::vector<std::string> filter_exprs_N;
        filter_exprs_N.reserve(QueryNum);
        if (!filter_exprs_10.empty()) {
            std::mt19937 rng(42);
            std::uniform_int_distribution<size_t> dist(0, filter_exprs_10.size() - 1);
            for (size_t i = 0; i < QueryNum; ++i) {
                filter_exprs_N.push_back(filter_exprs_10[dist(rng)]);
            }
        } else {
            filter_exprs_N.assign(QueryNum, std::string());
        }

        for (size_t i = 0; i < prepared_queries.size(); ++i) {
            prepared_queries[i].filter_expr = filter_exprs_N[i];
            // prepared_queries[i].filter_expr = "";
        }
        std::cout << "[Loader] Assigned filter expressions to all " << prepared_queries.size() << " queries\n";
    }

    if (!RunSupport::ValidatePreparedQueriesAgainstPrealloc(prepared_queries))
    {
        return -1;
    }
    FilterExpCompiler::WarmUpThreadLocalBuffers();

    MemoryEventLogCollector memory_log_collector(memory_events_log_enable != 0);
    if (memory_events_log_enable != 0)
    {
        memory_log_collector.Reserve(prepared_queries.size() * 8);
    }

    std::unique_ptr<DataReader::GroundTruthCache> ground_truth_cache;
    if (!ground_truth_cache_file.empty())
    {
        std::cout << "[Loader] Loading ground-truth cache...\n";
        ground_truth_cache = std::make_unique<DataReader::GroundTruthCache>(ground_truth_cache_file);
        if (!ground_truth_cache->Load())
        {
            std::cerr << "[Warn] Failed to load ground-truth cache from "
                      << ground_truth_cache->path()
                      << ". Continue without cache.\n";
            ground_truth_cache.reset();
        }
        else
        {
            std::cout << "[Loader] Ground-truth cache loaded from "
                      << ground_truth_cache->path()
                      << ". entries=" << ground_truth_cache->EntryCount() << "\n";
        }
    }
    else
    {
        std::cout << "[Loader] Ground-truth cache disabled by config.\n";
    }

    std::vector<uint64_t> bucket_doc_offsets;
    std::vector<uint32_t> bucket_doc_ids;
    int sync_result = SyncClusteringContextFromMain(dataset_buffers.vectors,
                                                    dataset_buffers.bitmaps,
                                                    bucket_doc_offsets,
                                                    bucket_doc_ids,
                                                    dataset_buffers.centroids,
                                                    total_bucket_num_level_1,
                                                    total_tag_num,
                                                    vector_dim,
                                                    total_doc_num,
                                                    max_doc_per_bucket_level_2);
    if (sync_result != 0)
    {
        std::cerr << "[Fatal] Failed to run or load clustering outputs (centroids.bin/buckets).\n";
        return -1;
    }

    if (bucket_doc_offsets.size() < 2)
    {
        std::cerr << "[Fatal] Invalid clustering output: bucket_doc_offsets is empty.\n";
        return -1;
    }

    const size_t clustered_bucket_count = bucket_doc_offsets.size() - 1;
    if (clustered_bucket_count != static_cast<size_t>(total_bucket_num_level_1))
    {
        std::cerr << "[Fatal] Invalid clustering output: config total_bucket_num_level_1="
                  << total_bucket_num_level_1
                  << ", but clustering produced "
                  << clustered_bucket_count
                  << " level-1 buckets.\n";
        return -1;
    }

    if (dataset_buffers.centroids.size() !=
        static_cast<size_t>(total_bucket_num_level_1) * static_cast<size_t>(vector_dim))
    {
        std::cerr << "[Fatal] Invalid clustering output: centroids size mismatch.\n";
        return -1;
    }

    InputDataset dataset = DataReader::BuildInputDataset(dataset_buffers);
    BucketDocTable level_1_bucket_doc_table;
    try
    {
        level_1_bucket_doc_table =
            RunSupport::BuildBucketDocTableFromOffsets(bucket_doc_offsets, bucket_doc_ids);
    }
    catch (const std::exception &e)
    {
        std::cerr << "[Fatal] Failed to build BucketDocTable from clustering output: "
                  << e.what() << "\n";
        return -1;
    }

    TwoLevelBucketLayout bucket_layout;
    try
    {
        bucket_layout = RunSupport::BuildTwoLevelBucketLayout(std::move(level_1_bucket_doc_table),
                                                              max_doc_per_bucket_level_2);
    }
    catch (const std::exception &e)
    {
        std::cerr << "[Fatal] Failed to adapt clustering output into two-level buckets: "
                  << e.what() << "\n";
        return -1;
    }

    total_bucket_num_level_2 = static_cast<int>(bucket_layout.Level2BucketCount());
    if (!FinalizePreallocationParams())
    {
        return -1;
    }

    std::cout << "[Loader] Two-level bucket layout built. StoredDocRefs="
              << RunSupport::CountStoredDocRefs(bucket_layout.level_1_bucket_doc_table)
              << ", Level1BucketCount=" << bucket_layout.Level1BucketCount()
              << ", Level2BucketCount=" << bucket_layout.Level2BucketCount() << "\n";

    std::cout << "[System] Initializing DataBaseCPU...\n";
    DataBaseCPU db(dataset, std::move(bucket_layout));

    const size_t scheduler_capacity = prepared_queries.size() + 10000;
    std::cout << "[System] Initializing Scheduler. capacity=" << scheduler_capacity << "\n";
    Scheduler scheduler(scheduler_capacity);

    std::cout << "[System] Launching Worker Threads...\n";

    // NUMA-aware core mapping is disabled: pinning worker threads to NPU-local
    // NUMA cores (144-167, 96-119) makes all data structures allocated by the
    // main thread on NUMA 0 remote, which drops QPS from ~1775 to ~727.
    // g_numa_aware_cores stays empty so BindThreadToCore uses logical core IDs
    // directly (cores 0-47), keeping data access fast on NUMA 0/1.
    // g_numa_aware_cores = BuildNumaAwareCoreMap();
    std::cout << "[NUMA] Using default sequential cores 0.." << (cpu_core_count - 1) << "\n";

    // Bind the main thread to a core outside the worker thread range (48+)
    // to avoid L3 cache and scheduler interference with worker threads on NUMA 0/1.
    {
        cpu_set_t main_cpuset;
        CPU_ZERO(&main_cpuset);
        CPU_SET(48, &main_cpuset);
        int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &main_cpuset);
        if (rc != 0)
        {
            std::cerr << "[NUMA] Warning: failed to bind main thread to core 48: " << rc << "\n";
        }
        else
        {
            std::cout << "[NUMA] Main thread pinned to core 48 (outside worker range 0-47)\n";
        }
    }

    std::vector<std::unique_ptr<WorkerGroup>> groups;
    std::vector<std::thread> threads;
    groups.reserve(group_count);
    threads.reserve(cpu_core_count);

    for (int i = 0; i < group_count; ++i)
    {
        groups.push_back(std::make_unique<WorkerGroup>(i, &db, &scheduler));
    }
    for (int i = 0; i < cpu_core_count; ++i)
    {
        const int gid = GetGroupId(i);
        threads.emplace_back(&WorkerGroup::Run, groups[gid].get(), i);
    }

    std::vector<std::unique_ptr<Query>> queries;
    queries.reserve(prepared_queries.size());
    std::vector<Query *> query_ptrs;
    query_ptrs.reserve(prepared_queries.size());
    for (const auto &prepared : prepared_queries)
    {
        auto query = std::make_unique<Query>();
        query->PrepareMemoryEventSession(prepared.line_no, memory_events_log_enable != 0);
        {
            ScopedMemoryEventSession memory_event_scope(query->memory_event_session());
            query->Reset(prepared.query_vec, prepared.filter_expr, prepared.top_k, prepared.line_no);
        }
        query_ptrs.push_back(query.get());
        queries.push_back(std::move(query));
    }

    std::cout << std::fixed << std::setprecision(5);
    std::cout << "[ParallelTest] Query objects prepared. count=" << query_ptrs.size() << "\n";
    std::cout << "[ParallelTest] Batch submit begins. target_qps=" << target_qps_parallel << "\n";

    const double target_qps = target_qps_parallel;
    const auto batch_start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < query_ptrs.size(); ++i)
    {
        const auto scheduled_submit_time =
            batch_start + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                              std::chrono::duration<double>(static_cast<double>(i) / target_qps));
        std::this_thread::sleep_until(scheduled_submit_time);
        scheduler.Push(query_ptrs[i]);
    }

    size_t completed_count = 0;
    auto next_progress_report = batch_start + std::chrono::seconds(1);
    while (completed_count < query_ptrs.size())
    {
        scheduler.PopResult();  // 阻塞等待，无 CPU 轮询
        ++completed_count;

        const auto now = std::chrono::steady_clock::now();
        if (now >= next_progress_report)
        {
            PrintParallelProgress(completed_count, query_ptrs.size());
            do
            {
                next_progress_report += std::chrono::seconds(1);
            } while (now >= next_progress_report);
        }
    }
    const auto batch_end = std::chrono::steady_clock::now();

    std::chrono::duration<double, std::milli> total_latency_ms = batch_end - batch_start;
    const double total_latency_seconds = total_latency_ms.count() / 1000.0;
    const double overall_qps = (total_latency_seconds > 0.0)
                                   ? (static_cast<double>(query_ptrs.size()) / total_latency_seconds)
                                   : 0.0;
    const double avg_query_latency_ms = query_ptrs.empty()
                                            ? 0.0
                                            : (total_latency_ms.count() / static_cast<double>(query_ptrs.size()));

    // ---- Aggregate per-query pipeline timing ----
    double agg_ivf = 0, agg_merge = 0, agg_attr = 0, agg_collect = 0, agg_final_merge = 0;
    int timed_count = 0;
    for (const auto &q : query_ptrs)
    {
        const auto &t = q->timing_metrics;
        if (t.inbucket_attr_filter_overlapped_ms > 0 || t.bucket_level_ivf_ms > 0)
        {
            agg_ivf += t.bucket_level_ivf_ms;
            agg_merge += t.candidate_bucket_merge_ms;
            agg_attr += t.inbucket_attr_filter_overlapped_ms;
            agg_collect += t.result_collection_ms;
            agg_final_merge += t.final_merge_ms;
            ++timed_count;
        }
    }

    std::cout << "\n========================================================\n";
    std::cout << "                Parallel Batch Test Result              \n";
    std::cout << "========================================================\n";
    std::cout << "Query Count                    : " << query_ptrs.size() << "\n";
    std::cout << "Completed Query Count          : " << completed_count << "\n";
    std::cout << "Target QPS                     : " << target_qps << "\n";
    std::cout << "Total Batch Latency            : " << total_latency_ms.count() << " ms\n";
    std::cout << "Overall QPS                    : " << overall_qps << "\n";
    std::cout << "Average Query Latency(Batch/N) : " << avg_query_latency_ms << " ms\n";
    if (timed_count > 0)
    {
        std::cout << "\n--- Per-Query Pipeline Breakdown (avg over " << timed_count << " queries) ---\n";
        std::cout << "  IVF Filter          : " << std::fixed << std::setprecision(3) << (agg_ivf / timed_count) << " ms\n";
        std::cout << "  Candidate Merge     : " << std::fixed << std::setprecision(3) << (agg_merge / timed_count) << " ms\n";
        std::cout << "  Attr Filter (mask)  : " << std::fixed << std::setprecision(3) << (agg_attr / timed_count) << " ms\n";
        std::cout << "  Collect Results     : " << std::fixed << std::setprecision(3) << (agg_collect / timed_count) << " ms\n";
        std::cout << "  Final Merge         : " << std::fixed << std::setprecision(3) << (agg_final_merge / timed_count) << " ms\n";
        double total_cpu = (agg_ivf + agg_merge + agg_attr + agg_collect + agg_final_merge) / timed_count;
        std::cout << "  Total CPU (sum)     : " << std::fixed << std::setprecision(3) << total_cpu << " ms\n";
    }
    std::cout << "========================================================\n";

    std::cout << "[System] All queries completed. Shutting down worker threads...\n";
    for (auto &group : groups)
    {
        group->Stop();
    }
    for (auto &thread : threads)
    {
        if (thread.joinable())
        {
            thread.join();
        }
    }

    // Print per-group query processing statistics
    std::cout << "\n=== Per-Group Query Processing Stats ===\n";
    std::vector<uint64_t> dev_totals(npu_device_count, 0);
    for (int i = 0; i < group_count; ++i)
    {
        int dev = npu_device_id_start + g_group_to_device[i];
        int numa = (i * cores_per_group) / 24;
        uint64_t cnt = groups[i]->GetProcessedCount();
        dev_totals[g_group_to_device[i]] += cnt;
        std::cout << "  Group " << i
                  << " (cores " << (i * cores_per_group) << "-" << (i * cores_per_group + cores_per_group - 1)
                  << ", NUMA " << numa
                  << ", Device " << dev
                  << "): " << cnt << " queries\n";
    }
    for (int d = 0; d < npu_device_count; ++d)
    {
        std::cout << "  --- Device " << (npu_device_id_start + d) << " (" << GroupCountForDevice(d) << " groups) total: "
                  << dev_totals[d] << " queries ---\n";
    }
    if (npu_device_count == 2 && dev_totals[0] > 0 && dev_totals[1] > 0)
    {
        double ratio = dev_totals[0] > dev_totals[1]
                           ? (double)dev_totals[0] / dev_totals[1]
                           : (double)dev_totals[1] / dev_totals[0];
        std::cout << "  --- Imbalance ratio: " << ratio << "x ---\n";
    }
    std::cout << "=========================================\n";

    for (const auto &query : queries)
    {
        query->AppendMemoryEventLogs(memory_log_collector);
    }

    std::vector<double> total_end_to_end_ms;
    if (!CollectParallelLatencyMetrics(queries, total_end_to_end_ms))
    {
        return -1;
    }

    std::vector<double> recall_rate_percent;
    if (!CollectParallelRecallMetrics(dataset,
                                      prepared_queries,
                                      queries,
                                      ground_truth_cache.get(),
                                      recall_rate_percent))
    {
        return -1;
    }

    if (!FlushParallelOutputs(query_result_root_dir,
                              total_end_to_end_ms,
                              recall_rate_percent,
                              overall_qps))
    {
        std::cerr << "[Fatal] Failed to write parallel outputs.\n";
        return -1;
    }
    if (!memory_log_collector.Flush(memory_log_file))
    {
        std::cerr << "[Fatal] Failed to write " << memory_log_file << ".\n";
        return -1;
    }

    std::cout << "[System] All threads stopped. Exiting safely.\n";
    return 0;
}
