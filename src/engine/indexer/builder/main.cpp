#include <gflags/gflags.h>
#include <filesystem>
#include "src/full_recall/indexer/builder/builder_impl.h"
#include "src/utils/logger.h"
#include "src/configuration/develop_configuration.h"

DEFINE_string(server_log_properties, "../conf/npuretrieval_log.properties", "log properties");
DEFINE_string(data_dir, "../data_dir", "data input");
DEFINE_string(data_out, "../data_out", "data output");
DEFINE_string(ipc_socket_path, "", "socket path for streaming data in memory build mode");
DEFINE_bool(build_from_memory, false, "enable memory build mode");

int main(int argc, char** argv) {
    google::ParseCommandLineFlags(&argc, &argv, false);  // gflags
    FLAGS_npuretrieval_log_level = "DEBUG";
    if (!NpuRetrieval::Logger::Instance().Init(FLAGS_server_log_properties)) {
        return -1;
    }

    std::string dataDir = FLAGS_data_dir;
    std::string output = FLAGS_data_out;
    std::ifstream ifs(dataDir + "/schema.json");
    if (!ifs) {
        LOG_ERROR("read schema file fail");
        return -1;
    }
    std::stringstream buffer;
    buffer << ifs.rdbuf();
    std::string schemaContent(buffer.str());
    std::string inputSource;
    if (FLAGS_build_from_memory) {
        if (FLAGS_ipc_socket_path.empty()) {
            LOG_ERROR("Memory build mode requires param： ipc_socket_path");
            return -1;
        }
        inputSource = FLAGS_ipc_socket_path;
        LOG_INFO("Using memory build mode, sockets: " << inputSource);
    } else {
        inputSource = FLAGS_data_dir;
        LOG_INFO("Using file build mode, input dir: " << inputSource);
    }
    bool ret = NpuRetrieval::Build(schemaContent, 0, inputSource, output);
    if (!ret) {
        LOG_ERROR("Build fail");
        return -1;
    }
    return 0;
}
