package(
    default_visibility = [
        "//visibility:public",
    ],
)

cc_library(
    name = "searcher",
    srcs = [
        "full_recall_searcher.cpp",
        "result_aggregator.cpp",
        "text_filter.cpp",
        "vector_scorer_mmad.cpp",
        "runtime/gm_memory_manager.cpp",
        "runtime/stream_manager.cpp"
    ],
    hdrs = [
        "full_recall_searcher.h",
        "result_aggregator.h",
        "text_filter.h",
        "vector_scorer_mmad.h",
        "runtime/gm_memory_pool.h",
        "runtime/gm_memory_manager.h",
        "runtime/stream_manager.h"
    ],
    deps = [
        "@ascend//:ascend",
        "//src/common:npuretrieval_common",
        "//src/configuration:npuretrieval_config",
        "//src/full_recall/index:data_table",
        "//src/full_recall/retrieval/query:query",
        "//src/full_recall/retrieval/device:ascend_device",
        "//src/full_recall/core/number:trans_number",
        "//src/full_recall/core:constant_definition",
        "//src/full_recall/core:full_recall_result",
        "//src/full_recall/core:log_definition",
        "//src/utils:npuretrieval_utils",
        "//src/workflow/async:executor",
        "//src/full_recall/retrieval/device:device_common_external"
    ],
    linkopts = [
          "-lascendcl",
          "-lregister",
          "-ltiling",
          "-ltiling_api",
          "-lplatform",
          "-lascendalog",
          "-ldl",
    ],
    alwayslink = True,
)