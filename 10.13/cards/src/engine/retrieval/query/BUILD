package(
    default_visibility = [
        "//visibility:public",
    ],
)

cc_library(
    name = "query",
    hdrs = [
        "query_node.h",
        "query_node_imp.h"
    ],
    srcs = [
        "query_node.cpp",
        "query_node_imp.cpp"

    ],
    deps = [
        "//src/full_recall/index:data_table",
    ],
    alwayslink = True,
)

cc_library(
    name = "mock_query",
    hdrs = [
        "mock_query_node.h"
    ],
    deps = [
        "//src/utils:npur_log",
    ],
    alwayslink = True,
)