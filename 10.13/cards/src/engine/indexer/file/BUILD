package(
    default_visibility = [
        "//visibility:public",
    ],
)

cc_library(
    name = "file_writer",
    srcs = [
        "file_writer.cpp",
    ],
    hdrs = [
        "file_writer.h",
    ],
    deps = [
        "//src/utils:npur_log",
    ],
    alwayslink = True,
)

cc_library(
    name = "file_reader",
    srcs = [
        "file_reader.cpp",
    ],
    hdrs = [
        "file_reader.h",
    ],
    deps = [
        "//src/utils:npur_log",
        "//src/full_recall/core:constant_definition",
        "//src/full_recall/indexer/file:memory_data_manager",
    ],
    alwayslink = True,
)

cc_library(
    name = "memory_data_manager",
    srcs = [
        "memory_data_manager.cpp",
    ],
    hdrs = [
        "memory_data_manager.h",
    ],
    deps = [
        "//src/utils:npur_log",
        "//src/full_recall/core:constant_definition",
    ],
    alwayslink = True,
)