package(
    default_visibility = [
        "//visibility:public",
    ],
)

cc_library(
    name = "idmapping_builder",
    srcs = [
        "segment/idmapping_builder.cpp",
    ],
    hdrs = [
        "segment/idmapping_builder.h",
    ],
    deps = [
        "//src/full_recall/indexer/schema:builder_schema",
        "//src/full_recall/core:constant_definition",
        "//src/full_recall/indexer/file:file_writer",
        "//src/full_recall/indexer/file:file_reader",
        "//src/utils:npur_log",
        "//src/utils:file_manager",
        "//third_party:gflags",
    ],
    alwayslink = True,
)

cc_library(
    name = "vector_builder",
    srcs = [
        "segment/vector_builder.cpp",
    ],
    hdrs = [
        "segment/vector_builder.h",
    ],
    deps = [
        "//src/full_recall/indexer/schema:builder_schema",
        "//src/full_recall/core:constant_definition",
        "//src/full_recall/indexer/file:file_writer",
        "//src/full_recall/indexer/file:file_reader",
        "//src/full_recall/indexer/proto:document_proto",
        "//src/utils:npur_log",
        "//src/utils:file_manager",
        ":idmapping_builder",
        "//src/full_recall/core/number:trans_number",
        "//third_party:grpc++",
        "//third_party:gflags",
    ],
    alwayslink = True,
)

cc_library(
    name = "meta_builder",
    srcs = [
        "meta_builder.cpp",
    ],
    hdrs = [
        "meta_builder.h",
    ],
    deps = [
        "//src/full_recall/indexer/schema:builder_schema",
        "//src/full_recall/core:constant_definition",
        "//src/full_recall/indexer/proto:index_meta_proto",
        "//src/utils:npur_log",
        "//third_party:grpc++",
    ],
    alwayslink = True,
)

cc_library(
    name = "inverted_builder",
    srcs = [
        "segment/inverted_builder.cpp",
    ],
    hdrs = [
        "segment/inverted_builder.h",
    ],
    deps = [
        "//src/full_recall/indexer/schema:builder_schema",
        "//src/full_recall/core:constant_definition",
        "//src/full_recall/indexer/file:file_writer",
        "//src/full_recall/indexer/file:file_reader",
        "//src/full_recall/indexer/file:memory_data_manager",
        "//src/full_recall/indexer/proto:document_proto",
        "//src/full_recall/core/hash:hash",
        "//src/full_recall/format:posting_serializer",
        "//src/utils:file_manager",
        ":idmapping_builder",
        "//src/utils:npur_log",
        "//third_party:grpc++",
        "//src/workflow/async:executor",
        "//third_party:gflags",
    ],
    alwayslink = True,
)

cc_library(
    name = "builder_impl",
    srcs = [
        "builder_impl.cpp",
        "stream_input.cpp",
    ],
    hdrs = [
        "builder_impl.h",
        "stream_input.h",
    ],
    deps = [
        "//src/full_recall/indexer/schema:builder_schema",
        "//src/utils:npuretrieval_utils",
        "//src/utils:npur_log",
        "//src/utils:file_manager",
        "//src/workflow/async:executor",
        ":idmapping_builder",
        ":vector_builder",
        ":meta_builder",
        ":inverted_builder",
        "//third_party:grpc++",
        "//huawei_platform:huaweiSecureC",
        "//third_party:gflags",
    ],
    alwayslink = True,
)

cc_binary(
    name = "FullRecallBuild",
    srcs = [
        "main.cpp",
    ],
    linkopts = [
        "-z noexecstack -z relro -z now -fPIC",
        "-lpthread",
        "-pie"
    ],
    deps = [
        "//third_party:gflags",
        "//third_party:jemalloc",
        "//src/full_recall/indexer/builder:builder_impl",
        "//huawei_platform:huaweiSecureC",
        "//src/utils:npur_log",
        "//src/utils:npuretrieval_utils",
    ],
)