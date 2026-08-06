package(
    default_visibility = [
        "//visibility:public",
    ],
)

cc_library(
    name = "doc_id_mapping",
    srcs = [
        "docid/doc_id_mapping.cpp",
    ],
    hdrs = [
        "docid/doc_id_mapping.h",
    ],
    deps = [
        "//src/full_recall/core:constant_definition",
        "//src/full_recall/core:log_definition",
        "//src/utils:npur_log",
        "//src/utils:file_manager",
        "//src/full_recall/format:file_header"
    ],
    alwayslink = True,
)

cc_library(
    name = "vector_field_data",
    srcs = [
        "vector/vector_field_data.cpp",
    ],
    hdrs = [
        "vector/vector_field_data.h",
    ],
    deps = [
        "@ascend//:acl",
        "//src/full_recall/core:constant_definition",
        "//src/full_recall/core:log_definition",
        "//src/utils:npur_log",
        "//src/full_recall/format:file_header",
        "//src/full_recall/core/number:trans_number"
    ],
    linkopts = [
        "-L{Toolkit_InstallPath}/latest/lib64",
        "-lascendcl",
        "-Wl,-rpath={Toolkit_InstallPath}/latest/lib64:{Toolkit_InstallPath}/latest/{arch}-linux/runtime/lib64/stub/{arch}",
    ],
    alwayslink = True,
)

cc_library(
    name = "vector_data",
    srcs = [
        "vector/vector_data.cpp",
    ],
    hdrs = [
        "vector/vector_data.h",
    ],
    deps = [
        "//src/full_recall/core:constant_definition",
        "//src/full_recall/core:log_definition",
        "//src/utils:npur_log",
        "//src/utils:file_manager",
        ":vector_field_data"
    ],
    alwayslink = True,
)

cc_library(
    name = "posting_field_data",
    srcs = [
        "posting/posting_field_data.cpp",
    ],
    hdrs = [
        "posting/posting_field_data.h",
    ],
    deps = [
        "@ascend//:acl",
        "//src/full_recall/core:constant_definition",
        "//src/full_recall/core:log_definition",
        "//src/utils:npur_log",
        "//src/full_recall/format:file_header",
        "//src/full_recall/format:posting_serializer",
    ],
    linkopts = [
        "-L{Toolkit_InstallPath}/latest/lib64",
        "-lascendcl",
        "-Wl,-rpath={Toolkit_InstallPath}/latest/lib64:{Toolkit_InstallPath}/latest/{arch}-linux/runtime/lib64/stub/{arch}",
        ],
    alwayslink = True,
)

cc_library(
    name = "posting_data",
    srcs = [
        "posting/posting_data.cpp",
    ],
    hdrs = [
        "posting/posting_data.h",
    ],
    deps = [
        "@ascend//:acl",
        "//src/full_recall/core:constant_definition",
        "//src/full_recall/core:log_definition",
        "//src/utils:npur_log",
        "//src/utils:file_manager",
        ":posting_field_data",
    ],
    alwayslink = True,
)

cc_library(
    name = "data_table",
    srcs = [
        "data_table.cpp",
    ],
    hdrs = [
        "data_table.h",
    ],
    deps = [
        ":doc_id_mapping",
        ":vector_data",
        ":posting_data",
        "//src/full_recall/core:constant_definition",
        "//src/full_recall/core:log_definition",
        "//src/full_recall/core/number:trans_number",
        "//src/utils:npur_log",
        "//src/utils:string_util",
        "//src/utils:file_manager",
        "//third_party:grpc++",
        "//src/full_recall/indexer/proto:index_meta_proto",
        "//src/common:npuretrieval_common"
    ],
    alwayslink = True,
)

cc_library(
    name = "mock_data_table",
    hdrs = [
        "mock_data_table.h",
    ],
    deps = [
        "//src/utils:npur_log",
    ],
    alwayslink = True,
)

cc_library(
    name = "data_table_repository",
    srcs = [
        "data_table_repository.cpp",
    ],
    hdrs = [
        "data_table_repository.h",
    ],
    deps = [
        "//src/full_recall/core:constant_definition",
        "//src/common:npuretrieval_common",
        "//src/utils:npuretrieval_utils",
        "//src/utils:npur_log",
        "//src/configuration:npuretrieval_develop_configuration",
        "//src/workflow/async:executor",
    ] + select(
        {
            "//src:with_ascend_c":["//src/full_recall/index:data_table","//src/full_recall/retrieval/searcher:searcher"],
            "//conditions:default":["//src/full_recall/index:mock_data_table"]
        }
    ),
    defines = select(
        {
            "//src:with_ascend_c":["ASCEND_C_ENABLED"],
            "//conditions:default":[]
        }
    ),
    alwayslink = True,
)