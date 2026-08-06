package(
    default_visibility = [
        "//visibility:public",
    ],
)

cc_library(
    name = "builder_schema",
    srcs = [
        "builder_schema.cpp",
    ],
    hdrs = [
        "builder_schema.h",
    ],
    deps = [
        "//src/full_recall/indexer/proto:index_builder_proto",
        "//third_party:grpc++",
        "//src/utils:npur_log",
    ],
    alwayslink = True,
)