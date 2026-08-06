load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_proto_library")
load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")

proto_library(
    name = "gen_index_builder_proto",
    srcs = [ "index_builder.proto" ],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_protobuf//:descriptor_proto",
    ]
)

cc_proto_library(
    name = "index_builder_proto",
    visibility = ["//visibility:public"],
    deps = [":gen_index_builder_proto"],
)

proto_library(
    name = "gen_document_proto",
    srcs = [ "document.proto" ],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_protobuf//:descriptor_proto",
    ]
)

cc_proto_library(
    name = "document_proto",
    visibility = ["//visibility:public"],
    deps = [":gen_document_proto"],
)

proto_library(
    name = "gen_index_meta_proto",
    srcs = [ "index_meta.proto" ],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_protobuf//:descriptor_proto",
    ],
)

cc_proto_library(
    name = "index_meta_proto",
    visibility = ["//visibility:public"],
    deps = [":gen_index_meta_proto"],
)