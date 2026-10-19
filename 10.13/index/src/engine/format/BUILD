package(
    default_visibility = [
        "//visibility:public",
    ],
)

cc_library(
    name = "posting_serializer",
    srcs = [
        "posting_serializer.cpp",
    ],
    hdrs = [
        "posting_layout.h",
        "posting_serializer.h",
    ],
    deps = [
        "//src/utils:npur_log",
    ],
    alwayslink = True,
)

cc_library(
    name = "file_header",
    srcs = [
        "file_header.cpp",
    ],
    hdrs = [
        "file_header.h",
    ],
    deps = [
        "//src/utils:npur_log",
    ],
    alwayslink = True,
)
