package(
    default_visibility = [
        "//visibility:public",
    ],
)

cc_library(
    name = "hash",
    srcs = [
        "hash.cpp",
    ],
    hdrs = [
        "hash.h",
    ],
    deps = [
        "//src/utils:npur_log",
        "//third_party:murmurhash3",
    ],
    alwayslink = True,
)