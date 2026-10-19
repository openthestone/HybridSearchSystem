package(
    default_visibility = [
        "//visibility:public",
    ],
)
cc_library(
    name = "full_recall_result",
    hdrs = [
        "full_recall_result.h",
    ],
    alwayslink = True,
)


cc_library(
    name = "constant_definition",
    hdrs = [
        "constant_definition.h",
    ],
    deps = [
        "//src/utils:npur_log",
    ],
    alwayslink = True,
)

cc_library(
    name = "log_definition",
    hdrs = [
        "log_definition.h",
    ],
    deps = [
        "//src/utils:npur_log",
        "@ascend//:ascend",
    ],
    alwayslink = True,
)