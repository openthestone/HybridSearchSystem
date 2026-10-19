genrule(
    name = "ascend_device-compile",
    srcs = glob(["**"]),
    outs = [
        "ascend_device/lib/libascend_device.so",
        "ascend_device/include/ascend_device/aclrtlaunch_Aggregator.h",
        "ascend_device/include/ascend_device/aclrtlaunch_BitmapTextFilter.h",
        "ascend_device/include/ascend_device/aclrtlaunch_kernel_vector_mmad.h",
        "ascend_device/include/ascend_device/aclrtlaunch_PostingBitListToSet.h",
        "ascend_device/include/ascend_device/aclrtlaunch_TOPK.h",
    ],
    cmd = " && ".join([
        "mkdir -p ./src/full_recall/retrieval/device/build",
        "cd ./src/full_recall/retrieval/device",
        "cmake -B build -DSOC_VERSION=Ascend910B4 -DCMAKE_BUILD_TYPE=Release",
        "cmake --build build -j",
        "cmake --install build",
        "cd -",
        "echo --------------------",
        "echo ./$(location ascend_device/lib/libascend_device.so)",
        "pwd",
        "cp ./src/full_recall/retrieval/device/build/lib/libascend_device.so ./$(location ascend_device/lib/libascend_device.so)",
        "cp ./src/full_recall/retrieval/device/build/include/ascend_device/aclrtlaunch_Aggregator.h ./$(location ascend_device/include/ascend_device/aclrtlaunch_Aggregator.h)",
        "cp ./src/full_recall/retrieval/device/build/include/ascend_device/aclrtlaunch_BitmapTextFilter.h ./$(location ascend_device/include/ascend_device/aclrtlaunch_BitmapTextFilter.h)",
        "cp ./src/full_recall/retrieval/device/build/include/ascend_device/aclrtlaunch_kernel_vector_mmad.h ./$(location ascend_device/include/ascend_device/aclrtlaunch_kernel_vector_mmad.h)",
        "cp ./src/full_recall/retrieval/device/build/include/ascend_device/aclrtlaunch_PostingBitListToSet.h ./$(location ascend_device/include/ascend_device/aclrtlaunch_PostingBitListToSet.h)",
        "cp ./src/full_recall/retrieval/device/build/include/ascend_device/aclrtlaunch_TOPK.h ./$(location ascend_device/include/ascend_device/aclrtlaunch_TOPK.h)",
    ]),
)

cc_library(
    name = "ascend_device",
    srcs = [":ascend_device-compile"], # depends on the shared library produced by the genrule
    hdrs = glob([
        "ascend_device/include/ascend_device/*.h"
    ]), # public headers
    includes = ["ascend_device/include"], # header search path
    strip_include_prefix = "ascend_device/include",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "device_common_external",
    hdrs = [
        "device_common_external.h",
    ],
    visibility = ["//visibility:public"],
)