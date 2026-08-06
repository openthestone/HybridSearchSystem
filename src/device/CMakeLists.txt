# =============================================================================
# src/device/CMakeLists.txt — standalone build of the AscendC device library.
#
# WHY SEPARATE: ascendc_library() creates ExternalProjects whose build step runs
# on every `cmake --build` and is NON-idempotent (the ld.lld merge step rewrites
# an object in place, failing on a second run with "unknown file type"). Building
# the device lib in its OWN tree, once, lets the main src/ build import the
# resulting libascend_device.so + generated aclrtlaunch_*.h headers without ever
# re-triggering the device build — so host-side iteration is fast and never needs
# `rm -rf build`.
#
# Build once:
#   cmake -S src/device -B build_device \
#         -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/ascend-toolkit/latest \
#         -DSOC_VERSION=Ascend910B3
#   cmake --build build_device -j
# Products: build_device/lib/libascend_device.so, build_device/include/ascend_device/*.h
# The main build auto-detects build_device/ (see src/CMakeLists.txt).
# =============================================================================
cmake_minimum_required(VERSION 3.16)
project(FrnewDevice CXX)

set(SOC_VERSION "Ascend910B3" CACHE STRING "Ascend SoC type")
# Match the main build (which builds the device lib cleanly): default to Release.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# ---- CANN detection (mirrors src/CMakeLists.txt) --------------------------
if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    foreach(_dir "/usr/local/Ascend/ascend-toolkit/latest"
                 "$ENV{HOME}/Ascend/ascend-toolkit/latest")
        if(IS_DIRECTORY "${_dir}")
            set(ASCEND_CANN_PACKAGE_PATH "${_dir}" CACHE PATH "CANN package path")
            break()
        endif()
    endforeach()
endif()
if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH OR NOT IS_DIRECTORY "${ASCEND_CANN_PACKAGE_PATH}")
    message(FATAL_ERROR "CANN toolkit not found. Pass -DASCEND_CANN_PACKAGE_PATH=...")
endif()

if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake not found under CANN package")
endif()
set(RUN_MODE "npu" CACHE STRING "run mode" FORCE)
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

# ---- build the 4 kernels into libascend_device.so --------------------------
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(DEVICE_DIR "${REPO_ROOT}/src/engine/retrieval/device")
file(GLOB DEVICE_KERNELS
    "${DEVICE_DIR}/kernel_bitmap_text_filter.cpp"
    "${DEVICE_DIR}/kernel_aggregator.cpp"
    "${DEVICE_DIR}/kernel_vector_mmad_op.cpp"
    "${DEVICE_DIR}/kernel_topk.cpp"
)
ascendc_library(ascend_device SHARED ${DEVICE_KERNELS})
ascendc_compile_definitions(ascend_device PRIVATE
    TRIGGER_ERROR=0
    $<$<BOOL:$<IN_LIST:${SOC_VERSION},Ascend910B1;Ascend910B2;Ascend910B3;Ascend910B4>>:SOC_910B>
)
