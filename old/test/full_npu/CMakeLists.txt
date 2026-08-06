cmake_minimum_required(VERSION 3.16)
project(full_npu CXX)

set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON CACHE BOOL "" FORCE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# --- 1. Core params (match sks_hw ascendc.cmake expectations) ---
set(RUN_MODE "npu" CACHE STRING "run mode" FORCE)
set(SOC_VERSION "Ascend910B3" CACHE STRING "system on chip type")

# --- CANN toolkit auto-detect ---
if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    set(_CANN_CANDIDATES
        "/usr/local/Ascend/ascend-toolkit/latest"
        "/home/lcy/Ascend/ascend-toolkit/latest"
    )
    foreach(_dir ${_CANN_CANDIDATES})
        if(IS_DIRECTORY "${_dir}")
            set(ASCEND_CANN_PACKAGE_PATH "${_dir}" CACHE PATH "ASCEND CANN package path")
            break()
        endif()
    endforeach()
    if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
        message(FATAL_ERROR "CANN toolkit not found. Pass -DASCEND_CANN_PACKAGE_PATH=...")
    endif()
endif()

# SOC version: default Ascend910B3, override via -DSOC_VERSION=...
if(NOT DEFINED SOC_VERSION)
    set(SOC_VERSION "Ascend910B3" CACHE STRING "system on chip type")
endif()
message(STATUS "full_npu: CANN=${ASCEND_CANN_PACKAGE_PATH} SOC=${SOC_VERSION} RUN_MODE=${RUN_MODE}")

find_package(OpenMP REQUIRED)

# --- AscendC kernel compile (bisheng via ascendc.cmake) ---
if(NOT EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    message(FATAL_ERROR "CANN kernel cmake not found at ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake")
endif()
set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

set(KERNEL_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/npu/kernel_score.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/npu/kernel_filter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/npu/kernel_filter_inverted.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/npu/kernel_aggregator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/npu/kernel_topk.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/npu/kernel_topk_fused.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/npu/kernel_topk_fused_multi.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/npu/kernel_topk_merge.cpp
)
ascendc_library(full_npu_kernels_${RUN_MODE} SHARED ${KERNEL_FILES})
ascendc_compile_definitions(full_npu_kernels_${RUN_MODE} PRIVATE
    $<$<BOOL:$<IN_LIST:${SOC_VERSION},Ascend910B1;Ascend910B2;Ascend910B3;Ascend910B4>>:SOC_910B>
)

# --- Host executable ---
add_executable(full_npu main.cpp)
add_dependencies(full_npu full_npu_kernels_${RUN_MODE})

target_include_directories(full_npu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/npu
    ${ASCEND_CANN_PACKAGE_PATH}/include
)

target_compile_options(full_npu PRIVATE
    -D_GLIBCXX_USE_CXX11_ABI=0
    -DFULL_NPU_ENABLE_NPU
    -Wall -Wno-sign-compare
)

target_link_directories(full_npu PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
    ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/lib64
)

target_link_libraries(full_npu PRIVATE
    full_npu_kernels_${RUN_MODE}
    ascendcl
    acl_op_compiler
    dl
    pthread
    OpenMP::OpenMP_CXX
)

# --- Inverted-index preprocessing tool ---
add_executable(build_inverted_index tools/build_inverted_index.cpp)
target_include_directories(build_inverted_index PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/common
)
target_compile_options(build_inverted_index PRIVATE
    -D_GLIBCXX_USE_CXX11_ABI=0
    -Wall -Wno-sign-compare
)
target_link_libraries(build_inverted_index PRIVATE dl pthread)
