cmake_minimum_required(VERSION 3.16)
project(filter_cmp CXX C)

set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON CACHE BOOL "" FORCE)

# --- 1. CANN auto-detect ---
if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    set(_CANN_CANDIDATES
        "/usr/local/Ascend/ascend-toolkit/latest"
        "$ENV{HOME}/Ascend/ascend-toolkit/latest"
    )
    foreach(_dir ${_CANN_CANDIDATES})
        if(IS_DIRECTORY "${_dir}")
            set(ASCEND_CANN_PACKAGE_PATH "${_dir}" CACHE PATH "CANN toolkit path" FORCE)
            break()
        endif()
    endforeach()
    if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
        message(FATAL_ERROR "CANN toolkit not found. Pass -DASCEND_CANN_PACKAGE_PATH=...")
    endif()
endif()

set(SOC_VERSION "Ascend910B3" CACHE STRING "SoC version")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Threads REQUIRED)

# --- 2. Kernel sources ---
set(SIMPLE_KERNEL   ${CMAKE_CURRENT_SOURCE_DIR}/npu_simple/kernel_simple_filter.cpp)
set(TIANJI_KERNELS
    ${CMAKE_CURRENT_SOURCE_DIR}/npu_tianji/kernel_bitmap_text_filter.cpp)

# --- 3. AscendC build ---
if(NOT EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    message(FATAL_ERROR "CANN kernel cmake missing at ${ASCEND_CANN_PACKAGE_PATH}")
endif()
set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

ascendc_library(ascendc_simple SHARED ${SIMPLE_KERNEL})
ascendc_compile_definitions(ascendc_simple PRIVATE
    $<$<BOOL:$<IN_LIST:${SOC_VERSION},Ascend910B1;Ascend910B2;Ascend910B3;Ascend910B4>>:SOC_910B>)

ascendc_library(ascendc_tianji SHARED ${TIANJI_KERNELS})
ascendc_compile_definitions(ascendc_tianji PRIVATE
    $<$<BOOL:$<IN_LIST:${SOC_VERSION},Ascend910B1;Ascend910B2;Ascend910B3;Ascend910B4>>:SOC_910B>)

# --- 4. Host executable ---
set(HOST_SRC
    main.cpp
    common/expr.cpp
    common/raw_data.cpp
    common/cpu_filter.cpp
    npu_common/npu_posting.cpp
    npu_simple/simple_filter.cpp
    npu_tianji/tianji_filter.cpp)

add_executable(filter_cmp ${HOST_SRC})

target_include_directories(filter_cmp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/npu_common
    ${CMAKE_CURRENT_SOURCE_DIR}/npu_simple
    ${CMAKE_CURRENT_SOURCE_DIR}/npu_tianji
    ${ASCEND_CANN_PACKAGE_PATH}/include)

target_compile_options(filter_cmp PRIVATE
    -O2 -g -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0
    -Wall -Wno-sign-compare -march=armv8-a)

target_link_libraries(filter_cmp PRIVATE
    ascendc_simple
    ascendc_tianji
    ascendcl
    acl_op_compiler
    dl
    pthread
    Threads::Threads)
