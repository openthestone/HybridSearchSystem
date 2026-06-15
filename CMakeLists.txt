cmake_minimum_required(VERSION 3.16)
project(HybridRetrievalSystem)

# Suppress CMake unused-variable warnings from CANN's internal subprojects
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON CACHE BOOL "" FORCE)

# --- 1. 核心参数 ---
set(RUN_MODE "npu" CACHE STRING "run mode" FORCE)
set(SOC_VERSION "Ascend910B3" CACHE STRING "system on chip type")
# Auto-detect CANN toolkit path if not specified
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

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type Release/Debug" FORCE)
endif()

# 分析专用构建：保留符号、保留帧指针、关闭LTO，便于 FlameGraph 展开函数栈。
option(ENABLE_PROFILING_BUILD "Use profiling-friendly flags for better stack visibility" OFF)
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)

if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "path for install()" FORCE)
endif()
# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

# --- 1b. 查找依赖包 ---
list(APPEND CMAKE_PREFIX_PATH "/usr/local")

find_package(gflags REQUIRED)
find_package(glog REQUIRED)
find_package(folly CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(BLAS REQUIRED)
find_package(fmt REQUIRED)

if(ENABLE_PROFILING_BUILD)
    message(STATUS "ENABLE_PROFILING_BUILD=ON: using flamegraph-friendly compile/link flags")
endif()


# --- 2. 算子文件路径 ---
set(DEVICE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/NPU/device")
set(KERNEL_FILES "${DEVICE_DIR}/kernel_vector_mmad_op.cpp")

# --- 2b. AI CPU mask filter kernel (pure C++ .so) ---
set(AICPU_KERNEL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/include/NPU/aicpu_mask_filter_kernel.cpp")
set(AICPU_KERNEL_SO "${CMAKE_CURRENT_BINARY_DIR}/libcust_aicpu_kernels_sks_final.so")

add_custom_command(
    OUTPUT ${AICPU_KERNEL_SO}
    COMMAND ${CMAKE_CXX_COMPILER} -shared -fPIC -o ${AICPU_KERNEL_SO}
            ${AICPU_KERNEL_SRC}
            -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++17 -O2
    DEPENDS ${AICPU_KERNEL_SRC}
    COMMENT "Building AI CPU mask filter kernel .so"
)
add_custom_target(aicpu_mask_filter_kernel ALL DEPENDS ${AICPU_KERNEL_SO})

# --- 3. NPU 构建逻辑 ---
if(NOT EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    message(FATAL_ERROR "CANN package not found at ${ASCEND_CANN_PACKAGE_PATH}")
endif()
set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

# --- 4. 编译算子库 ---
ascendc_library(ascendc_kernels_${RUN_MODE} SHARED ${KERNEL_FILES})
ascendc_compile_definitions(ascendc_kernels_${RUN_MODE} PRIVATE
    $<$<BOOL:$<IN_LIST:${SOC_VERSION},Ascend910B1;Ascend910B2;Ascend910B3;Ascend910B4>>:SOC_910B>
)

# --- 5. 编译 Host 主程序 ---
function(configure_host_target target_name source_file)
    add_executable(${target_name}
        ${source_file}
        include/NPU/npuAPI.cpp
    )

    target_include_directories(${target_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/NPU/device
        ${CMAKE_CURRENT_SOURCE_DIR}/include/Clustering/SuperKMeans/include
        ${ASCEND_CANN_PACKAGE_PATH}/include
    )

    add_dependencies(${target_name} aicpu_mask_filter_kernel)

    if(ENABLE_ASAN)
        target_compile_options(${target_name} PRIVATE
            -O1 -g -fno-omit-frame-pointer -fsanitize=address -fno-lto
            -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0
            -Wall -Wno-unknown-pragmas -Wno-sign-compare -Wno-alloc-size-larger-than -march=armv8-a
        )
        target_link_options(${target_name} PRIVATE -fsanitize=address -fno-lto)
    elseif(ENABLE_PROFILING_BUILD)
        target_compile_options(${target_name} PRIVATE
            -O2 -g -fno-omit-frame-pointer -fno-optimize-sibling-calls
            -fno-lto -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0
            -Wall -Wno-unknown-pragmas -Wno-sign-compare -Wno-alloc-size-larger-than -march=armv8-a
        )
        target_link_options(${target_name} PRIVATE -fno-lto)
    else()
        target_compile_options(${target_name} PRIVATE
            -Ofast -flto -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0
            -Wall -Wno-unknown-pragmas -Wno-alloc-size-larger-than -march=armv8-a -Wno-sign-compare
        )
        target_link_options(${target_name} PRIVATE -flto -Ofast)
    endif()

# --- 6. 链接库 ---
    target_link_libraries(${target_name} PRIVATE
        ascendc_kernels_${RUN_MODE}
        ascendcl
        acl_op_compiler
        dl
        pthread
        OpenMP::OpenMP_CXX
        Folly::folly
        Eigen3::Eigen
        BLAS::BLAS
        fmt::fmt
    )
endfunction()

configure_host_target(serial src/serial.cpp)
configure_host_target(parallel src/parallel.cpp)

# --- 6b. 分析工具 ---
configure_host_target(analyze_t1 src/analyze_t1.cpp)
configure_host_target(analyze_t2 src/analyze_t2.cpp)
configure_host_target(analyze_t3 src/analyze_t3.cpp)
configure_host_target(analyze_t8 src/analyze_t8.cpp)

# --- 7. 安装 ---
install(TARGETS serial parallel analyze_t1 analyze_t2 analyze_t3 analyze_t8 DESTINATION bin)
install(TARGETS ascendc_kernels_${RUN_MODE} DESTINATION lib)

# Install AI CPU kernel .so to build output (run.sh deploys to CANN vendor)
install(FILES ${AICPU_KERNEL_SO} DESTINATION bin)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/config.txt")
    install(FILES config.txt DESTINATION bin)
else()
    message(WARNING "config.txt not found in source root!")
endif()
