cmake_minimum_required(VERSION 3.16)
project(HybridRetrievalSystem)

# --- 1. 核心参数 ---
set(RUN_MODE "npu" CACHE STRING "run mode" FORCE)
set(SOC_VERSION "Ascend910B3" CACHE STRING "system on chip type" FORCE)
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest" CACHE PATH "ASCEND CANN package path")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type Release/Debug" FORCE)
endif()

# 分析专用构建：保留符号、保留帧指针、关闭LTO，便于 FlameGraph 展开函数栈。
option(ENABLE_PROFILING_BUILD "Use profiling-friendly flags for better stack visibility" OFF)

if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "path for install()" FORCE)
endif()
# 设置 C++ 标准 
set(CMAKE_CXX_STANDARD 17)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

# ================= 新增部分 =================
# 1. 显式添加 /usr/local 到查找路径（防止系统默认未包含）
list(APPEND CMAKE_PREFIX_PATH "/usr/local")

find_package(gflags REQUIRED)
find_package(glog REQUIRED)

# 强制将 /usr/local 的库目录加入链接阶段的搜索路径
link_directories(/usr/local/lib /usr/local/lib64)

# 2. 查找 Folly 及其依赖包
find_package(folly CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(BLAS REQUIRED)
find_package(fmt REQUIRED)
# ================= END =================

if(ENABLE_PROFILING_BUILD)
    message(STATUS "ENABLE_PROFILING_BUILD=ON: using flamegraph-friendly compile/link flags")
endif()


# --- 2. 算子文件路径 ---
set(DEVICE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/NPU/device")
file(GLOB KERNEL_FILES 
    "${DEVICE_DIR}/kernel_vector_mmad_op.cpp"
)

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

    if(ENABLE_PROFILING_BUILD)
        target_compile_options(${target_name} PRIVATE
            -O2 -g -fno-omit-frame-pointer -fno-optimize-sibling-calls
            -fno-lto -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0
            -Wall -Wno-unknown-pragmas -Wno-sign-compare -march=armv8-a
        )
        target_link_options(${target_name} PRIVATE -fno-lto)
    else()
        # 注意：保留了 -D_GLIBCXX_USE_CXX11_ABI=0 以兼容 CANN
        target_compile_options(${target_name} PRIVATE
            -Ofast -flto -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0
            -Wall -Wno-unknown-pragmas -march=armv8-a -Wno-sign-compare
        )
        target_link_options(${target_name} PRIVATE -flto -Ofast)
    endif()

# --- 6. 链接库 ---
    target_link_libraries(${target_name} PRIVATE
        ascendc_kernels_${RUN_MODE}
        ascendcl
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

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/config.txt")
    install(FILES config.txt DESTINATION bin)
else()
    message(WARNING "config.txt not found in source root!")
endif()
