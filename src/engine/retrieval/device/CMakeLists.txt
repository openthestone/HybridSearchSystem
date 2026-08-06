cmake_minimum_required(VERSION 3.16)
project(Ascend_c)

set(ASCEND_CANN_PACKAGE_PATH "{Toolkit_InstallPath}/latest"
        CACHE STRING "ASCEND CANN package installation directory" )
if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist ,please check whether the cann package is installed")
endif()

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

set(CMAKE_INSTALL_PREFIX "../ascend_device/" CACHE STRING "path for install()" FORCE)
set(CMAKE_INSTALL_LIBDIR lib)
file(GLOB KERNEL_FILES kernel_bitmap_text_filter.cpp kernel_aggregator.cpp kernel_vector_mmad_op.cpp kernel_topk.cpp)

ascendc_library(ascend_device SHARED ${KERNEL_FILES})
ascendc_compile_definitions(ascend_device PRIVATE TRIGGER_ERROR=0)