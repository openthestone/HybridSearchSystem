# =============================================================================
# src/CMakeLists.txt — build the extracted `engine/` (NpuRetrieval full-recall)
# path against local stubs, on a CANN/Ascend server. Route A (see ../TODO.md).
#
# This is the Phase 1 build skeleton + Phase 4 compile-probe driver. It is
# written to be run ON THE NPU SERVER (needs CANN, protobuf, gflags). It cannot
# be configured on the local dev box (no CANN). Expect to iterate a few rounds
# on the server — items marked "SERVER-VERIFY" are the most likely to need
# adjustment against the real toolkit layout.
#
# Usage:
#   cmake -S src -B build \
#         -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/ascend-toolkit/latest \
#         -DSOC_VERSION=Ascend910B3
#   cmake --build build -j
# =============================================================================
cmake_minimum_required(VERSION 3.16)
project(NpuRetrieval CXX)

set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON CACHE BOOL "" FORCE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# ---- options ---------------------------------------------------------------
set(SOC_VERSION "Ascend910B3" CACHE STRING "Ascend SoC type")
option(NPUR_BUILD_BUILDER "Build the offline index builder (needs rapidjson)" ON)
option(NPUR_BUILD_SEARCH  "Build the search harness (needs src/harness/main.cpp)" ON)
# Huawei CANN libs are typically built with the pre-C++11 libstdc++ ABI. If you
# hit std::string/std::list link errors against CANN, flip this to 0 and make
# sure protobuf/gflags are built with the same ABI. (old/ links with ABI=0.)
set(NPUR_GLIBCXX_ABI "1" CACHE STRING "_GLIBCXX_USE_CXX11_ABI value (0 or 1)")

# ---- CANN toolkit detection (mirrors old/CMakeLists.txt) -------------------
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
message(STATUS "CANN: ${ASCEND_CANN_PACKAGE_PATH}  SoC: ${SOC_VERSION}")

# ascendc kernel cmake (CANN >= newer uses compiler/, older uses tools/)
if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake not found under CANN package")
endif()
set(RUN_MODE "npu" CACHE STRING "run mode" FORCE)
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

# ---- third-party host deps -------------------------------------------------
# Match the original project: third-party packages on the server are installed
# under /usr/local, but some CMake defaults on the Ascend host do not search it.
list(APPEND CMAKE_PREFIX_PATH "/usr/local")

find_package(Threads REQUIRED)
find_package(gflags REQUIRED)

# protobuf 4.25 depends on Abseil. NOTE: this host's protobuf CONFIG package is
# broken (protobuf-config.cmake references a missing libprotobuf-lite.so.25.1.0),
# so we force MODULE mode. The module target protobuf::libprotobuf does not
# propagate absl, which normally shows up at link as
# "libabsl_status.so: DSO missing from command line". Instead of enumerating the
# ~100 absl libs, we let the linker follow libprotobuf.so's own DT_NEEDED absl
# entries via --copy-dt-needed-entries (see add_link_options below).
find_package(Protobuf MODULE REQUIRED)
set(NPUR_PROTOBUF_LIBS protobuf::libprotobuf)

# Extracted engine/ sources include <google/protobuf/json/json.h> (the newer
# protobuf JSON API). Older protobuf only ships util/json_util.h, so we carry a
# compat shim in src/compat/. But on NEWER protobuf the real header exists AND
# util/json_util.h forwards to google::protobuf::json::, so the shim would
# shadow the real header and break the build (circular using-decls). Only add
# the shim's include dir when the system protobuf does NOT provide json/json.h.
find_file(NPUR_SYSTEM_PB_JSON
    NAMES google/protobuf/json/json.h
    PATHS ${Protobuf_INCLUDE_DIRS} ${Protobuf_INCLUDE_DIR} /usr/include /usr/local/include)
if(NPUR_SYSTEM_PB_JSON)
    message(STATUS "protobuf json/json.h: system (${NPUR_SYSTEM_PB_JSON}); compat shim disabled")
    set(NPUR_PB_JSON_COMPAT_DIR "")
else()
    message(STATUS "protobuf json/json.h: not found; enabling compat shim in src/compat")
    set(NPUR_PB_JSON_COMPAT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/compat")
endif()

# Resolve absl (and any other transitive DSO deps of the libs we link) through
# their libprotobuf.so DT_NEEDED entries, rather than requiring each on our link
# line. Applies to all targets defined below (not the device ExternalProject).
add_link_options(-Wl,--copy-dt-needed-entries)

# ---- paths -----------------------------------------------------------------
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(ENGINE_DIR   "${REPO_ROOT}/src/engine")
set(STUBS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/platform")
set(DEVICE_DIR "${ENGINE_DIR}/retrieval/device")

# ---- include-root mapping (symlinks created at configure time) -------------
# The engine sources include both "src/full_recall/..." and "full_recall/...". Point
# both prefixes at ENGINE_DIR via symlinks in the build tree, so we never touch the
# engine source dir. Recreate unconditionally every configure: a plain NOT-EXISTS
# guard leaves a STALE link (pointing at a previous ENGINE_DIR) in place after the
# engine dir is moved/renamed, which breaks the build. file(REMOVE) on a symlink
# drops the link itself (even if dangling), then we relink to the current ENGINE_DIR.
set(INCLUDE_ROOT "${CMAKE_CURRENT_BINARY_DIR}/include_root")
file(MAKE_DIRECTORY "${INCLUDE_ROOT}/src")
file(REMOVE "${INCLUDE_ROOT}/src/full_recall")
file(CREATE_LINK "${ENGINE_DIR}" "${INCLUDE_ROOT}/src/full_recall" SYMBOLIC)
file(REMOVE "${INCLUDE_ROOT}/full_recall")
file(CREATE_LINK "${ENGINE_DIR}" "${INCLUDE_ROOT}/full_recall" SYMBOLIC)

# Stub include roots: `${STUBS_DIR}` resolves "src/utils/...", "murmurhash/...",
# "huawei_platform/...", "configuration/..."; `${STUBS_DIR}/src` also resolves
# the short forms "utils/...", "common/...", "configuration/..." used by `engine/`.
set(COMMON_INCLUDES
    "${INCLUDE_ROOT}"
    "${STUBS_DIR}"
    "${STUBS_DIR}/src"
    "${ASCEND_CANN_PACKAGE_PATH}/include"
)
# Only present on protobuf too old to ship google/protobuf/json/json.h.
if(NPUR_PB_JSON_COMPAT_DIR)
    list(APPEND COMMON_INCLUDES "${NPUR_PB_JSON_COMPAT_DIR}")
endif()

set(COMMON_DEFINES
    ASCEND_C_ENABLED                       # data_table_repository real (non-mock) path
    _GLIBCXX_USE_CXX11_ABI=${NPUR_GLIBCXX_ABI}
)

set(NPUR_EXTRA_RPATH_DIRS "" CACHE STRING
    "Semicolon-separated extra runtime library directories for host tools")
set(NPUR_HOST_RPATH_DIRS
    "${REPO_ROOT}/lib64"
    "${REPO_ROOT}/lib"
    "${REPO_ROOT}/third_party/lib64"
    "${REPO_ROOT}/third_party/lib"
    "${ASCEND_CANN_PACKAGE_PATH}/lib64"
    "${ASCEND_CANN_PACKAGE_PATH}/opp/vendors/aicpu_mask/op_impl/cpu/aicpu_kernel/impl"
    "/usr/local/lib64"
    "/usr/local/lib"
    "/usr/lib64"
    "/usr/lib"
    "/usr/lib/aarch64-linux-gnu"
    "/opt/OpenBLAS/lib"
    "/lib64"
    "/lib"
)
if(NPUR_EXTRA_RPATH_DIRS)
    list(APPEND NPUR_HOST_RPATH_DIRS ${NPUR_EXTRA_RPATH_DIRS})
endif()
list(REMOVE_DUPLICATES NPUR_HOST_RPATH_DIRS)
link_directories(${NPUR_HOST_RPATH_DIRS})
message(STATUS "Host runtime library dirs: ${NPUR_HOST_RPATH_DIRS}")

function(npur_apply_host_link_dirs target_name)
    target_link_directories(${target_name} PUBLIC ${NPUR_HOST_RPATH_DIRS})
endfunction()

function(npur_apply_executable_runtime_paths target_name)
    target_link_directories(${target_name} PRIVATE ${NPUR_HOST_RPATH_DIRS})
    foreach(_rpath_dir IN LISTS NPUR_HOST_RPATH_DIRS)
        if(IS_DIRECTORY "${_rpath_dir}")
            target_link_options(${target_name} PRIVATE "-Wl,-rpath,${_rpath_dir}")
        endif()
    endforeach()
endfunction()

# Force-include a prelude of common std headers into every `engine/` TU. The
# extracted code assumes these are pulled in transitively; GCC 12 disagrees.
# Set PUBLIC on npur_common so it propagates to all downstream targets.
set(NPUR_PRELUDE_OPT -include "${STUBS_DIR}/npur_prelude.h")

# =============================================================================
# 1. AscendC device kernels -> libascend_device.so + aclrtlaunch_*.h headers.
#
# The ascendc_library() ExternalProject build step re-runs on EVERY `cmake
# --build` and is non-idempotent (fails the 2nd time with "unknown file type"),
# which would force `rm -rf build` on every rebuild. To avoid that we PREFER a
# device lib prebuilt out-of-tree by src/device/CMakeLists.txt:
#
#   cmake -S src/device -B build_device -DASCEND_CANN_PACKAGE_PATH=... -DSOC_VERSION=...
#   cmake --build build_device -j
#
# If build_device/ is present we IMPORT it (no re-run, fast host iteration).
# Otherwise we fall back to building it inline here (works on a clean tree, but
# then every subsequent rebuild needs a clean build).
# =============================================================================
set(NPUR_DEVICE_BUILD "${REPO_ROOT}/build_device"
    CACHE PATH "prebuilt device build dir (from src/device); imported if present")
set(_dev_lib "${NPUR_DEVICE_BUILD}/lib/libascend_device.so")
set(_dev_inc "${NPUR_DEVICE_BUILD}/include")

if(EXISTS "${_dev_lib}")
    message(STATUS "Device: importing prebuilt ${_dev_lib}")
    add_library(ascend_device SHARED IMPORTED GLOBAL)
    set_target_properties(ascend_device PROPERTIES IMPORTED_LOCATION "${_dev_lib}")
    set(NPUR_DEVICE_INC_DIR "${_dev_inc}")
    set(NPUR_DEVICE_PREBUILT ON)
else()
    message(STATUS "Device: no prebuilt at ${_dev_lib}; building inline "
                   "(subsequent rebuilds will need a clean build — prebuild via src/device to avoid).")
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
    set(NPUR_DEVICE_INC_DIR "${CMAKE_BINARY_DIR}/include")
    set(NPUR_DEVICE_PREBUILT OFF)
endif()

# =============================================================================
# 2. protobuf: generate src/full_recall/indexer/proto/*.pb.{h,cc}
#    Run protoc from INCLUDE_ROOT so the emitted path matches the include
#    "src/full_recall/indexer/proto/document.pb.h".
# =============================================================================
set(PROTO_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/proto_gen")
file(MAKE_DIRECTORY "${PROTO_GEN_DIR}")
set(PROTO_FILES document index_builder index_meta)
set(PROTO_SRCS)
foreach(_p ${PROTO_FILES})
    set(_out_h "${PROTO_GEN_DIR}/src/full_recall/indexer/proto/${_p}.pb.h")
    set(_out_cc "${PROTO_GEN_DIR}/src/full_recall/indexer/proto/${_p}.pb.cc")
    add_custom_command(
        OUTPUT "${_out_h}" "${_out_cc}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${PROTO_GEN_DIR}/src/full_recall/indexer/proto"
        COMMAND ${Protobuf_PROTOC_EXECUTABLE}
                -I "${INCLUDE_ROOT}"
                --cpp_out "${PROTO_GEN_DIR}"
                "${INCLUDE_ROOT}/src/full_recall/indexer/proto/${_p}.proto"
        DEPENDS "${ENGINE_DIR}/indexer/proto/${_p}.proto"
        COMMENT "protoc ${_p}.proto"
        VERBATIM
    )
    list(APPEND PROTO_SRCS "${_out_cc}")
endforeach()
add_custom_target(npur_proto_gen DEPENDS ${PROTO_SRCS})
list(APPEND COMMON_INCLUDES "${PROTO_GEN_DIR}")

# =============================================================================
# 3. Host libraries (bottom-up). Split so the search runtime can build even if
#    the builder (rapidjson) needs more work.
# =============================================================================

# ---- npur_common: common + codec + proto + owned stubs --------------------
add_library(npur_common STATIC
    "${ENGINE_DIR}/core/hash/hash.cpp"
    "${ENGINE_DIR}/core/number/trans_number.cpp"
    "${ENGINE_DIR}/format/file_header.cpp"
    "${ENGINE_DIR}/format/posting_serializer.cpp"
    "${STUBS_DIR}/src/configuration/develop_configuration.cpp"
    "${STUBS_DIR}/murmurhash/MurmurHash3.cpp"
    ${PROTO_SRCS}
)
add_dependencies(npur_common npur_proto_gen)
target_include_directories(npur_common PUBLIC ${COMMON_INCLUDES})
target_compile_definitions(npur_common PUBLIC ${COMMON_DEFINES})
target_compile_options(npur_common PUBLIC ${NPUR_PRELUDE_OPT})
npur_apply_host_link_dirs(npur_common)
target_link_libraries(npur_common PUBLIC ${NPUR_PROTOBUF_LIBS} gflags Threads::Threads)

# ---- npur_search: data + query + searcher (the online runtime) ------------
# NOTE: data_table_repository.cpp is deliberately excluded. It is the multi-shard
# registry used only by process/ (which we bypass), and it references
# ParseIndexLoadCards/ParseDeviceIds that were not part of the extraction. The
# harness drives DataTable directly, so it is not needed. Re-add (with those
# Parse* helpers stubbed) if multi-shard serving is pursued later.
add_library(npur_search STATIC
    "${ENGINE_DIR}/index/data_table.cpp"
    "${ENGINE_DIR}/index/docid/doc_id_mapping.cpp"
    "${ENGINE_DIR}/index/posting/posting_data.cpp"
    "${ENGINE_DIR}/index/posting/posting_field_data.cpp"
    "${ENGINE_DIR}/index/vector/vector_data.cpp"
    "${ENGINE_DIR}/index/vector/vector_field_data.cpp"
    "${ENGINE_DIR}/retrieval/query/query_node.cpp"
    "${ENGINE_DIR}/retrieval/query/query_node_imp.cpp"
    "${ENGINE_DIR}/retrieval/searcher/full_recall_searcher.cpp"
    "${ENGINE_DIR}/retrieval/searcher/result_aggregator.cpp"
    "${ENGINE_DIR}/retrieval/searcher/text_filter.cpp"
    "${ENGINE_DIR}/retrieval/searcher/vector_scorer_mmad.cpp"
    "${ENGINE_DIR}/retrieval/searcher/runtime/gm_memory_manager.cpp"
    "${ENGINE_DIR}/retrieval/searcher/runtime/stream_manager.cpp"
)
# ascend_device provides the generated launch headers (ascend_device/aclrtlaunch_*.h)
# in ${NPUR_DEVICE_INC_DIR}, plus the kernel entry symbols in libascend_device.so.
target_include_directories(npur_search PUBLIC ${COMMON_INCLUDES} "${NPUR_DEVICE_INC_DIR}")
target_compile_definitions(npur_search PUBLIC ${COMMON_DEFINES})
if(NOT NPUR_DEVICE_PREBUILT)
    add_dependencies(npur_search ascend_device)  # inline build must finish first
endif()
# CANN host libs (ascendcl, tiling, register, platform, ...) live in <CANN>/lib64;
# add it to the link search path and embed as rpath so fr_search also runs.
set(CANN_LIB_DIR "${ASCEND_CANN_PACKAGE_PATH}/lib64")
# CANN libs present in this toolkit's lib64: libascendcl.so, libregister.so,
# libplatform.so, libascendalog.so, libtiling_api.a. There is NO libtiling.so
# here (the BUILD file listed -ltiling, but this CANN version folds it into
# tiling_api), so it is omitted.
target_link_libraries(npur_search PUBLIC
    npur_common
    ascend_device
    ascendcl register tiling_api platform ascendalog
    dl Threads::Threads
)
npur_apply_host_link_dirs(npur_search)

# ---- npur_build: offline index builder libraries --------------------------
# RapidJSON is header-only (yum: rapidjson-devel, or vendor headers + pass
# -DRAPIDJSON_INCLUDE_DIR=...). The wrapper needs fr_builder, so fail early with
# an actionable error instead of silently skipping the target.
if(NPUR_BUILD_BUILDER)
    if(RAPIDJSON_INCLUDE_DIR
       AND NOT EXISTS "${RAPIDJSON_INCLUDE_DIR}/rapidjson/document.h"
       AND EXISTS "${RAPIDJSON_INCLUDE_DIR}/include/rapidjson/document.h")
        set(RAPIDJSON_INCLUDE_DIR "${RAPIDJSON_INCLUDE_DIR}/include" CACHE PATH
            "RapidJSON include directory containing rapidjson/document.h" FORCE)
    endif()

    if(RAPIDJSON_INCLUDE_DIR)
        if(NOT EXISTS "${RAPIDJSON_INCLUDE_DIR}/rapidjson/document.h")
            message(FATAL_ERROR
                "RAPIDJSON_INCLUDE_DIR is '${RAPIDJSON_INCLUDE_DIR}', but "
                "rapidjson/document.h was not found under it. Pass the include "
                "root, for example /opt/rapidjson-1.1.0/include")
        endif()
    else()
        find_path(RAPIDJSON_INCLUDE_DIR
            NAMES rapidjson/document.h
            PATHS
                "${REPO_ROOT}/rapidjson/include"
                "${REPO_ROOT}/rapidjson-1.1.0/include"
                "${REPO_ROOT}/third_party/rapidjson/include"
                "${REPO_ROOT}/third_party/rapidjson-1.1.0/include"
                "${REPO_ROOT}/../rapidjson/include"
                "${REPO_ROOT}/../rapidjson-1.1.0/include"
                "/opt/rapidjson/include"
                "/opt/rapidjson-1.1.0/include"
        )
    endif()

    if(NOT RAPIDJSON_INCLUDE_DIR)
        message(FATAL_ERROR
            "RapidJSON headers not found. Put RapidJSON under hx_npu/rapidjson-1.1.0/include "
            "or pass -DRAPIDJSON_INCLUDE_DIR=/path/to/rapidjson/include.")
    endif()
    message(STATUS "RapidJSON: ${RAPIDJSON_INCLUDE_DIR}")
endif()
if(NPUR_BUILD_BUILDER)
    add_library(npur_build STATIC
        "${ENGINE_DIR}/indexer/file/file_reader.cpp"
        "${ENGINE_DIR}/indexer/file/file_writer.cpp"
        "${ENGINE_DIR}/indexer/file/memory_data_manager.cpp"
        "${ENGINE_DIR}/indexer/schema/builder_schema.cpp"
        "${ENGINE_DIR}/indexer/builder/builder_impl.cpp"
        "${ENGINE_DIR}/indexer/builder/idmapping_builder.cpp"
        "${ENGINE_DIR}/indexer/builder/inverted_builder.cpp"
        "${ENGINE_DIR}/indexer/builder/meta_builder.cpp"
        "${ENGINE_DIR}/indexer/builder/vector_builder.cpp"
    )
    target_include_directories(npur_build PUBLIC ${COMMON_INCLUDES} "${RAPIDJSON_INCLUDE_DIR}")
    target_compile_definitions(npur_build PUBLIC ${COMMON_DEFINES})
    npur_apply_host_link_dirs(npur_build)
    # The offline builder is CPU-only (no acl usage), so it does not link ascendcl.
    target_link_libraries(npur_build PUBLIC npur_common dl Threads::Threads)

    add_executable(fr_builder "${ENGINE_DIR}/indexer/builder/main.cpp")
    target_link_libraries(fr_builder PRIVATE npur_build)
    npur_apply_executable_runtime_paths(fr_builder)
endif()

# =============================================================================
# 4. Search harness executable (Phase 6). Gated on the driver existing.
# =============================================================================
if(NPUR_BUILD_SEARCH AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/harness/main.cpp")
    add_executable(fr_search "${CMAKE_CURRENT_SOURCE_DIR}/harness/main.cpp")
    target_link_libraries(fr_search PRIVATE npur_search npur_common)
    npur_apply_executable_runtime_paths(fr_search)
    # When importing a prebuilt device lib, embed its dir as rpath so the executable
    # finds libascend_device.so at runtime (CMake auto-adds this only for in-tree libs).
    if(NPUR_DEVICE_PREBUILT)
        target_link_options(fr_search PRIVATE "-Wl,-rpath,${NPUR_DEVICE_BUILD}/lib")
    endif()
elseif(NPUR_BUILD_SEARCH)
    message(STATUS "src/harness/main.cpp not present yet; skipping fr_search (Phase 6).")
endif()

# =============================================================================
# 5. Data converter (Phase 5). Standalone: only needs the generated proto +
#    protobuf; no CANN. Emits schema.json + per-segment builder input.
# =============================================================================
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/converter/build_input.cpp")
    add_executable(fr_converter
        "${CMAKE_CURRENT_SOURCE_DIR}/converter/build_input.cpp"
        ${PROTO_SRCS}
    )
    add_dependencies(fr_converter npur_proto_gen)
    target_include_directories(fr_converter PRIVATE ${COMMON_INCLUDES} "${PROTO_GEN_DIR}")
    target_compile_definitions(fr_converter PRIVATE _GLIBCXX_USE_CXX11_ABI=${NPUR_GLIBCXX_ABI})
    target_link_libraries(fr_converter PRIVATE ${NPUR_PROTOBUF_LIBS} Threads::Threads)
    npur_apply_executable_runtime_paths(fr_converter)
endif()
