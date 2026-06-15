#!/bin/bash
set -euo pipefail

# --- 核心配置 ---
RUN_MODE="npu"

# Auto-detect SOC version from npu-smi (override via SOC_VERSION env)
if [[ -z "${SOC_VERSION:-}" ]]; then
    _chip_name=$(npu-smi info 2>/dev/null | grep -oP '\b910B[1-4]\b' | head -1 || true)
    case "${_chip_name}" in
        910B1) SOC_VERSION="Ascend910B1" ;;
        910B2) SOC_VERSION="Ascend910B2" ;;
        910B3) SOC_VERSION="Ascend910B3" ;;
        910B4) SOC_VERSION="Ascend910B4" ;;
        *)     SOC_VERSION="Ascend910B3" ;;  # fallback
    esac
fi

# Auto-detect CANN toolkit path (first existing wins; override via ASCEND_INSTALL_PATH env)
if [[ -z "${ASCEND_INSTALL_PATH:-}" ]]; then
    for _candidate in \
        "/usr/local/Ascend/ascend-toolkit/latest" \
        "/home/lcy/Ascend/ascend-toolkit/latest"
    do
        if [[ -d "${_candidate}" ]]; then
            ASCEND_INSTALL_PATH="${_candidate}"
            break
        fi
    done
    if [[ -z "${ASCEND_INSTALL_PATH:-}" ]]; then
        echo "[ERROR] CANN toolkit not found. Set ASCEND_INSTALL_PATH env var."
        exit 1
    fi
fi
VARIANT="s"
PROFILE_MODE="0"
COMPILE_MODE="1"
PERF_FREQ=49
CPU_SAMPLE_INTERVAL=15
NPU_SAMPLE_INTERVAL=5
NPU_IDS=()
TOP_PID=""
NPU_MONITOR_PID=""
STACKCOLLAPSE=""
FLAMEGRAPH=""

usage() {
    cat <<'USAGE'
Usage: ./run.sh -v [s|p|a1|a2|a3|a8] -p [0|1] -c [0|1]

Options:
  -v, --variant <s|p|a1|a2|a3|a8>   executable variant: s=serial, p=parallel, a1=analyze_t1, a2=analyze_t2, a3=analyze_t3, a8=analyze_t8
  -p <0|1>              0=正常构建并运行；1=清空 ./profile 后执行 perf 采样并生成火焰图
  -c <0|1>              0=不编译，直接运行已有产物；1=先编译再运行
  -h, --help            show this help

Environment:
  FLAMEGRAPH_DIR        FlameGraph 目录。默认: <project>/FlameGraph
USAGE
}

extract_config_value() {
    local key="$1"
    local file="$2"
    local line
    line="$(grep -E "^[[:space:]]*${key}[[:space:]]*=" "${file}" | tail -n 1 || true)"
    line="$(printf '%s\n' "${line}" | sed -E 's/[[:space:]]*#.*$//; s/[[:space:]]*\/\/.*$//')"
    line="${line#*=}"
    printf '%s' "${line}" | sed -E "s/^[[:space:]]+//; s/[[:space:]]+$//; s/;$//; s/^['\"]//; s/['\"]$//"
}

resolve_result_root() {
    local raw_root="$1"
    if [[ -z "${raw_root}" ]]; then
        printf '\n'
        return 0
    fi

    if [[ "${raw_root}" = /* ]]; then
        printf '%s\n' "${raw_root}"
    else
        printf '%s\n' "${OUT_DIR}/bin/${raw_root}"
    fi
}

detect_available_npu_ids() {
    local summary
    local id
    local usage_out
    local detected_ids=()

    summary="$(npu-smi info 2>/dev/null || true)"

    if [[ -n "${summary}" ]]; then
        while IFS= read -r id; do
            [[ -z "${id}" ]] && continue
            usage_out="$(npu-smi info -t usages -i "${id}" 2>/dev/null || true)"
            if [[ -n "${usage_out}" && "${usage_out}" == *"NPU ID"* ]]; then
                detected_ids+=("${id}")
            fi
        done < <(
            printf '%s\n' "${summary}" \
                | awk '{for (i = 1; i <= NF; ++i) if ($i == "NPU" && (i + 1) <= NF && $(i + 1) ~ /^[0-9]+$/) print $(i + 1)}' \
                | sort -n -u
        )
    fi

    if [[ "${#detected_ids[@]}" -eq 0 ]]; then
        for id in $(seq 0 15); do
            usage_out="$(npu-smi info -t usages -i "${id}" 2>/dev/null || true)"
            if [[ -n "${usage_out}" && "${usage_out}" == *"NPU ID"* ]]; then
                detected_ids+=("${id}")
            fi
        done
    fi

    printf '%s\n' "${detected_ids[@]}"
}

select_npu_ids_for_variant() {
    local available_ids=()

    mapfile -t available_ids < <(detect_available_npu_ids)

    if [[ "${#available_ids[@]}" -eq 0 ]]; then
        echo "[ERROR] No available NPU IDs detected by npu-smi."
        exit 1
    fi

    local config_file_path="${CONFIG_FILE}"

    local required_count
    local device_id_start

    if [[ "${VARIANT}" == "p" ]]; then
        required_count="$(extract_config_value "npu_device_count_parallel" "${config_file_path}")"
    else
        required_count="$(extract_config_value "npu_device_count_serial" "${config_file_path}")"
    fi
    device_id_start="$(extract_config_value "npu_device_id_start" "${config_file_path}")"

    if [[ -z "${required_count}" ]]; then
        echo "[ERROR] Failed to read npu_device_count from config.txt"
        exit 1
    fi
    if [[ -z "${device_id_start}" ]]; then
        echo "[ERROR] Failed to read npu_device_id_start from config.txt"
        exit 1
    fi

    # shellcheck disable=SC2015
    [[ "${required_count}" =~ ^[0-9]+$ ]] && [[ "${required_count}" -gt 0 ]] || {
        echo "[ERROR] Invalid npu_device_count value: '${required_count}'"
        exit 1
    }
    # shellcheck disable=SC2015
    [[ "${device_id_start}" =~ ^[0-9]+$ ]] && [[ "${device_id_start}" -ge 0 ]] || {
        echo "[ERROR] Invalid npu_device_id_start value: '${device_id_start}'"
        exit 1
    }

    local available_count="${#available_ids[@]}"
    if [[ "${device_id_start}" -ge "${available_count}" ]]; then
        echo "[ERROR] npu_device_id_start=${device_id_start} exceeds available NPU count=${available_count}."
        echo "        Available NPU IDs: ${available_ids[*]}"
        exit 1
    fi

    local end_index=$((device_id_start + required_count))
    if [[ "${end_index}" -gt "${available_count}" ]]; then
        echo "[ERROR] npu_device_id_start(${device_id_start}) + npu_device_count(${required_count}) = ${end_index} exceeds available NPU count=${available_count}."
        echo "        Available NPU IDs: ${available_ids[*]}"
        exit 1
    fi

    NPU_IDS=("${available_ids[@]:${device_id_start}:${required_count}}")
}

start_cpu_monitor() {
    local app_pid="$1"
    local top_output_file="$2"
    LC_ALL=C top -b -d "${CPU_SAMPLE_INTERVAL}" -p "${app_pid}" > "${top_output_file}" 2>/dev/null &
    TOP_PID=$!
}

stop_cpu_monitor() {
    if [[ -n "${TOP_PID}" ]] && kill -0 "${TOP_PID}" 2>/dev/null; then
        kill "${TOP_PID}" 2>/dev/null || true
        wait "${TOP_PID}" 2>/dev/null || true
    fi
    TOP_PID=""
}

start_npu_monitor() {
    local npu_usage_file="$1"
    shift
    local npu_ids=("$@")

    if ! command -v npu-smi >/dev/null 2>&1; then
        echo "[ERROR] npu-smi command not found."
        exit 1
    fi

    mkdir -p "$(dirname "${npu_usage_file}")"
    : > "${npu_usage_file}"

    (
        local warned_once=0
        while true; do
            local total_aicore=0
            local raw_usage
            local aicore

            for npu_id in "${npu_ids[@]}"; do
                raw_usage="$(npu-smi info -t usages -i "${npu_id}" 2>/dev/null)" || raw_usage=""

                if [[ -z "${raw_usage}" ]]; then
                    if [[ "${warned_once}" -eq 0 ]]; then
                        echo "[WARN] Failed to read NPU usages from npu-smi. Missing samples will be recorded as 0." >&2
                        warned_once=1
                    fi
                    continue
                fi

                aicore="$(printf '%s\n' "${raw_usage}" | awk -F: '/Aicore Usage Rate\(%\)/ {gsub(/[[:space:]]/, "", $2); print $2; exit}')"
                total_aicore=$((total_aicore + ${aicore:-0}))
            done

            printf '%s\n' "${total_aicore}" >> "${npu_usage_file}"

            sleep "${NPU_SAMPLE_INTERVAL}" || break
        done
    ) &
    NPU_MONITOR_PID=$!
}

stop_npu_monitor() {
    if [[ -n "${NPU_MONITOR_PID}" ]] && kill -0 "${NPU_MONITOR_PID}" 2>/dev/null; then
        kill "${NPU_MONITOR_PID}" 2>/dev/null || true
        wait "${NPU_MONITOR_PID}" 2>/dev/null || true
    fi
    NPU_MONITOR_PID=""
}

write_cpu_usage_log() {
    local app_pid="$1"
    local top_output_file="$2"
    local cpu_usage_file="$3"

    mkdir -p "$(dirname "${cpu_usage_file}")"

    awk -v pid="${app_pid}" '
        /^[[:space:]]*PID[[:space:]]+/ {
            cpu_col = 0
            for (i = 1; i <= NF; ++i) {
                if ($i == "%CPU" || $i == "CPU%") {
                    cpu_col = i
                    break
                }
            }
            next
        }
        cpu_col > 0 && $1 == pid {
            print $cpu_col
        }
    ' "${top_output_file}" > "${cpu_usage_file}"

    if [[ ! -s "${cpu_usage_file}" ]]; then
        echo "[WARN] No CPU usage samples were captured by top." >&2
    fi
}

resolve_flamegraph_tools() {
    local flamegraph_dir="$1"

    if [[ -x "${flamegraph_dir}/stackcollapse-perf.pl" && -x "${flamegraph_dir}/flamegraph.pl" ]]; then
        STACKCOLLAPSE="${flamegraph_dir}/stackcollapse-perf.pl"
        FLAMEGRAPH="${flamegraph_dir}/flamegraph.pl"
        return 0
    fi

    if command -v stackcollapse-perf.pl >/dev/null 2>&1 && command -v flamegraph.pl >/dev/null 2>&1; then
        STACKCOLLAPSE="$(command -v stackcollapse-perf.pl)"
        FLAMEGRAPH="$(command -v flamegraph.pl)"
        return 0
    fi

    echo "[ERROR] FlameGraph scripts not found."
    echo "        Expected either:"
    echo "        1) ${flamegraph_dir}/{stackcollapse-perf.pl,flamegraph.pl}"
    echo "        2) stackcollapse-perf.pl and flamegraph.pl in PATH"
    exit 1
}

run_normal_mode() {
    local executable_name="$1"
    local result_root_path="$2"
    local top_output_file
    local cpu_usage_file
    local npu_usage_file
    local app_pid
    local app_exit_code

    if ! command -v top >/dev/null 2>&1; then
        echo "[ERROR] top command not found."
        exit 1
    fi
    if ! command -v npu-smi >/dev/null 2>&1; then
        echo "[ERROR] npu-smi command not found."
        exit 1
    fi

    top_output_file="$(mktemp /tmp/hybrid_top_XXXXXX.log)"
    cpu_usage_file="${result_root_path}/log/cpu_usage.txt"
    npu_usage_file="${result_root_path}/log/npu_usages.txt"

    pushd "${OUT_DIR}/bin" >/dev/null
    "./${executable_name}" &
    app_pid=$!
    start_cpu_monitor "${app_pid}" "${top_output_file}"
    start_npu_monitor "${npu_usage_file}" "${NPU_IDS[@]}"

    set +e
    wait "${app_pid}"
    app_exit_code=$?
    set -e
    popd >/dev/null

    stop_cpu_monitor
    stop_npu_monitor
    write_cpu_usage_log "${app_pid}" "${top_output_file}" "${cpu_usage_file}"
    rm -f "${top_output_file}"

    # CANN cleanup may segfault after all results are written (cosmetic).
    # If result files exist, treat as success.
    if [[ "${app_exit_code}" -ne 0 ]]; then
        local has_results="false"
        if compgen -G "${RESULT_ROOT_PATH}/recall/*" >/dev/null 2>&1; then
            has_results="true"
        fi
        if [[ "${has_results}" == "true" ]]; then
            echo ">>> Application exited with code ${app_exit_code} but results exist — treating as success (CANN cleanup segfault is known)."
        else
            # Dump CANN GE debug logs on failure
            echo ">>> Application failed (exit ${app_exit_code}). Dumping CANN GE logs..."
            local _ge_log_dirs=()
            for _d in "${HOME}/ascend/log/debug/plog" "${ASCEND_INSTALL_PATH}/../log/debug/plog"; do
                [[ -d "${_d}" ]] && _ge_log_dirs+=("${_d}")
            done
            if [[ ${#_ge_log_dirs[@]} -gt 0 ]]; then
                for _d in "${_ge_log_dirs[@]}"; do
                    local _latest
                    _latest="$(find "${_d}" -name 'plog-*.log' -type f -mmin -5 2>/dev/null | head -3)"
                    if [[ -n "${_latest}" ]]; then
                        echo ">>> GE log dir: ${_d}"
                        while IFS= read -r _f; do
                            echo "--- $(basename "${_f}") (last 50 lines) ---"
                            tail -50 "${_f}"
                        done <<< "${_latest}"
                    fi
                done
            else
                echo ">>> No GE log directory found. Searched: ${HOME}/ascend/log/debug/plog"
            fi
            # Print CANN env vars for diagnostics
            echo ">>> CANN environment:"
            for _v in ASCEND_HOME_PATH ASCEND_TOOLKIT_HOME ASCEND_OPP_PATH ASCEND_AICPU_PATH ASCEND_OPP_BUILT_IN; do
                echo "    ${_v}=${!_v:-<not set>}"
            done
            exit "${app_exit_code}"
        fi
    fi
}

run_profile_mode() {
    local executable_name="$1"
    local result_root_path="$2"
    local flamegraph_dir="$3"
    local profile_dir="$4"
    local run_name="${executable_name}_profile"
    local data_file="${profile_dir}/${run_name}.data"
    local perf_txt_file="${profile_dir}/${run_name}.perf"
    local folded_file="${profile_dir}/${run_name}.folded"
    local svg_file="${profile_dir}/${run_name}.svg"
    local validate_err_file="${profile_dir}/${run_name}.perf_validate.err"
    local top_output_file
    local cpu_usage_file="${result_root_path}/log/cpu_usage.txt"
    local npu_usage_file="${result_root_path}/log/npu_usages.txt"
    local target_pid_file
    local perf_stderr_file
    local app_pid=""
    local perf_pid
    local perf_rc
    local validate_rc

    command -v perf >/dev/null 2>&1 || {
        echo "[ERROR] 'perf' not found. Install perf first."
        exit 1
    }
    command -v bash >/dev/null 2>&1 || {
        echo "[ERROR] 'bash' not found."
        exit 1
    }
    if ! command -v top >/dev/null 2>&1; then
        echo "[ERROR] top command not found."
        exit 1
    fi
    if ! command -v npu-smi >/dev/null 2>&1; then
        echo "[ERROR] npu-smi command not found."
        exit 1
    fi

    resolve_flamegraph_tools "${flamegraph_dir}"
    mkdir -p "${profile_dir}"

    echo "[Info] FlameGraph dir : ${flamegraph_dir}"
    echo "[Info] Output profile : ${profile_dir}"
    echo "[Info] Sample freq    : ${PERF_FREQ} Hz"
    echo "[Info] Program output : live on terminal"

    rm -f "${data_file}" "${perf_txt_file}" "${folded_file}" "${svg_file}" \
          "${validate_err_file}"

    top_output_file="$(mktemp /tmp/hybrid_top_XXXXXX.log)"
    target_pid_file="$(mktemp /tmp/hybrid_target_pid_XXXXXX)"
    perf_stderr_file="$(mktemp /tmp/hybrid_perf_stderr_XXXXXX.log)"

    pushd "${OUT_DIR}/bin" >/dev/null
    set +e
    (
        ulimit -c 0
        perf record -F "${PERF_FREQ}" -g --call-graph dwarf --all-user \
            -o "${data_file}" -- \
            bash -lc "echo \$\$ > '${target_pid_file}'; exec './${executable_name}'"
    ) 2> >(tee "${perf_stderr_file}" >&2) &
    perf_pid=$!
    set -e
    start_npu_monitor "${npu_usage_file}" "${NPU_IDS[@]}"

    for _ in $(seq 1 100); do
        if [[ -s "${target_pid_file}" ]]; then
            read -r app_pid < "${target_pid_file}"
            break
        fi
        if ! kill -0 "${perf_pid}" 2>/dev/null; then
            break
        fi
        sleep 0.1
    done

    if [[ -n "${app_pid}" ]]; then
        start_cpu_monitor "${app_pid}" "${top_output_file}"
    else
        echo "[WARN] Failed to capture target PID for top monitoring."
    fi

    set +e
    wait "${perf_pid}"
    perf_rc=$?
    set -e
    popd >/dev/null

    stop_cpu_monitor
    stop_npu_monitor
    if [[ -n "${app_pid}" ]]; then
        write_cpu_usage_log "${app_pid}" "${top_output_file}" "${cpu_usage_file}"
    fi
    rm -f "${top_output_file}" "${target_pid_file}"

    if grep -qi "No space left on device" "${perf_stderr_file}"; then
        rm -f "${perf_stderr_file}"
        echo "[ERROR] perf record failed: No space left on device."
        echo "[HINT] Clean old profile files or free disk, then rerun."
        exit 28
    fi

    if [[ "${perf_rc}" -ne 0 ]]; then
        rm -f "${perf_stderr_file}"
        echo "[ERROR] Profiled execution failed with status: ${perf_rc}"
        echo "[HINT] See terminal output above for perf/program diagnostics."
        exit "${perf_rc}"
    fi

    if [[ ! -s "${data_file}" ]]; then
        rm -f "${perf_stderr_file}"
        echo "[ERROR] No perf data generated: ${data_file}"
        exit 1
    fi

    echo "[Step] Validating perf data..."
    set +e
    perf script --no-inline -i "${data_file}" >/dev/null 2> "${validate_err_file}"
    validate_rc=$?
    set -e
    if [[ "${validate_rc}" -ne 0 ]] || grep -q "data size field is 0" "${validate_err_file}"; then
        rm -f "${perf_stderr_file}"
        echo "[ERROR] perf data is invalid or incomplete."
        echo "[HINT] See validation error: ${validate_err_file}"
        exit 2
    fi

    echo "[Step] Converting perf data..."
    perf script --no-inline -i "${data_file}" > "${perf_txt_file}"

    echo "[Step] Folding stacks..."
    "${STACKCOLLAPSE}" "${perf_txt_file}" > "${folded_file}"

    echo "[Step] Rendering flame graph..."
    "${FLAMEGRAPH}" "${folded_file}" > "${svg_file}"

    rm -f "${perf_stderr_file}"
    echo "[Done] Flame graph generated: ${svg_file}"
    echo "[Done] Perf data retained : ${data_file}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -v|--variant)
            if [[ $# -lt 2 ]]; then
                usage
                exit 1
            fi
            VARIANT="$2"
            shift 2
            ;;
        -p)
            if [[ $# -lt 2 ]]; then
                usage
                exit 1
            fi
            PROFILE_MODE="$2"
            shift 2
            ;;
        -c)
            if [[ $# -lt 2 ]]; then
                usage
                exit 1
            fi
            COMPILE_MODE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ "${VARIANT}" != "s" && "${VARIANT}" != "p" && "${VARIANT}" != "a1" && "${VARIANT}" != "a2" && "${VARIANT}" != "a3" && "${VARIANT}" != "a8" ]]; then
    usage
    exit 1
fi

if [[ "${PROFILE_MODE}" != "0" && "${PROFILE_MODE}" != "1" ]]; then
    echo "[ERROR] -p only accepts 0 or 1."
    usage
    exit 1
fi

if [[ "${COMPILE_MODE}" != "0" && "${COMPILE_MODE}" != "1" ]]; then
    echo "[ERROR] -c only accepts 0 or 1."
    usage
    exit 1
fi

if [[ "${VARIANT}" == "s" ]]; then
    EXECUTABLE_NAME="serial"
elif [[ "${VARIANT}" == "p" ]]; then
    EXECUTABLE_NAME="parallel"
elif [[ "${VARIANT}" == "a1" ]]; then
    EXECUTABLE_NAME="analyze_t1"
elif [[ "${VARIANT}" == "a2" ]]; then
    EXECUTABLE_NAME="analyze_t2"
elif [[ "${VARIANT}" == "a3" ]]; then
    EXECUTABLE_NAME="analyze_t3"
elif [[ "${VARIANT}" == "a8" ]]; then
    EXECUTABLE_NAME="analyze_t8"
fi

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE:-$0}")" && pwd)"
CONFIG_FILE="${CURRENT_DIR}/config.txt"

select_npu_ids_for_variant

if [[ "${PROFILE_MODE}" == "1" ]]; then
    BUILD_TYPE="RelWithDebInfo"
    ENABLE_PROFILING_BUILD="ON"
else
    BUILD_TYPE="Release"
    ENABLE_PROFILING_BUILD="OFF"
fi

BUILD_DIR="${CURRENT_DIR}/build"
OUT_DIR="${CURRENT_DIR}/out"
PROFILE_DIR="${CURRENT_DIR}/profile"
FLAMEGRAPH_DIR="${FLAMEGRAPH_DIR:-${CURRENT_DIR}/FlameGraph}"

echo "========================================"
echo "Target:     ${SOC_VERSION}"
echo "Mode:       ${RUN_MODE}"
echo "Build:      ${BUILD_TYPE}"
echo "Variant:    ${EXECUTABLE_NAME} ($(case "${VARIANT}" in s|a1|a2|a3|a8) echo 'serial';; p) echo 'parallel';; esac))"
echo "Compile:    ${COMPILE_MODE}"
echo "Profile:    ${PROFILE_MODE}"
echo "NPU IDs:    ${NPU_IDS[*]}"
echo "========================================"

export ASCEND_HOME_PATH="${ASCEND_INSTALL_PATH}"

if [[ -f "${ASCEND_INSTALL_PATH}/bin/setenv.bash" ]]; then
    set +u
    set +e
    # shellcheck disable=SC1090
    source "${ASCEND_INSTALL_PATH}/bin/setenv.bash"
    set -e
    set -u
fi

export LD_LIBRARY_PATH="${OUT_DIR}/lib:/usr/local/lib:/usr/local/lib64:/usr/lib64:/opt/OpenBLAS/lib:/lib:${ASCEND_INSTALL_PATH}/opp/vendors/aicpu_mask/op_impl/cpu/aicpu_kernel/impl:${LD_LIBRARY_PATH:-}"

if [[ "${PROFILE_MODE}" == "1" ]]; then
    echo ">>> Cleaning profile directory: ${PROFILE_DIR}"
    rm -rf "${PROFILE_DIR}"
fi

if [[ "${COMPILE_MODE}" == "1" ]]; then
    rm -rf "${BUILD_DIR}" "${OUT_DIR}"
    mkdir -p "${BUILD_DIR}" "${OUT_DIR}"

    cmake -B "${BUILD_DIR}" \
        -DRUN_MODE="${RUN_MODE}" \
        -DSOC_VERSION="${SOC_VERSION}" \
        -DASCEND_CANN_PACKAGE_PATH="${ASCEND_INSTALL_PATH}" \
        -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
        -DENABLE_PROFILING_BUILD="${ENABLE_PROFILING_BUILD}" \
        -DCMAKE_INSTALL_PREFIX="${OUT_DIR}"

    echo ">>> Compiling..."
    cmake --build "${BUILD_DIR}" -j8
    cmake --install "${BUILD_DIR}"

    # Deploy vendor files (op_proto, AI CPU kernel .so, kernel.json) to CANN
    echo ">>> Installing vendor files to CANN..."
    VENDOR_SRC="${CURRENT_DIR}/include/NPU/vendor"
    CANN_VENDOR="${ASCEND_INSTALL_PATH}/opp/vendors/aicpu_mask"

    mkdir -p "${CANN_VENDOR}/op_proto/inc"
    mkdir -p "${CANN_VENDOR}/op_proto/lib/linux/aarch64"
    mkdir -p "${CANN_VENDOR}/op_impl/cpu/aicpu_kernel/impl"
    mkdir -p "${CANN_VENDOR}/op_impl/cpu/config"

    # Register in vendors/config.ini (idempotent)
    # Ensure aicpu_mask is listed and no conflicting vendor registers MaskFilter
    VENDORS_INI="${ASCEND_INSTALL_PATH}/opp/vendors/config.ini"
    if [[ ! -f "${VENDORS_INI}" ]]; then
        echo "load_priority=aicpu_mask" > "${VENDORS_INI}"
    elif ! grep -q 'aicpu_mask' "${VENDORS_INI}" 2>/dev/null; then
        existing="$(head -1 "${VENDORS_INI}" | sed 's/[[:space:]]*$//')"
        echo "${existing},aicpu_mask" > "${VENDORS_INI}"
    fi
    # Remove sks_hw from vendors if present (conflicts with aicpu_mask for MaskFilter)
    if grep -q 'sks_hw' "${VENDORS_INI}" 2>/dev/null; then
        existing="$(head -1 "${VENDORS_INI}" | sed 's/[[:space:]]*$//')"
        cleaned="$(echo "${existing}" | tr ',' '\n' | grep -v 'sks_hw' | tr '\n' ',' | sed 's/,$//')"
        echo "${cleaned}" > "${VENDORS_INI}"
        echo ">>> Removed conflicting sks_hw vendor from config.ini"
    fi

    # Copy kernel config
    cp -f "${VENDOR_SRC}/op_impl/cpu/config/cust_aicpu_kernel.json" "${CANN_VENDOR}/op_impl/cpu/config/" && \
        echo ">>> kernel config deployed: $(wc -c < "${CANN_VENDOR}/op_impl/cpu/config/cust_aicpu_kernel.json") bytes" || \
        echo ">>> WARNING: kernel config copy failed"
    chmod 644 "${CANN_VENDOR}/op_impl/cpu/config/cust_aicpu_kernel.json"

    # Copy AI CPU kernel .so
    KERNEL_SO="${OUT_DIR}/bin/libcust_aicpu_kernels_sks_final.so"
    if [[ -f "${KERNEL_SO}" ]]; then
        cp -f "${KERNEL_SO}" "${CANN_VENDOR}/op_impl/cpu/aicpu_kernel/impl/" && \
            echo ">>> AI CPU kernel .so deployed: $(wc -c < "${KERNEL_SO}") bytes" || \
            echo ">>> WARNING: kernel .so copy failed"
        chmod 755 "${CANN_VENDOR}/op_impl/cpu/aicpu_kernel/impl/libcust_aicpu_kernels_sks_final.so"
    else
        echo ">>> WARNING: AI CPU kernel .so not found at ${KERNEL_SO}"
        # Try build directory as fallback
        KERNEL_SO_BUILD="${BUILD_DIR}/libcust_aicpu_kernels_sks_final.so"
        if [[ -f "${KERNEL_SO_BUILD}" ]]; then
            cp -f "${KERNEL_SO_BUILD}" "${CANN_VENDOR}/op_impl/cpu/aicpu_kernel/impl/" && \
                echo ">>> AI CPU kernel .so deployed from build dir: $(wc -c < "${KERNEL_SO_BUILD}") bytes"
            chmod 755 "${CANN_VENDOR}/op_impl/cpu/aicpu_kernel/impl/libcust_aicpu_kernels_sks_final.so"
        else
            echo ">>> ERROR: AI CPU kernel .so not found in build dir either!"
        fi
    fi

    # --- Compile and deploy op_proto with CANN ABI=0 ---
    # Must compile with -D_GLIBCXX_USE_CXX11_ABI=0 to match libgraph.so ABI.
    OP_PROTO_SRC="${VENDOR_SRC}/op_proto/inc/op_proto.h"
    OP_PROTO_CC="/tmp/_sks_op_proto_compile.cpp"
    OP_PROTO_SO="${CANN_VENDOR}/op_proto/libcust_op_proto.so"
    if [[ -f "${OP_PROTO_SRC}" ]]; then
        echo '#include "op_proto.h"' > "${OP_PROTO_CC}"
        CANN_INC="${ASCEND_INSTALL_PATH}/aarch64-linux/include"
        CANN_LIB="${ASCEND_INSTALL_PATH}/aarch64-linux/lib64"
        g++ -shared -fPIC -o "${OP_PROTO_SO}" "${OP_PROTO_CC}" \
            -I"$(dirname "${OP_PROTO_SRC}")" \
            -I"${CANN_INC}" \
            -L"${CANN_LIB}" -lgraph \
            -std=c++17 -O2 \
            -D_GLIBCXX_USE_CXX11_ABI=0 \
            -Wl,-rpath,"${CANN_LIB}" 2>&1 && \
            echo ">>> op_proto deployed (ABI=0)" || \
            echo ">>> WARNING: op_proto compilation failed"
        chmod 755 "${OP_PROTO_SO}" 2>/dev/null
        # Also copy to lib/linux/aarch64/ — some CANN builds expect it there
        cp -f "${OP_PROTO_SO}" "${CANN_VENDOR}/op_proto/lib/linux/aarch64/" 2>/dev/null
        rm -f "${OP_PROTO_CC}"

        # Copy op_proto.h to vendor inc
        cp -f "${OP_PROTO_SRC}" "${CANN_VENDOR}/op_proto/inc/" 2>/dev/null

        # Write version.info
        echo "custom_opp_compiler_version=8.3.0.2.220" > "${CANN_VENDOR}/version.info"

        # Deploy ops-info.json (required by CANN for custom op registration)
        OPS_INFO_DIR="${CANN_VENDOR}/op_impl/ai_core/tbe/config/${SOC_VERSION}"
        mkdir -p "${OPS_INFO_DIR}"
        cat > "${OPS_INFO_DIR}/aic-${SOC_VERSION}-ops-info.json" <<'OPSJSON'
{
    "MaskFilter": {
        "coreType": {
            "value": "AiCpu"
        },
        "input0": {
            "dtype": "float32",
            "format": "ND",
            "name": "scores",
            "paramType": "required",
            "shape": "all"
        },
        "input1": {
            "dtype": "uint64",
            "format": "ND",
            "name": "masks",
            "paramType": "required",
            "shape": "all"
        },
        "input2": {
            "dtype": "float32",
            "format": "ND",
            "name": "output_buf",
            "paramType": "required",
            "shape": "all"
        },
        "input3": {
            "dtype": "int32",
            "format": "ND",
            "name": "index_buf",
            "paramType": "required",
            "shape": "all"
        },
        "input4": {
            "dtype": "int32",
            "format": "ND",
            "name": "count_buf",
            "paramType": "required",
            "shape": "all"
        },
        "input5": {
            "dtype": "uint64",
            "format": "ND",
            "name": "task_buf",
            "paramType": "required",
            "shape": "all"
        },
        "output0": {
            "dtype": "float32",
            "format": "ND",
            "name": "result",
            "paramType": "required",
            "shape": "all"
        }
    }
}
OPSJSON
        chmod 644 "${OPS_INFO_DIR}/aic-${SOC_VERSION}-ops-info.json"
        echo ">>> ops-info.json deployed to ${OPS_INFO_DIR}/"
    fi

    # Verify deployment
    echo ">>> Vendor deployment verification:"
    echo "    config.ini: $(cat "${VENDORS_INI}" 2>/dev/null)"
    echo "    kernel.json: $(ls -la "${CANN_VENDOR}/op_impl/cpu/config/cust_aicpu_kernel.json" 2>/dev/null || echo MISSING)"
    echo "    kernel.so: $(ls -la "${CANN_VENDOR}/op_impl/cpu/aicpu_kernel/impl/libcust_aicpu_kernels_sks_final.so" 2>/dev/null || echo MISSING)"
    echo "    op_proto.so: $(ls -la "${CANN_VENDOR}/op_proto/libcust_op_proto.so" 2>/dev/null || echo MISSING)"
    echo "    ops-info.json: $(ls -la "${CANN_VENDOR}/op_impl/ai_core/tbe/config/${SOC_VERSION}/aic-${SOC_VERSION}-ops-info.json" 2>/dev/null || echo MISSING)"

    echo ">>> Vendor files installed to ${CANN_VENDOR}"
else
    echo ">>> Skip compile due to -c 0. Reuse existing out/ artifacts."

    # Always deploy vendor files even in -c 0 mode
    VENDOR_SRC="${CURRENT_DIR}/include/NPU/vendor"
    CANN_VENDOR="${ASCEND_INSTALL_PATH}/opp/vendors/aicpu_mask"
    if [[ -f "${VENDOR_SRC}/op_proto/inc/op_proto.h" ]]; then
        echo ">>> Deploying vendor files to CANN (no-compile mode)..."
        mkdir -p "${CANN_VENDOR}/op_proto/inc"
        mkdir -p "${CANN_VENDOR}/op_proto/lib/linux/aarch64"
        mkdir -p "${CANN_VENDOR}/op_impl/cpu/aicpu_kernel/impl"
        mkdir -p "${CANN_VENDOR}/op_impl/cpu/config"

        # Register in vendors/config.ini (idempotent)
        VENDORS_INI="${ASCEND_INSTALL_PATH}/opp/vendors/config.ini"
        if [[ ! -f "${VENDORS_INI}" ]]; then
            echo "load_priority=aicpu_mask" > "${VENDORS_INI}"
        elif ! grep -q 'aicpu_mask' "${VENDORS_INI}" 2>/dev/null; then
            existing="$(head -1 "${VENDORS_INI}" | sed 's/[[:space:]]*$//')"
            echo "${existing},aicpu_mask" > "${VENDORS_INI}"
        fi
        # Remove conflicting sks_hw vendor
        if grep -q 'sks_hw' "${VENDORS_INI}" 2>/dev/null; then
            existing="$(head -1 "${VENDORS_INI}" | sed 's/[[:space:]]*$//')"
            cleaned="$(echo "${existing}" | tr ',' '\n' | grep -v 'sks_hw' | tr '\n' ',' | sed 's/,$//')"
            echo "${cleaned}" > "${VENDORS_INI}"
        fi

        # Copy kernel config
        cp -f "${VENDOR_SRC}/op_impl/cpu/config/cust_aicpu_kernel.json" "${CANN_VENDOR}/op_impl/cpu/config/" 2>/dev/null && \
            echo ">>> kernel config deployed" || echo ">>> WARNING: kernel config copy failed"
        chmod 644 "${CANN_VENDOR}/op_impl/cpu/config/cust_aicpu_kernel.json" 2>/dev/null
        KERNEL_SO="${OUT_DIR}/bin/libcust_aicpu_kernels_sks_final.so"
        if [[ -f "${KERNEL_SO}" ]]; then
            cp -f "${KERNEL_SO}" "${CANN_VENDOR}/op_impl/cpu/aicpu_kernel/impl/" && \
                echo ">>> AI CPU kernel .so deployed: $(wc -c < "${KERNEL_SO}") bytes" || \
                echo ">>> WARNING: kernel .so copy failed"
            chmod 755 "${CANN_VENDOR}/op_impl/cpu/aicpu_kernel/impl/libcust_aicpu_kernels_sks_final.so"
        else
            echo ">>> WARNING: AI CPU kernel .so not found at ${KERNEL_SO}"
            KERNEL_SO_BUILD="${BUILD_DIR}/libcust_aicpu_kernels_sks_final.so"
            if [[ -f "${KERNEL_SO_BUILD}" ]]; then
                cp -f "${KERNEL_SO_BUILD}" "${CANN_VENDOR}/op_impl/cpu/aicpu_kernel/impl/" && \
                    echo ">>> AI CPU kernel .so deployed from build dir"
                chmod 755 "${CANN_VENDOR}/op_impl/cpu/aicpu_kernel/impl/libcust_aicpu_kernels_sks_final.so"
            else
                echo ">>> ERROR: AI CPU kernel .so not found anywhere!"
            fi
        fi

        # Compile and deploy op_proto with CANN ABI=0
        OP_PROTO_SRC="${VENDOR_SRC}/op_proto/inc/op_proto.h"
        OP_PROTO_CC="/tmp/_sks_op_proto_compile.cpp"
        OP_PROTO_SO="${CANN_VENDOR}/op_proto/libcust_op_proto.so"
        echo '#include "op_proto.h"' > "${OP_PROTO_CC}"
        CANN_INC="${ASCEND_INSTALL_PATH}/aarch64-linux/include"
        CANN_LIB="${ASCEND_INSTALL_PATH}/aarch64-linux/lib64"
        g++ -shared -fPIC -o "${OP_PROTO_SO}" "${OP_PROTO_CC}" \
            -I"$(dirname "${OP_PROTO_SRC}")" \
            -I"${CANN_INC}" \
            -L"${CANN_LIB}" -lgraph \
            -std=c++17 -O2 \
            -D_GLIBCXX_USE_CXX11_ABI=0 \
            -Wl,-rpath,"${CANN_LIB}" 2>&1 && \
            echo ">>> op_proto deployed (ABI=0)" || \
            echo ">>> WARNING: op_proto compilation failed"
        chmod 755 "${OP_PROTO_SO}" 2>/dev/null
        cp -f "${OP_PROTO_SO}" "${CANN_VENDOR}/op_proto/lib/linux/aarch64/" 2>/dev/null
        rm -f "${OP_PROTO_CC}"

        # Copy op_proto.h to vendor inc
        cp -f "${OP_PROTO_SRC}" "${CANN_VENDOR}/op_proto/inc/" 2>/dev/null

        # Write version.info
        echo "custom_opp_compiler_version=8.3.0.2.220" > "${CANN_VENDOR}/version.info"

        # Deploy ops-info.json
        OPS_INFO_DIR="${CANN_VENDOR}/op_impl/ai_core/tbe/config/${SOC_VERSION}"
        mkdir -p "${OPS_INFO_DIR}"
        cat > "${OPS_INFO_DIR}/aic-${SOC_VERSION}-ops-info.json" <<'OPSJSON'
{
    "MaskFilter": {
        "coreType": {
            "value": "AiCpu"
        },
        "input0": {
            "dtype": "float32",
            "format": "ND",
            "name": "scores",
            "paramType": "required",
            "shape": "all"
        },
        "input1": {
            "dtype": "uint64",
            "format": "ND",
            "name": "masks",
            "paramType": "required",
            "shape": "all"
        },
        "input2": {
            "dtype": "float32",
            "format": "ND",
            "name": "output_buf",
            "paramType": "required",
            "shape": "all"
        },
        "input3": {
            "dtype": "int32",
            "format": "ND",
            "name": "index_buf",
            "paramType": "required",
            "shape": "all"
        },
        "input4": {
            "dtype": "int32",
            "format": "ND",
            "name": "count_buf",
            "paramType": "required",
            "shape": "all"
        },
        "input5": {
            "dtype": "uint64",
            "format": "ND",
            "name": "task_buf",
            "paramType": "required",
            "shape": "all"
        },
        "output0": {
            "dtype": "float32",
            "format": "ND",
            "name": "result",
            "paramType": "required",
            "shape": "all"
        }
    }
}
OPSJSON
        chmod 644 "${OPS_INFO_DIR}/aic-${SOC_VERSION}-ops-info.json"
        echo ">>> ops-info.json deployed to ${OPS_INFO_DIR}/"

        # Verify deployment
        echo ">>> Vendor deployment verification:"
        echo "    config.ini: $(cat "${VENDORS_INI}" 2>/dev/null)"
        echo "    kernel.so: $(ls -la "${CANN_VENDOR}/op_impl/cpu/aicpu_kernel/impl/libcust_aicpu_kernels_sks_final.so" 2>/dev/null || echo MISSING)"
        echo "    op_proto.so: $(ls -la "${CANN_VENDOR}/op_proto/libcust_op_proto.so" 2>/dev/null || echo MISSING)"
        echo "    ops-info.json: $(ls -la "${CANN_VENDOR}/op_impl/ai_core/tbe/config/${SOC_VERSION}/aic-${SOC_VERSION}-ops-info.json" 2>/dev/null || echo MISSING)"
        echo ">>> Vendor files installed to ${CANN_VENDOR}"
    fi
fi

if [[ ! -f "${OUT_DIR}/bin/${EXECUTABLE_NAME}" ]]; then
    if [[ "${COMPILE_MODE}" == "0" ]]; then
        echo "[ERROR] Executable not found: ${OUT_DIR}/bin/${EXECUTABLE_NAME}"
        echo "[ERROR] Build the project first, or rerun with -c 1."
    else
        echo "[ERROR] Build failed, executable ${EXECUTABLE_NAME} not found."
    fi
    exit 1
fi

# --- Pre-flight checks ---
preflight_ok=true

# Check CANN toolkit
if [[ ! -d "${ASCEND_INSTALL_PATH}" ]]; then
    echo "[PREFLIGHT] ERROR: CANN toolkit not found at ${ASCEND_INSTALL_PATH}"
    preflight_ok=false
fi

# Check NPU devices
if ! npu-smi info >/dev/null 2>&1; then
    echo "[PREFLIGHT] WARN: npu-smi info failed — no NPU devices visible"
fi

# Check vendor deployment exists (needed for CANN runtime)
CANN_VENDOR_CHECK="${ASCEND_INSTALL_PATH}/opp/vendors/aicpu_mask"
if [[ ! -d "${CANN_VENDOR_CHECK}" ]]; then
    echo "[PREFLIGHT] WARN: Vendor directory not found at ${CANN_VENDOR_CHECK}"
    echo "[PREFLIGHT]        Run with -c 1 to deploy vendor files."
fi

# Check config.txt exists
if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "[PREFLIGHT] ERROR: config.txt not found at ${CONFIG_FILE}"
    preflight_ok=false
fi

# Check shared libraries
if ! ldd "${OUT_DIR}/bin/${EXECUTABLE_NAME}" 2>/dev/null | grep -q "not found"; then
    : # OK
else
    missing_libs=$(ldd "${OUT_DIR}/bin/${EXECUTABLE_NAME}" 2>/dev/null | grep "not found")
    echo "[PREFLIGHT] ERROR: Missing shared libraries:"
    echo "${missing_libs}"
    preflight_ok=false
fi

if [[ "${preflight_ok}" != "true" ]]; then
    echo "[PREFLIGHT] Pre-flight checks failed. Aborting."
    exit 1
fi
echo "[PREFLIGHT] Checks passed."

# CANN environment diagnostics
echo "[CANN] ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-<not set>}"
echo "[CANN] ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-<not set>}"
echo "[CANN] ASCEND_AICPU_PATH=${ASCEND_AICPU_PATH:-<not set>}"
echo "[CANN] ASCEND_OPP_BUILT_IN=${ASCEND_OPP_BUILT_IN:-<not set>}"
echo "[CANN] LD_LIBRARY_PATH (first 5):"
printf '    %s\n' $(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | head -5)

RESULT_ROOT_RAW="$(extract_config_value "query_result_root" "${CONFIG_FILE}")"
RESULT_ROOT_PATH="$(resolve_result_root "${RESULT_ROOT_RAW}")"
if [[ -z "${RESULT_ROOT_PATH}" || "${RESULT_ROOT_PATH}" == "/" || "${RESULT_ROOT_PATH}" == "${OUT_DIR}/bin/." ]]; then
    echo "[ERROR] Refusing to clean unsafe result directory: ${RESULT_ROOT_PATH}"
    exit 1
fi

echo ">>> Cleaning result directory: ${RESULT_ROOT_PATH}"
rm -rf "${RESULT_ROOT_PATH}"
mkdir -p "${RESULT_ROOT_PATH}"

echo ">>> Executing ${EXECUTABLE_NAME}..."
echo "--------------------------------------------------"

if [[ "${PROFILE_MODE}" == "1" ]]; then
    run_profile_mode "${EXECUTABLE_NAME}" "${RESULT_ROOT_PATH}" "${FLAMEGRAPH_DIR}" "${PROFILE_DIR}"
else
    run_normal_mode "${EXECUTABLE_NAME}" "${RESULT_ROOT_PATH}"
fi

echo "--------------------------------------------------"
