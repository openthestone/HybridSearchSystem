#!/bin/bash
set -euo pipefail

# --- 核心配置 ---
RUN_MODE="npu"
SOC_VERSION="Ascend910B4"
ASCEND_INSTALL_PATH="/usr/local/Ascend/ascend-toolkit/latest"
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
Usage: ./run.sh -v [s|p] -p [0|1] -c [0|1]

Options:
  -v, --variant <s|p>   executable variant: s=serial, p=parallel
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
    local required_count=1
    local available_ids=()

    if [[ "${VARIANT}" == "p" ]]; then
        required_count=8
    fi

    mapfile -t available_ids < <(detect_available_npu_ids)

    if [[ "${#available_ids[@]}" -eq 0 ]]; then
        echo "[ERROR] No available NPU IDs detected by npu-smi."
        exit 1
    fi

    if [[ "${required_count}" -eq 1 ]]; then
        NPU_IDS=("${available_ids[0]}")
    else
        if [[ "${#available_ids[@]}" -lt "${required_count}" ]]; then
            echo "[WARN] Only ${#available_ids[@]} NPU(s) detected, less than expected ${required_count} for parallel mode."
            NPU_IDS=("${available_ids[@]}")
        else
            NPU_IDS=("${available_ids[@]:0:${required_count}}")
        fi
    fi
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

    if [[ "${app_exit_code}" -ne 0 ]]; then
        exit "${app_exit_code}"
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

if [[ "${VARIANT}" != "s" && "${VARIANT}" != "p" ]]; then
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
else
    EXECUTABLE_NAME="parallel"
fi

select_npu_ids_for_variant

if [[ "${PROFILE_MODE}" == "1" ]]; then
    BUILD_TYPE="RelWithDebInfo"
    ENABLE_PROFILING_BUILD="ON"
else
    BUILD_TYPE="Release"
    ENABLE_PROFILING_BUILD="OFF"
fi

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE:-$0}")" && pwd)"
BUILD_DIR="${CURRENT_DIR}/build"
OUT_DIR="${CURRENT_DIR}/out"
CONFIG_FILE="${CURRENT_DIR}/config.txt"
PROFILE_DIR="${CURRENT_DIR}/profile"
FLAMEGRAPH_DIR="${FLAMEGRAPH_DIR:-${CURRENT_DIR}/FlameGraph}"

echo "========================================"
echo "Target: ${SOC_VERSION}"
echo "Mode:   ${RUN_MODE}"
echo "Build:  ${BUILD_TYPE}"
echo "FlameGraph mode: ${PROFILE_MODE}"
echo "Compile mode: ${COMPILE_MODE}"
echo "Executable: ${EXECUTABLE_NAME}"
echo "NPU IDs: ${NPU_IDS[*]}"
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

export LD_LIBRARY_PATH="${OUT_DIR}/lib:/usr/local/lib:/usr/local/lib64:/usr/lib64:${LD_LIBRARY_PATH:-}"

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
else
    echo ">>> Skip compile due to -c 0. Reuse existing out/ artifacts."
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
