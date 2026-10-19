#!/usr/bin/env bash
# install_deps.sh -- what fr_converter / fr_builder / fr_search need, and getting it in place.
#
#   ./scripts/install_deps.sh                 check only; prints what is missing and what to run
#   ./scripts/install_deps.sh --install       install the missing pieces (uses sudo if not root)
#   ./scripts/install_deps.sh --install --local
#                                             no root: stage into <repo>/third_party, which CMake
#                                             and run.sh already search, so nothing to configure
#   ./scripts/install_deps.sh --bundle DIR    pack this machine's libs into DIR, to carry to a box
#                                             with no network (see docs/deps-bundle.md)
#   ./scripts/install_deps.sh --from DIR      install from such a bundle
#
# The CANN toolkit is NOT installed here -- it ships with the NPU driver. The check reports it.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TP="${ROOT_DIR}/third_party"
DO_INSTALL=0; LOCAL_ONLY=0; BUNDLE_DIR=""; FROM_DIR=""
SCRATCH="$(mktemp -d)"; trap 'rm -rf "${SCRATCH}"' EXIT
MISSING=""      # space-separated component keys the check could not satisfy
BLOCKED=""      # ... of those, the ones this script cannot fix

RAPIDJSON_URL="https://github.com/Tencent/rapidjson/archive/refs/tags/v1.1.0.tar.gz"
GFLAGS_URL="https://github.com/gflags/gflags/archive/refs/tags/v2.2.2.tar.gz"

# Callers use it as tmp="$(scratch)", which is a SUBSHELL -- so the dir and its cleanup trap
# have to be owned by the script, not created in here, or the trap fires as the subshell exits.
scratch() { mktemp -d "${SCRATCH}/XXXXXX"; }
say()  { printf '%s\n' "$*"; }
ok()   { printf '  \033[32mok\033[0m    %-14s %s\n' "$1" "$2"; }
warn() { printf '  \033[33mwarn\033[0m  %-14s %s\n' "$1" "$2"; }
bad()  { printf '  \033[31mMISS\033[0m  %-14s %s\n' "$1" "$2"; MISSING="${MISSING} $3"; }
die()  { printf '[ERROR] %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

usage() { sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install) DO_INSTALL=1; shift ;;
        --local)   LOCAL_ONLY=1; shift ;;
        --bundle)  [[ $# -ge 2 ]] || die "--bundle needs a directory"; BUNDLE_DIR="$2"; shift 2 ;;
        --from)    [[ $# -ge 2 ]] || die "--from needs a directory";   FROM_DIR="$2";   shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1 (--help for usage)" ;;
    esac
done

# ---- environment -------------------------------------------------------------
SUDO=""
if [[ "$(id -u)" != "0" ]]; then
    have sudo && SUDO="sudo"
fi
PKG=""
for m in dnf yum apt-get zypper; do have "$m" && { PKG="$m"; break; }; done

# openEuler/CentOS, Ubuntu/Debian and SLES name these differently.
pkg_for() {  # <MISSING key> -> package names, empty if this key is not a package
    case "${PKG}" in
        dnf|yum) case "$1" in
            cmake) echo cmake ;;      cxx) echo gcc-c++ ;;
            protobuf) echo "protobuf-devel protobuf-compiler" ;;
            gflags) echo gflags-devel ;;  rapidjson) echo rapidjson-devel ;;
            python3) echo python3 ;;  coreutils) echo "coreutils gawk sed grep tar" ;;
        esac ;;
        apt-get) case "$1" in
            cmake) echo cmake ;;      cxx) echo g++ ;;
            protobuf) echo "libprotobuf-dev protobuf-compiler" ;;
            gflags) echo libgflags-dev ;; rapidjson) echo rapidjson-dev ;;
            python3) echo python3 ;;  coreutils) echo "coreutils gawk sed grep tar" ;;
        esac ;;
        zypper) case "$1" in
            cmake) echo cmake ;;      cxx) echo gcc-c++ ;;
            protobuf) echo protobuf-devel ;;
            gflags) echo libgflags-devel ;; rapidjson) echo rapidjson-devel ;;
            python3) echo python3 ;;  coreutils) echo "coreutils gawk sed grep tar" ;;
        esac ;;
    esac
}

# Only what the check reported missing. Handing over the whole list instead means
# `dnf install python3` on a box that already has it, which is an upgrade, not a fix.
pkg_list() {
    local key out=""
    for key in ${MISSING}; do out="${out} $(pkg_for "${key}")"; done
    echo "${out}" | tr -s ' ' | sed 's/^ //; s/ $//'
}

# Kept out of pkg_list on purpose: dnf and apt refuse the WHOLE transaction over one name they
# cannot match, and openEuler 24.03 carries no dtach at all -- so a nice-to-have in that list
# means protobuf does not get installed either. These are installed one at a time, failure ok.
pkg_optional() { echo "dtach"; }

# ---- checks ------------------------------------------------------------------
# Each prints one row; a MISS records a key that --install knows how to act on.
CANN="${CANN:-}"
check_cann() {
    # Same precedence as run.sh:43, including the bare CANN= override.
    CANN="${CANN:-${ASCEND_CANN_PACKAGE_PATH:-${ASCEND_INSTALL_PATH:-}}}"
    [[ -n "${CANN}" ]] || for c in "${HOME}/Ascend/ascend-toolkit/latest" /usr/local/Ascend/ascend-toolkit/latest; do
        [[ -d "${c}" ]] && { CANN="${c}"; break; }
    done
    if [[ -z "${CANN}" || ! -d "${CANN}" ]]; then
        printf '  \033[31mMISS\033[0m  %-14s %s\n' "CANN" "not found -- ships with the NPU driver, cannot be installed here"
        BLOCKED="${BLOCKED} CANN"
        return 0
    fi
    if [[ -d "${CANN}/compiler/tikcpp/ascendc_kernel_cmake" || -d "${CANN}/tools/tikcpp/ascendc_kernel_cmake" ]]; then
        ok "CANN" "${CANN}"
    else
        warn "CANN" "${CANN} has no ascendc_kernel_cmake -- device kernels will not build"
    fi
}

check_cmake() {
    if ! have cmake; then bad "cmake" "not found (need >= 3.16)" "cmake"; return 0; fi
    local v; v="$(cmake --version | head -1 | awk '{print $3}')"
    ok "cmake" "${v}"
}

check_cxx() {
    local cxx="${CXX:-g++}"
    if ! have "${cxx}"; then bad "c++" "no ${cxx} (need a C++17 compiler)" "cxx"; return 0; fi
    ok "c++" "$("${cxx}" --version | head -1)"
}

# Returns the directory holding libprotobuf.so, or empty.
find_lib() {  # <soname-stem> [extra dirs...]
    local stem="$1"; shift
    local d ext
    for d in "${TP}/lib64" "${TP}/lib" "${ROOT_DIR}/lib64" "${ROOT_DIR}/lib" \
             /usr/local/lib64 /usr/local/lib /usr/lib64 /usr/lib /usr/lib/aarch64-linux-gnu "$@"; do
        for ext in so dylib; do   # dylib only so a dev-box check is honest; the servers are Linux
            [[ -e "${d}/${stem}.${ext}" ]] && { echo "${d}"; return 0; }
        done
    done
    return 1
}
find_inc() {  # <header path>
    local d
    for d in "${TP}/include" "${ROOT_DIR}/include" /usr/local/include /usr/include; do
        [[ -f "${d}/$1" ]] && { echo "${d}"; return 0; }
    done
    return 1
}

check_protobuf() {
    local libdir incdir
    libdir="$(find_lib libprotobuf || true)"
    incdir="$(find_inc google/protobuf/message.h || true)"
    if [[ -z "${libdir}" ]]; then
        # the documented trap: a versioned .so with no bare symlink is invisible to FindProtobuf
        local versioned d
        for d in "${TP}/lib64" "${TP}/lib" /usr/local/lib64 /usr/lib64 /usr/lib/aarch64-linux-gnu; do
            versioned="$(ls "${d}"/libprotobuf.so.* 2>/dev/null | head -1 || true)"
            [[ -n "${versioned}" ]] && break
        done
        if [[ -n "${versioned}" ]]; then
            bad "protobuf" "only ${versioned}; FindProtobuf needs a bare libprotobuf.so symlink" "pb-symlink"
            return 0
        fi
        bad "protobuf" "libprotobuf.so not found" "protobuf"; return 0
    fi
    [[ -n "${incdir}" ]] || { bad "protobuf" "lib in ${libdir} but headers missing" "protobuf"; return 0; }
    have protoc || [[ -x "${TP}/bin/protoc" ]] || { bad "protobuf" "protoc not found" "protobuf"; return 0; }
    ok "protobuf" "${libdir} + ${incdir}"
}

check_absl() {
    # protobuf 4.25 links absl; older protobuf does not, so absent is only a warning.
    local d; d="$(find_lib libabsl_status || true)"
    if [[ -n "${d}" ]]; then ok "abseil" "${d}"
    else warn "abseil" "not found -- only needed if protobuf >= 4.22 (link error names libabsl_*)"; fi
}

check_gflags() {
    local libdir incdir
    libdir="$(find_lib libgflags || true)"
    incdir="$(find_inc gflags/gflags.h || true)"
    if [[ -n "${libdir}" && -n "${incdir}" ]]; then ok "gflags" "${libdir}"
    else bad "gflags" "libgflags.so or gflags/gflags.h not found" "gflags"; fi
}

RAPIDJSON_FOUND=""
check_rapidjson() {
    local d
    for d in "${RAPIDJSON_INCLUDE_DIR:-}" "${TP}/rapidjson/include" "${TP}/rapidjson-1.1.0/include" \
             "${ROOT_DIR}/rapidjson/include" "${ROOT_DIR}/rapidjson-1.1.0/include" \
             /opt/rapidjson/include /usr/local/include /usr/include; do
        [[ -n "${d}" && -f "${d}/rapidjson/document.h" ]] && { RAPIDJSON_FOUND="${d}"; break; }
    done
    if [[ -n "${RAPIDJSON_FOUND}" ]]; then ok "rapidjson" "${RAPIDJSON_FOUND}"
    else bad "rapidjson" "rapidjson/document.h not found (fr_builder needs it)" "rapidjson"; fi
}

check_python3() { have python3 && ok "python3" "$(python3 --version 2>&1)" || bad "python3" "not found (run.sh's perf tables and tools/*.py)" "python3"; }

check_tools() {
    local miss="" t
    for t in awk sed grep du tar; do have "$t" || miss="${miss} $t"; done
    [[ -z "${miss}" ]] && ok "coreutils" "awk sed grep du tar" || bad "coreutils" "missing:${miss}" "coreutils"
    have dtach && ok "dtach" "$(command -v dtach)" || warn "dtach" "not found -- experiments are meant to run under it (see AGENTS.md)"
    have curl  && ok "curl"  "$(command -v curl)"  || warn "curl"  "not found -- only needed for --notify"
    have npu-smi && ok "npu-smi" "$(command -v npu-smi)" || warn "npu-smi" "not found -- ships with the driver; banner/sampler degrade without it"
}

run_checks() {
    MISSING=""; BLOCKED=""
    say "== dependency check =================================================="
    check_cann; check_cmake; check_cxx; check_protobuf; check_absl
    check_gflags; check_rapidjson; check_python3; check_tools
    say ""
}

# ---- install actions ---------------------------------------------------------
need() { case " ${MISSING} " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

install_via_pkg() {
    [[ -n "${PKG}" ]] || { say "  no supported package manager (dnf/yum/apt-get/zypper); use --local"; return 1; }
    local list p rc=0
    list="$(pkg_list)"
    [[ -n "${list}" ]] || { say "  nothing for ${PKG} to install"; return 0; }
    say "  ${SUDO:+${SUDO} }${PKG} install ${list}"
    case "${PKG}" in
        apt-get) { ${SUDO} apt-get update && ${SUDO} apt-get install -y ${list}; } || rc=$? ;;
        *)       ${SUDO} "${PKG}" install -y ${list} || rc=$? ;;
    esac
    [[ "${rc}" == "0" ]] || { say "  ${PKG} install failed (rc=${rc})"; return 1; }
    for p in $(pkg_optional); do
        have "${p}" && continue
        ${SUDO} "${PKG}" install -y "${p}" >/dev/null 2>&1 \
            || say "  ${p} is not in this distro's repos; install it by hand (experiments want it)"
    done
    return 0
}

# --from DIR may hold staged directories (what --bundle writes) and/or source tarballs.
# Returns non-zero rather than aborting, so one absent piece still leaves a complete report.
fetch() {  # <url> <dest-tarball>
    local base; base="$(basename "$2")"
    if [[ -n "${FROM_DIR}" ]]; then
        if [[ ! -f "${FROM_DIR}/${base}" ]]; then
            say "  ${base} is not in ${FROM_DIR} -- put it there, or drop --from to download it"
            return 1
        fi
        cp "${FROM_DIR}/${base}" "$2"; return 0
    fi
    have curl || { say "  curl is needed to download ${base}, or pass --from DIR"; return 1; }
    say "  downloading ${base}"
    curl -fsSL "$1" -o "$2"
}

stage_rapidjson() {
    local tmp; tmp="$(scratch)"
    fetch "${RAPIDJSON_URL}" "${tmp}/rapidjson.tar.gz" || return 1
    tar -xzf "${tmp}/rapidjson.tar.gz" -C "${tmp}"
    mkdir -p "${TP}/rapidjson"
    cp -r "${tmp}"/rapidjson-*/include "${TP}/rapidjson/"
    say "  rapidjson headers -> ${TP}/rapidjson/include"
}

stage_gflags() {
    have cmake || { say "  cmake is needed to build gflags"; return 1; }
    local tmp log extra=()
    tmp="$(scratch)"; log="${tmp}/build.log"
    # gflags 2.2.2 still declares cmake_minimum_required(2.8), which CMake >= 4 refuses outright.
    [[ "$(cmake --version | head -1 | awk '{print $3}' | cut -d. -f1)" -ge 4 ]] && \
        extra+=(-DCMAKE_POLICY_VERSION_MINIMUM=3.5)
    fetch "${GFLAGS_URL}" "${tmp}/gflags.tar.gz" || return 1
    tar -xzf "${tmp}/gflags.tar.gz" -C "${tmp}"
    # This runs under `|| true`, where `set -e` is suppressed -- so check every step by hand or a
    # failed build still reports success.
    { cmake -S "${tmp}"/gflags-* -B "${tmp}/b" -DCMAKE_INSTALL_PREFIX="${TP}" \
            -DBUILD_SHARED_LIBS=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON "${extra[@]}" \
      && cmake --build "${tmp}/b" -j "$(nproc 2>/dev/null || echo 4)" \
      && cmake --install "${tmp}/b"; } >"${log}" 2>&1 || {
        say "  gflags build FAILED; last 15 lines:"
        tail -15 "${log}" | sed 's/^/    /'
        return 1
    }
    say "  gflags -> ${TP}"
}

fix_pb_symlink() {
    local d versioned
    for d in "${TP}/lib64" "${TP}/lib" /usr/local/lib64 /usr/lib64 /usr/lib/aarch64-linux-gnu; do
        versioned="$(ls "${d}"/libprotobuf.so.* 2>/dev/null | head -1 || true)"
        [[ -n "${versioned}" ]] || continue
        case "${d}" in
            "${TP}"/*) ln -sf "$(basename "${versioned}")" "${d}/libprotobuf.so" ;;
            *)         ${SUDO} ln -sf "$(basename "${versioned}")" "${d}/libprotobuf.so" ;;
        esac
        say "  linked ${d}/libprotobuf.so -> $(basename "${versioned}")"
        # CMake looks in lib as well as lib64 under a prefix
        [[ -d "${TP}/lib64" && ! -e "${TP}/lib" ]] && ln -sfn lib64 "${TP}/lib" && say "  linked ${TP}/lib -> lib64"
        return 0
    done
    return 1
}

do_install() {
    say "== installing ========================================================"
    mkdir -p "${TP}"
    if need pb-symlink; then fix_pb_symlink || say "  could not find a versioned libprotobuf.so"; fi
    if [[ "${LOCAL_ONLY}" == "1" ]]; then
        need rapidjson && { stage_rapidjson || true; }
        need gflags    && { stage_gflags    || true; }
        if need protobuf; then
            say "  protobuf: building it from source here is a long job and it is the one dependency"
            say "            the bundle path exists for. Take it from a machine that has it:"
            say "              on that machine: ./scripts/install_deps.sh --bundle /tmp/npur-deps"
            say "              here:            ./scripts/install_deps.sh --install --local --from /tmp/npur-deps"
        fi
    else
        if need protobuf || need gflags || need rapidjson || need cmake || need cxx || need python3 || need coreutils; then
            install_via_pkg || say "  falling back: re-run with --local to stage into third_party/"
        fi
    fi
    say ""
}

# ---- bundle ------------------------------------------------------------------
do_bundle() {
    mkdir -p "${BUNDLE_DIR}/lib64" "${BUNDLE_DIR}/include" "${BUNDLE_DIR}/bin"
    local d
    for d in $(find_lib libprotobuf || true) $(find_lib libgflags || true) $(find_lib libabsl_status || true); do
        # -L: copy what the symlink points at. A bundled symlink whose target is absent is the
        # single most common way this bundle arrives broken.
        cp -Lr "${d}"/libprotobuf*.so* "${d}"/libprotoc*.so* "${d}"/libgflags*.so* \
           "${d}"/libabsl_*.so* "${d}"/libutf8_*.so* "${BUNDLE_DIR}/lib64/" 2>/dev/null || true
    done
    for d in $(find_inc google/protobuf/message.h || true) $(find_inc gflags/gflags.h || true); do
        cp -Lr "${d}/google" "${d}/gflags" "${d}/absl" "${BUNDLE_DIR}/include/" 2>/dev/null || true
    done
    have protoc && cp -L "$(command -v protoc)" "${BUNDLE_DIR}/bin/protoc"
    [[ -n "${RAPIDJSON_FOUND}" ]] && { mkdir -p "${BUNDLE_DIR}/rapidjson"; cp -Lr "${RAPIDJSON_FOUND}" "${BUNDLE_DIR}/rapidjson/"; }
    rmdir "${BUNDLE_DIR}"/* 2>/dev/null || true   # an empty lib64/bin would misrepresent the bundle
    if [[ -z "$(ls -A "${BUNDLE_DIR}" 2>/dev/null)" ]]; then
        say "[ERROR] nothing to bundle -- this machine has none of protobuf / gflags / rapidjson" >&2
        exit 1
    fi
    say "bundle written to ${BUNDLE_DIR} ($(du -sh "${BUNDLE_DIR}" 2>/dev/null | cut -f1)):"
    (cd "${BUNDLE_DIR}" && ls -1) | sed 's/^/  /' 
    say "copy it to the offline box, then there:"
    say "  ./scripts/install_deps.sh --install --local --from ${BUNDLE_DIR}"
}

install_from_bundle() {
    [[ -d "${FROM_DIR}" ]] || die "${FROM_DIR} is not a directory"
    mkdir -p "${TP}"
    for d in lib64 include bin rapidjson; do
        [[ -d "${FROM_DIR}/${d}" ]] && { cp -r "${FROM_DIR}/${d}" "${TP}/"; say "  ${d} -> ${TP}/${d}"; }
    done
    [[ -f "${TP}/bin/protoc" ]] && chmod +x "${TP}/bin/protoc"
    [[ -d "${TP}/lib64" && ! -e "${TP}/lib" ]] && ln -sfn lib64 "${TP}/lib"
    fix_pb_symlink || true
}

# ---- main --------------------------------------------------------------------
run_checks

if [[ -n "${BUNDLE_DIR}" ]]; then do_bundle; exit 0; fi

if [[ "${DO_INSTALL}" == "1" ]]; then
    if [[ -n "${FROM_DIR}" ]]; then
        say "== unpacking bundle =================================================="
        install_from_bundle
        say ""
        run_checks   # re-check first: what the bundle supplied must not be looked for again
    fi
    do_install
    run_checks
fi

if [[ -n "${MISSING// /}" ]]; then
    say "still missing:${MISSING}"
    [[ "${DO_INSTALL}" == "1" ]] || say "run with --install (add --local if you have no root)"
    exit 1
fi
if [[ -n "${BLOCKED// /}" ]]; then
    say "everything this script can provide is in place, but:${BLOCKED} is not."
    say "CANN comes with the NPU driver; point ASCEND_CANN_PACKAGE_PATH at it if it is elsewhere."
    exit 1
fi
say "all dependencies satisfied. Next:"
say "  bash src/tests/run_tests.sh      # host-side unit tests, no NPU needed"
say "  ./run.sh -c 1 --convert-query 0 --convert-data 0 --build-index 0 --search 0   # compile only"
