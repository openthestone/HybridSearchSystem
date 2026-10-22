#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/format.sh          Format staged C/C++ files only
  scripts/format.sh --changed
                            Format changed C/C++ files
  scripts/format.sh --all    Format all tracked project C/C++ files
  scripts/format.sh --check  Check staged C/C++ files without rewriting

You can also pass explicit files:
  scripts/format.sh old/src/serial.cpp src/engine/index/data_table.h
EOF
}

find_clang_format() {
    if [[ -n "${CLANG_FORMAT:-}" ]]; then
        if [[ -x "${CLANG_FORMAT}" ]]; then
            printf '%s\n' "${CLANG_FORMAT}"
            return 0
        fi
        echo "CLANG_FORMAT is set but not executable: ${CLANG_FORMAT}" >&2
        return 1
    fi

    local candidates=(
        clang-format
        clang-format-19
        clang-format-18
        clang-format-17
        clang-format-16
        clang-format-15
        clang-format-14
        /opt/homebrew/opt/llvm/bin/clang-format
        /usr/local/opt/llvm/bin/clang-format
    )

    local candidate
    for candidate in "${candidates[@]}"; do
        if command -v "${candidate}" >/dev/null 2>&1; then
            command -v "${candidate}"
            return 0
        fi
        if [[ -x "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    cat >&2 <<'EOF'
clang-format was not found.

Install it on macOS with:
  brew install clang-format

Or point the script at an existing binary:
  CLANG_FORMAT=/path/to/clang-format scripts/format.sh
EOF
    return 1
}

is_cpp_file() {
    case "$1" in
        *.c|*.cc|*.cpp|*.cxx|*.h|*.hh|*.hpp|*.hxx) return 0 ;;
        *) return 1 ;;
    esac
}

is_excluded_file() {
    case "$1" in
        old/include/nlohmann/*) return 0 ;;
        old/include/Clustering/SuperKMeans/*) return 0 ;;
        old/include/NPU/vendor/*) return 0 ;;
        *) return 1 ;;
    esac
}

add_file() {
    local file="$1"
    [[ -f "${file}" ]] || return 0
    is_cpp_file "${file}" || return 0
    is_excluded_file "${file}" && return 0
    files+=("${file}")
}

collect_changed_files() {
    local file

    while IFS= read -r file; do
        add_file "${file}"
    done < <(git diff --name-only --diff-filter=ACMRTUXB)

    while IFS= read -r file; do
        add_file "${file}"
    done < <(git diff --name-only --cached --diff-filter=ACMRTUXB)

    while IFS= read -r file; do
        add_file "${file}"
    done < <(git ls-files --others --exclude-standard)
}

collect_staged_files() {
    local file
    while IFS= read -r file; do
        add_file "${file}"
    done < <(git diff --name-only --cached --diff-filter=ACMRTUXB)
}

collect_all_files() {
    local file
    while IFS= read -r file; do
        add_file "${file}"
    done < <(git ls-files \
        '*.c' '*.cc' '*.cpp' '*.cxx' \
        '*.h' '*.hh' '*.hpp' '*.hxx')
}

files=()
mode="write"

case "${1:-}" in
    -h|--help)
        usage
        exit 0
        ;;
    --changed)
        shift
        collect_changed_files
        ;;
    --all)
        shift
        collect_all_files
        ;;
    --check)
        shift
        mode="check"
        collect_staged_files
        ;;
    "")
        collect_staged_files
        ;;
    *)
        for file in "$@"; do
            add_file "${file}"
        done
        ;;
esac

if [[ "${#files[@]}" -eq 0 ]]; then
    echo "No C/C++ files found."
    exit 0
fi

formatter="$(find_clang_format)"

echo "Using formatter: ${formatter}"
echo "Formatting ${#files[@]} file(s)."

index=0
for file in "${files[@]}"; do
    index=$((index + 1))
    printf '[%d/%d] %s\n' "${index}" "${#files[@]}" "${file}"
    if [[ "${mode}" == "check" ]]; then
        "${formatter}" --dry-run --Werror "${file}"
    else
        "${formatter}" -i "${file}"
    fi
done
