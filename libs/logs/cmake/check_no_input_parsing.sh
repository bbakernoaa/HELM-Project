#!/usr/bin/env bash
# ##############################################################################
# check_no_input_parsing.sh — No-Input-Parsing Static Verification
#
# LOGS is strictly an output and status mechanism. It contains NO input-parsing
# logic of any kind: no YAML, no namelist, no configuration-file reading, and
# it never opens a path for reading. This script enforces that invariant by
# scanning all LOGS source and header files for input-parsing indicators.
#
# Input-parsing indicators (closed set per Requirement 10.5):
#   1. YAML / markup-parsing library references:
#      yaml-cpp, yaml.h, ryml, rapidyaml, json.hpp, nlohmann,
#      toml, ini, pugixml, tinyxml, expat
#   2. Namelist / config-format readers:
#      namelist, config_file, config_reader
#   3. File-open-for-read or read-mode file APIs:
#      ifstream, fopen with "r" mode, std::filesystem::*read,
#      open with O_RDONLY
#      Note: ofstream (write-only) is explicitly permitted.
#
# Usage:
#   check_no_input_parsing.sh [LOGS_ROOT]
#
# Arguments:
#   LOGS_ROOT  Path to the LOGS library root (defaults to parent of cmake/).
#
# Exit codes:
#   0  All files pass — no input-parsing indicators found.
#   1  One or more input-parsing indicators detected.
#
# Requirements: 10.5, 10.6, 10.7, 10.8
# ##############################################################################

set -euo pipefail

# Determine LOGS root directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS_ROOT="${1:-"${SCRIPT_DIR}/.."}"

# Resolve to absolute path.
LOGS_ROOT="$(cd "${LOGS_ROOT}" && pwd)"

# Directories to scan.
INCLUDE_DIR="${LOGS_ROOT}/include"
SRC_DIR="${LOGS_ROOT}/src"

# Collect all source and header files (.hpp, .cpp, .h).
FILES=()
for dir in "${INCLUDE_DIR}" "${SRC_DIR}"; do
    if [[ -d "${dir}" ]]; then
        while IFS= read -r -d '' file; do
            FILES+=("${file}")
        done < <(find "${dir}" -type f \( -name "*.hpp" -o -name "*.cpp" -o -name "*.h" \) -print0)
    fi
done

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "NO_INPUT_PARSING: No source/header files found to scan in ${LOGS_ROOT}"
    echo "NO_INPUT_PARSING: PASSED (no files)"
    exit 0
fi

# ─── Pattern Definitions ─────────────────────────────────────────────────────

# 1. YAML / markup-parsing library includes
YAML_LIBS_PATTERN='#[[:space:]]*include[[:space:]]*[<"][^>"]*\b(yaml-cpp|yaml\.h|ryml|rapidyaml|json\.hpp|nlohmann|toml|pugixml|tinyxml|expat)\b'

# 2. INI library includes (separate to avoid false positives — matches include directives only)
INI_INCLUDE_PATTERN='#[[:space:]]*include[[:space:]]*[<"][^>"]*\bini\b[^>"]*[>"]'

# 3. Namelist / config-format readers (identifiers in source code)
NAMELIST_PATTERN='\b(namelist|config_file|config_reader)\b'

# 4. File-open-for-read APIs
#    - ifstream (read-capable stream; ofstream is OK)
#    - fopen with "r" mode
#    - std::filesystem::.*read
#    - open with O_RDONLY
IFSTREAM_PATTERN='\bifstream\b'
FOPEN_READ_PATTERN='\bfopen\s*\([^)]*"r'
FILESYSTEM_READ_PATTERN='std::filesystem::[[:alnum:]_]*read'
OPEN_RDONLY_PATTERN='\bopen\s*\([^)]*O_RDONLY'

VIOLATIONS=()
VIOLATION_COUNT=0

# Helper: check a file against a pattern with a description.
check_pattern() {
    local file="$1"
    local pattern="$2"
    local description="$3"

    while IFS= read -r match; do
        VIOLATIONS+=("${file}:${match}  [${description}]")
        ((VIOLATION_COUNT++))
    done < <(grep -nEi "${pattern}" "${file}" 2>/dev/null || true)
}

for file in "${FILES[@]}"; do
    # 1. YAML / markup-parsing library references
    check_pattern "${file}" "${YAML_LIBS_PATTERN}" "YAML/markup-parsing library"

    # 2. INI library include
    check_pattern "${file}" "${INI_INCLUDE_PATTERN}" "INI config library"

    # 3. Namelist / config-format readers
    check_pattern "${file}" "${NAMELIST_PATTERN}" "namelist/config-format reader"

    # 4. File-open-for-read APIs
    check_pattern "${file}" "${IFSTREAM_PATTERN}" "ifstream (read-mode file API)"
    check_pattern "${file}" "${FOPEN_READ_PATTERN}" "fopen read-mode"
    check_pattern "${file}" "${FILESYSTEM_READ_PATTERN}" "std::filesystem read API"
    check_pattern "${file}" "${OPEN_RDONLY_PATTERN}" "open O_RDONLY"
done

if [[ ${VIOLATION_COUNT} -gt 0 ]]; then
    echo "NO_INPUT_PARSING: FAILED — ${VIOLATION_COUNT} input-parsing indicator(s) detected."
    echo ""
    echo "Offending files and lines:"
    echo "─────────────────────────────────────────────────────────────────"
    for violation in "${VIOLATIONS[@]}"; do
        echo "  ${violation}"
    done
    echo "─────────────────────────────────────────────────────────────────"
    echo ""
    echo "LOGS is strictly an output and status mechanism. It must NOT"
    echo "contain any input-parsing logic: no YAML, no namelist, no"
    echo "configuration-file reading, and it must never open a path"
    echo "for reading. (Requirements 10.5, 10.6, 10.7, 10.8)"
    exit 1
else
    echo "NO_INPUT_PARSING: PASSED — scanned ${#FILES[@]} file(s), no input-parsing indicators found."
    exit 0
fi
