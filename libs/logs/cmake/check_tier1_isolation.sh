#!/usr/bin/env bash
# ##############################################################################
# check_tier1_isolation.sh — Tier 1 Isolation Static Verification
#
# Scans all LOGS source and header files for #include directives that reference
# forbidden HELM component path segments (tick/, halo/, axis/, amio/, span/,
# dagr/), case-insensitive.
#
# The C++20 standard header <span> (bare, without a trailing '/') is explicitly
# permitted. Only includes whose path contains a "span/" directory segment are
# forbidden (e.g., <span/interop.hpp> or "span/bar.hpp").
#
# Usage:
#   check_tier1_isolation.sh [LOGS_ROOT]
#
# Arguments:
#   LOGS_ROOT  Path to the LOGS library root (defaults to parent of cmake/).
#
# Exit codes:
#   0  All files pass — no forbidden includes found.
#   1  One or more forbidden includes detected.
#
# Requirements: 14.1, 14.5, 14.6, 14.7
# ##############################################################################

set -uo pipefail

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
    echo "TIER1_ISOLATION: No source/header files found to scan in ${LOGS_ROOT}"
    echo "TIER1_ISOLATION: PASSED (no files)"
    exit 0
fi

# Forbidden HELM component path segments (case-insensitive).
# Pattern matches #include directives containing these path segments followed by '/'.
# E.g., tick/, halo/, axis/, amio/, span/, dagr/
# Note: bare <span> (without '/') is NOT forbidden — it's the C++20 standard header.
FORBIDDEN_PATTERN='#[[:space:]]*include[[:space:]]*[<"][^>"]*\b(tick|halo|axis|amio|span|dagr)/[^>"]*[>"]'

VIOLATIONS=()
VIOLATION_COUNT=0

for file in "${FILES[@]}"; do
    # Use grep with extended regex, case-insensitive, to find offending lines.
    while IFS= read -r match; do
        VIOLATIONS+=("${file}:${match}")
        ((VIOLATION_COUNT++))
    done < <(grep -inE "${FORBIDDEN_PATTERN}" "${file}" 2>/dev/null || true)
done

if [[ ${VIOLATION_COUNT} -gt 0 ]]; then
    echo "TIER1_ISOLATION: FAILED — ${VIOLATION_COUNT} forbidden #include directive(s) detected."
    echo ""
    echo "Offending files and lines:"
    echo "─────────────────────────────────────────────────────────────────"
    for violation in "${VIOLATIONS[@]}"; do
        echo "  ${violation}"
    done
    echo "─────────────────────────────────────────────────────────────────"
    echo ""
    echo "LOGS is a Tier 1 library and must NOT include headers from other"
    echo "HELM components (TICK, HALO, AXIS, AMIO, SPAN, DAGR)."
    echo "The C++20 standard <span> header is permitted (no trailing '/')."
    exit 1
else
    echo "TIER1_ISOLATION: PASSED — scanned ${#FILES[@]} file(s), no forbidden includes found."
    exit 0
fi
