#!/usr/bin/env bash
# ##############################################################################
# check_tier1_isolation.sh — Tier 2 Isolation Static Verification
#
# Scans all SPAN source and header files for #include directives that reference
# forbidden HELM component path segments (halo/, logs/, axis/, amio/, dagr/,
# conf/, tick/, blend/), case-insensitive.
#
# SPAN is a Tier 2 library and must not depend on peer or lower-tier HELM
# components. The C++20 standard header <span> (bare, without a trailing '/')
# is explicitly permitted. Only includes whose path contains a forbidden
# directory segment followed by '/' are rejected.
#
# Usage:
#   check_tier1_isolation.sh [SPAN_ROOT]
#
# Arguments:
#   SPAN_ROOT  Path to the SPAN library root (defaults to parent of cmake/).
#
# Exit codes:
#   0  All files pass — no forbidden includes found.
#   1  One or more forbidden includes detected.
#
# Requirements: 9.1, 9.3, 9.7, 9.8
# ##############################################################################

set -uo pipefail

# Determine SPAN root directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPAN_ROOT="${1:-"${SCRIPT_DIR}/.."}"

# Resolve to absolute path.
SPAN_ROOT="$(cd "${SPAN_ROOT}" && pwd)"

# Directories to scan.
INCLUDE_DIR="${SPAN_ROOT}/include"
SRC_DIR="${SPAN_ROOT}/src"

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
    echo "TIER2_ISOLATION: No source/header files found to scan in ${SPAN_ROOT}"
    echo "TIER2_ISOLATION: PASSED (no files)"
    exit 0
fi

# Forbidden HELM component path segments (case-insensitive).
# Pattern matches #include directives containing these path segments followed by '/'.
# E.g., halo/, logs/, axis/, amio/, dagr/, conf/, tick/, blend/
# Note: bare <span> (without '/') is NOT forbidden — it's the C++20 standard header.
# SPAN does not forbid span/ (its own library).
FORBIDDEN_PATTERN='#[[:space:]]*include[[:space:]]*[<"][^>"]*\b(halo|logs|axis|amio|dagr|conf|tick|blend)/[^>"]*[>"]'

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
    echo "TIER2_ISOLATION: FAILED — ${VIOLATION_COUNT} forbidden #include directive(s) detected."
    echo ""
    echo "Offending files and lines:"
    echo "─────────────────────────────────────────────────────────────────"
    for violation in "${VIOLATIONS[@]}"; do
        echo "  ${violation}"
    done
    echo "─────────────────────────────────────────────────────────────────"
    echo ""
    echo "SPAN is a Tier 2 library and must NOT include headers from other"
    echo "HELM components (HALO, LOGS, AXIS, AMIO, DAGR, CONF, TICK, BLEND)."
    echo "The C++20 standard <span> header is permitted (no trailing '/')."
    exit 1
else
    echo "TIER2_ISOLATION: PASSED — scanned ${#FILES[@]} file(s), no forbidden includes found."
    exit 0
fi
