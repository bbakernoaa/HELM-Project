#!/bin/sh
# ─────────────────────────────────────────────────────────────────────────────
# check_zero_computation.sh — Static zero-computation enforcement for DAGR.
#
# DAGR is the Tier 3 orchestrator and performs ZERO mathematical calculations.
# It routes pointers between lower-tier engines (TICK, AMIO, BLEND, HALO) and
# manages execution dependencies. This script scans DAGR's shipping source and
# header files for violations of the zero-computation constraint and tier
# isolation rules.
#
# Checks:
#   1. Prohibited math includes: <cmath>, <numeric>, <complex>, <valarray>,
#      <random>, <numbers> (C++ Numerics library headers)
#   2. Raw MPI function calls: any direct MPI_* call (all MPI is mediated
#      through HALO's Communicator interface)
#   3. Direct stdout/stderr writes: std::cout, std::cerr, printf, fprintf
#      (all output goes through HELM::LOGS)
#
#   Requirements: 7.1, 7.6, 7.7, 9.6, 10.6, 11.6
#
# Usage:
#   sh check_zero_computation.sh [ROOT_DIR]
#
#   ROOT_DIR  Directory to scan. Defaults to the script's parent directory
#             (i.e., libs/dagr). Scans include/ and src/ subdirectories,
#             excluding test files.
#
# Exit status:
#   0  No violations found.
#   1  One or more violations found (each printed as file:line:text).
#   2  Usage / environment error.
# ─────────────────────────────────────────────────────────────────────────────

set -u

# Default ROOT to the dagr library directory (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${1:-$(dirname "$SCRIPT_DIR")}"

if [ ! -d "$ROOT" ]; then
    echo "check_zero_computation: error: '$ROOT' is not a directory" >&2
    exit 2
fi

# Build scan paths: include/ and src/ only (not tests/)
SCAN_ROOTS=""
for d in include src; do
    if [ -d "$ROOT/$d" ]; then
        SCAN_ROOTS="$SCAN_ROOTS $ROOT/$d"
    fi
done

if [ -z "$SCAN_ROOTS" ]; then
    echo "check_zero_computation: error: no include/ or src/ directories found under '$ROOT'" >&2
    exit 2
fi

violations=""
violation_count=0

# ─── Check 1: Prohibited math includes ────────────────────────────────────────
# Headers from the C++ Numerics library category.
PROHIBITED_INCLUDES='<cmath>|<numeric>|<complex>|<valarray>|<random>|<numbers>'
INCLUDE_PATTERN="^[[:space:]]*#[[:space:]]*include[[:space:]]*(${PROHIBITED_INCLUDES})"

# shellcheck disable=SC2086  # intentional word-splitting of SCAN_ROOTS
include_hits=$(find $SCAN_ROOTS \
    \( -path '*/build/*' -o -path '*/_deps/*' -o -path '*/CMakeFiles/*' -o -path '*/.git/*' -o -path '*/tests/*' \) -prune -o \
    -type f \( -name '*.hpp' -o -name '*.cpp' -o -name '*.h' -o -name '*.cc' \) \
    -exec grep -nEH "$INCLUDE_PATTERN" {} + 2>/dev/null)

if [ -n "$include_hits" ]; then
    violations="${violations}
── Prohibited math includes (Requirement 7.1) ──
${include_hits}"
    hit_count=$(printf '%s\n' "$include_hits" | grep -c .)
    violation_count=$((violation_count + hit_count))
fi

# ─── Check 2: Raw MPI function calls ──────────────────────────────────────────
# Any token matching MPI_[A-Z] indicates a direct MPI call. All MPI operations
# must be mediated through HALO's Communicator interface (Requirement 9.6).
MPI_PATTERN='MPI_[A-Z][A-Za-z_]+'

# shellcheck disable=SC2086
mpi_hits=$(find $SCAN_ROOTS \
    \( -path '*/build/*' -o -path '*/_deps/*' -o -path '*/CMakeFiles/*' -o -path '*/.git/*' -o -path '*/tests/*' \) -prune -o \
    -type f \( -name '*.hpp' -o -name '*.cpp' -o -name '*.h' -o -name '*.cc' \) \
    -exec grep -nEH "$MPI_PATTERN" {} + 2>/dev/null)

if [ -n "$mpi_hits" ]; then
    violations="${violations}
── Raw MPI function calls (Requirement 9.6) ──
${mpi_hits}"
    hit_count=$(printf '%s\n' "$mpi_hits" | grep -c .)
    violation_count=$((violation_count + hit_count))
fi

# ─── Check 3: Direct stdout/stderr writes ─────────────────────────────────────
# All diagnostic output must go through HELM::LOGS (Requirement 10.6).
STDOUT_PATTERN='std::cout|std::cerr|[^_]printf[[:space:]]*\(|fprintf[[:space:]]*\('

# shellcheck disable=SC2086
stdout_hits=$(find $SCAN_ROOTS \
    \( -path '*/build/*' -o -path '*/_deps/*' -o -path '*/CMakeFiles/*' -o -path '*/.git/*' -o -path '*/tests/*' \) -prune -o \
    -type f \( -name '*.hpp' -o -name '*.cpp' -o -name '*.h' -o -name '*.cc' \) \
    -exec grep -nEH "$STDOUT_PATTERN" {} + 2>/dev/null)

if [ -n "$stdout_hits" ]; then
    violations="${violations}
── Direct stdout/stderr writes (Requirement 10.6) ──
${stdout_hits}"
    hit_count=$(printf '%s\n' "$stdout_hits" | grep -c .)
    violation_count=$((violation_count + hit_count))
fi

# ─── Report results ───────────────────────────────────────────────────────────

if [ $violation_count -gt 0 ]; then
    echo "check_zero_computation: FAIL — $violation_count violation(s) found:" >&2
    printf '%s\n' "$violations" >&2
    echo "" >&2
    echo "DAGR is the Tier 3 orchestrator and performs ZERO computation." >&2
    echo "All math is delegated to BLEND/AXIS, all MPI to HALO, all output to LOGS." >&2
    echo "(Requirements 7.1, 7.6, 7.7, 9.6, 10.6)" >&2
    exit 1
fi

echo "check_zero_computation: PASS — no zero-computation violations found under:$SCAN_ROOTS"
exit 0
