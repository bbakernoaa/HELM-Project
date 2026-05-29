#!/bin/sh
# ─────────────────────────────────────────────────────────────────────────────
# check_tier1_isolation.sh — Static Tier 1 isolation guard for HALO.
#
# HALO is a Tier 1 HELM micro-library and MUST have ZERO compile-time
# dependencies on any other HELM component (TICK, LOGS, AXIS, AMIO, SPAN, DAGR).
# This script scans HALO's shipping source and header files for `#include`
# directives (C/C++) or `use` statements (Fortran) that reference one of those
# forbidden components and fails the build if any are found.
#
#   Requirement 13: Tier 1 Isolation Compliance (13.1–13.5)
#
# Usage:
#   sh check_tier1_isolation.sh [ROOT_DIR]
#
#   ROOT_DIR  Directory to scan. Defaults to the current directory. The script
#             prefers the shipping trees (include/, src/, fortran/) when they
#             exist under ROOT_DIR and otherwise scans ROOT_DIR directly (this
#             keeps the script usable against a scratch directory for testing).
#
# Exit status:
#   0  No forbidden HELM component references found.
#   1  One or more forbidden references found (each printed as file:line:text).
#   2  Usage / environment error.
# ─────────────────────────────────────────────────────────────────────────────

set -u

ROOT="${1:-.}"

if [ ! -d "$ROOT" ]; then
    echo "check_tier1_isolation: error: '$ROOT' is not a directory" >&2
    exit 2
fi

# Forbidden HELM component directory / module names (matched case-insensitively).
COMPONENTS='tick|logs|axis|amio|span|dagr'

# C / C++ include violation:
#   #include <COMPONENT/...>   or   #include "COMPONENT/..."
#
# The REQUIRED trailing slash distinguishes a HELM component *directory*
# (e.g. <span/halo_span.hpp>, forbidden) from the C++ standard header <span>
# (no slash, allowed). The optional leading path group also catches namespaced
# layouts such as <helm/span/foo.hpp>.
CPP_PATTERN="^[[:space:]]*#[[:space:]]*include[[:space:]]*[<\"]([^>\"]*/)?($COMPONENTS)/"

# Fortran module violation:
#   use COMPONENT        use COMPONENT_mod        use COMPONENT, only: ...
#
# Bounded so HALO's own modules and intrinsics stay allowed
# (e.g. `use halo_mod`, `use mpi`, `use, intrinsic :: iso_c_binding`).
FORTRAN_PATTERN="^[[:space:]]*use[[:space:]]+($COMPONENTS)([_,[:space:]]|\$)"

# Prefer the shipping source/header trees. Fall back to the whole ROOT when
# none are present (e.g. when scanning a scratch directory during testing).
SCAN_ROOTS=""
for d in include src fortran; do
    if [ -d "$ROOT/$d" ]; then
        SCAN_ROOTS="$SCAN_ROOTS $ROOT/$d"
    fi
done
if [ -z "$SCAN_ROOTS" ]; then
    SCAN_ROOTS="$ROOT"
fi

# Directories that never contain shipping source (build artifacts, deps, vcs).
# shellcheck disable=SC2086  # intentional word-splitting of SCAN_ROOTS
cpp_hits=$(find $SCAN_ROOTS \
    \( -path '*/build/*' -o -path '*/_deps/*' -o -path '*/CMakeFiles/*' -o -path '*/.git/*' \) -prune -o \
    -type f \( -name '*.hpp' -o -name '*.hh' -o -name '*.hxx' -o -name '*.h' \
               -o -name '*.cpp' -o -name '*.cc' -o -name '*.cxx' \) \
    -exec grep -nEiH "$CPP_PATTERN" {} + 2>/dev/null)

# shellcheck disable=SC2086
fortran_hits=$(find $SCAN_ROOTS \
    \( -path '*/build/*' -o -path '*/_deps/*' -o -path '*/CMakeFiles/*' -o -path '*/.git/*' \) -prune -o \
    -type f \( -name '*.f90' -o -name '*.F90' -o -name '*.f' -o -name '*.F' \) \
    -exec grep -nEiH "$FORTRAN_PATTERN" {} + 2>/dev/null)

all_hits=""
if [ -n "$cpp_hits" ]; then
    all_hits="$cpp_hits"
fi
if [ -n "$fortran_hits" ]; then
    if [ -n "$all_hits" ]; then
        all_hits="$all_hits
$fortran_hits"
    else
        all_hits="$fortran_hits"
    fi
fi

if [ -n "$all_hits" ]; then
    count=$(printf '%s\n' "$all_hits" | grep -c .)
    echo "check_tier1_isolation: FAIL — forbidden HELM component reference(s) found:" >&2
    printf '%s\n' "$all_hits" >&2
    echo "" >&2
    echo "Found $count forbidden include/use reference(s)." >&2
    echo "HALO is a Tier 1 library and MUST NOT depend on TICK, LOGS, AXIS, AMIO, SPAN, or DAGR (Requirement 13)." >&2
    exit 1
fi

echo "check_tier1_isolation: PASS — no forbidden HELM component includes found under:$SCAN_ROOTS"
exit 0
