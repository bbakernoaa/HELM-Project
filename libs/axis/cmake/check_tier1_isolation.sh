#!/usr/bin/env bash
# ##############################################################################
# check_tier1_isolation.sh — Static scan for HELM Tier 1 isolation violations
#
# Scans AXIS source and header files for forbidden #include directives and
# forbidden symbols that would violate HELM Law #4 (no cross-Tier-1 deps).
#
# Usage:
#   ./cmake/check_tier1_isolation.sh [AXIS_ROOT_DIR]
#
# Arguments:
#   AXIS_ROOT_DIR  Path to the AXIS source tree (default: script's parent/../)
#
# Returns:
#   Exit 0  — No violations found (clean).
#   Exit 1  — One or more violations found (printed to stdout with file:line).
#
# Forbidden patterns:
#   Include directives for: HALO, AMIO, TICK, LOGS, SPAN, DAGR, eckit, domain
#   Symbols: MPI_Comm, nc_*, codes_handle, eckit::, yaml-cpp, TensorStore,
#            NetCDF_Handle, Grib_Handle
# ##############################################################################

set -euo pipefail

# Determine AXIS root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AXIS_ROOT="${1:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

# Validate that the directory exists
if [[ ! -d "${AXIS_ROOT}" ]]; then
    echo "ERROR: AXIS root directory not found: ${AXIS_ROOT}" >&2
    exit 1
fi

# Directories to scan (source + headers, excluding third_party and build)
SCAN_DIRS=(
    "${AXIS_ROOT}/include"
    "${AXIS_ROOT}/src"
)

# Verify at least one scan directory exists
found_dir=0
for dir in "${SCAN_DIRS[@]}"; do
    if [[ -d "${dir}" ]]; then
        found_dir=1
    fi
done
if [[ ${found_dir} -eq 0 ]]; then
    echo "ERROR: No scannable directories found under ${AXIS_ROOT}" >&2
    exit 1
fi

violations=0

# ─── Forbidden #include patterns ─────────────────────────────────────────────
# These grep patterns match #include directives referencing forbidden libraries.
FORBIDDEN_INCLUDES=(
    # HELM sibling Tier 1/2/3 components
    '#include.*[<"]halo/'
    '#include.*[<"]amio/'
    '#include.*[<"]tick/'
    '#include.*[<"]logs/'
    '#include.*[<"]span/'
    '#include.*[<"]dagr/'
    '#include.*[<"]HALO/'
    '#include.*[<"]AMIO/'
    '#include.*[<"]TICK/'
    '#include.*[<"]LOGS/'
    '#include.*[<"]SPAN/'
    '#include.*[<"]DAGR/'
    # eckit (AMIO's YAML/config dependency)
    '#include.*[<"]eckit/'
    # File-format libraries (belong to AMIO, not AXIS)
    '#include.*[<"]netcdf'
    '#include.*[<"]hdf5'
    '#include.*[<"]eccodes'
    '#include.*[<"]yaml-cpp/'
    '#include.*[<"]tensorstore/'
    '#include.*[<"]g2c'
    '#include.*[<"]nceplibs'
    # Domain-science headers (AXIS must be blind to domain)
    '#include.*[<"]domain/'
    '#include.*[<"]model/'
)

# ─── Forbidden symbol patterns ───────────────────────────────────────────────
# These match identifiers that must never appear in AXIS code (outside comments).
FORBIDDEN_SYMBOLS=(
    '\bMPI_Comm\b'
    '\bMPI_Request\b'
    '\bnc_[a-z]'
    '\bcodes_handle'
    '\beckit::'
    '\bTensorStore\b'
    '\bNetCDF_Handle\b'
    '\bGrib_Handle\b'
    '\byaml-cpp\b'
)

# ─── Scan function ───────────────────────────────────────────────────────────

scan_pattern() {
    local pattern="$1"
    local label="$2"
    local skip_comments="${3:-no}"  # "yes" to filter out comment-only lines

    for dir in "${SCAN_DIRS[@]}"; do
        if [[ ! -d "${dir}" ]]; then
            continue
        fi
        # Find all .hpp and .cpp files, excluding third_party and build dirs
        while IFS= read -r -d '' file; do
            matches=$(grep -nE "${pattern}" "${file}" 2>/dev/null || true)
            if [[ -n "${matches}" ]]; then
                while IFS= read -r line; do
                    # Optionally skip lines that are C/C++ comments
                    if [[ "${skip_comments}" == "yes" ]]; then
                        # Extract the code portion (after line number)
                        code_part="${line#*:}"  # strip "linenum:"
                        code_part="${code_part#*:}"  # strip second field if file:line:content
                        # Trim leading whitespace
                        trimmed="${code_part#"${code_part%%[![:space:]]*}"}"
                        # Skip if line is a comment (// or /* or * continuation)
                        if [[ "${trimmed}" == //* ]] || [[ "${trimmed}" == \** ]] || [[ "${trimmed}" == \#* ]]; then
                            continue
                        fi
                    fi
                    echo "VIOLATION [${label}]: ${file}:${line}"
                    violations=$((violations + 1))
                done <<< "${matches}"
            fi
        done < <(find "${dir}" \( -name "*.hpp" -o -name "*.cpp" -o -name "*.h" \) \
                   -not -path "*/third_party/*" \
                   -not -path "*/build/*" \
                   -print0)
    done
}

# ─── Fortran use-statement scan ──────────────────────────────────────────────

scan_fortran_use() {
    local pattern="$1"
    local label="$2"

    for dir in "${SCAN_DIRS[@]}"; do
        if [[ ! -d "${dir}" ]]; then
            continue
        fi
        while IFS= read -r -d '' file; do
            matches=$(grep -niE "${pattern}" "${file}" 2>/dev/null || true)
            if [[ -n "${matches}" ]]; then
                while IFS= read -r line; do
                    echo "VIOLATION [${label}]: ${file}:${line}"
                    violations=$((violations + 1))
                done <<< "${matches}"
            fi
        done < <(find "${dir}" \( -name "*.f90" -o -name "*.F90" \) \
                   -not -path "*/third_party/*" \
                   -not -path "*/build/*" \
                   -print0)
    done
}

# ─── Run scans ───────────────────────────────────────────────────────────────

echo "=== AXIS Tier 1 Isolation Scan ==="
echo "  Scanning: ${AXIS_ROOT}"
echo ""

for pattern in "${FORBIDDEN_INCLUDES[@]}"; do
    scan_pattern "${pattern}" "forbidden-include"
done

for pattern in "${FORBIDDEN_SYMBOLS[@]}"; do
    scan_pattern "${pattern}" "forbidden-symbol" "yes"
done

# Fortran use statements for forbidden modules
FORBIDDEN_FORTRAN_USE=(
    '^\s*use\s+halo'
    '^\s*use\s+amio'
    '^\s*use\s+tick'
    '^\s*use\s+logs'
    '^\s*use\s+span'
    '^\s*use\s+dagr'
    '^\s*use\s+eckit'
)

for pattern in "${FORBIDDEN_FORTRAN_USE[@]}"; do
    scan_fortran_use "${pattern}" "forbidden-fortran-use"
done

# ─── Report results ──────────────────────────────────────────────────────────

echo ""
if [[ ${violations} -eq 0 ]]; then
    echo "PASS: No Tier 1 isolation violations found."
    exit 0
else
    echo "FAIL: ${violations} Tier 1 isolation violation(s) found."
    exit 1
fi
