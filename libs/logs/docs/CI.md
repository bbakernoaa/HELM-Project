# LOGS CI Pipeline

This document describes the Continuous Integration pipeline for the LOGS
(Lightweight Operational Global Status) micro-library. The pipeline verifies
architectural constraints, builds the library in isolation, and runs the full
test suite under multiple configurations including sanitizers.

All stages run **inside the HELM project Docker container** (`helm-dev`), so
the toolchain matches local development exactly.

## Pipeline Triggers

The pipeline runs on:

- Pushes modifying `libs/logs/**`, `Dockerfile`, `docker-compose.yml`, or
  the workflow file itself.
- Pull requests touching the same paths.
- Manual dispatch (`workflow_dispatch`).

---

## Stages

### Stage 1: Static Analysis

Two independent verification scans run before any compilation, catching
architectural violations early.

#### 1a. Tier 1 Isolation Scan

```bash
cd libs/logs && sh cmake/check_tier1_isolation.sh .
```

Scans all LOGS source and header files for `#include` directives referencing
forbidden HELM component path segments (`tick/`, `halo/`, `axis/`, `amio/`,
`span/`, `dagr/`). The C++20 standard `<span>` header (without a trailing `/`)
is explicitly permitted.

- **Exit 0**: All files pass.
- **Exit 1**: Forbidden includes detected — reports offending files and lines.

Requirements: 14.1, 14.5, 14.6, 14.7

#### 1b. No-Input-Parsing Scan

```bash
cd libs/logs && sh cmake/check_no_input_parsing.sh .
```

Scans all LOGS source and header files for indicators of input-parsing or
file-reading logic:

- YAML/markup-parsing library references (`yaml-cpp`, `nlohmann`, `tinyxml`,
  `rapidjson`, etc.)
- Namelist or configuration-format readers
- File-open-for-read or read-mode file APIs (`std::ifstream`, `fopen(...,"r")`,
  `open(..., O_RDONLY)`, etc.)

LOGS is strictly an output and status mechanism — it never reads input.

- **Exit 0**: No input-parsing indicators found.
- **Exit 1**: Offending references detected — reports files and lines.

Requirements: 10.5, 10.6, 10.7, 10.8

---

### Stage 2: Standalone CMake Build Inside Docker

```bash
cd libs/logs
cmake -B build-ci -G Ninja \
  -DCMAKE_CXX_STANDARD=20 \
  -DBUILD_TESTING=ON
cmake --build build-ci --parallel $(nproc)
```

This stage builds LOGS **in isolation** — without any other HELM component
source trees present in the build graph. The build must produce:

- The `logs` shared library (`liblogs.so`)
- The **`HELM::LOGS`** CMake namespace alias
- All test executables (unit + property)

A downstream consumer test verifies the exported `HELM::LOGS` target is
consumable by an independent project that links only LOGS and MPI:

```bash
# Creates a minimal consumer project in /tmp, adds libs/logs as a subdirectory,
# and links against HELM::LOGS to prove the alias works standalone.
cmake -B build -G Ninja -DBUILD_TESTING=OFF
cmake --build build --target consumer
```

**This proves LOGS builds and exports correctly without requiring TICK, HALO,
AXIS, AMIO, SPAN, or DAGR source trees.**

Requirements: 15.2, 15.3, 15.7

---

### Stage 3: Unit Tests (`mpirun -np 4`)

```bash
cd libs/logs/build-ci && ctest -L unit --output-on-failure
```

Runs the Google Test unit test suite under MPI with 4 ranks via `ctest`. Tests
are labeled `unit` in CMake. The test suite covers:

- Rank stamping verification
- Consolidation (contiguous spans, distinct keys, single-rank)
- FATAL abort sequence
- Scoped context exception unwind
- Formatting round-trip
- Sink dispatch and failure isolation
- Thread-safety concurrency
- Exception boundary behavior

Requirements: 15.2, 15.3

---

### Stage 4: Property Tests (Single-Rank, Mocked MPI)

```bash
cd libs/logs/build-ci && ctest -L property --output-on-failure
```

Runs RapidCheck property-based tests at single rank with mocked MPI via the
interposition spy layer. Tests are labeled `property` in CMake. Properties
verified include:

- Severity ordering and label mapping (Properties 1–2)
- Log_Record immutability and FATAL classification (Properties 3–5)
- Rank stamping invariance (Properties 6–7)
- Severity filtering monotonicity (Properties 9–11)
- Consolidation key/range/lifecycle (Properties 12–18)
- Synchronized abort properties (Properties 19–22)
- Sink behavior (Properties 23–25)
- Stack-trace formatting (Properties 26–27)
- Scoped context LIFO/threading (Properties 29–33)
- Exception boundary (Properties 34–35)
- Concurrent record integrity (Property 36)
- MPI thread-level serialization (Property 37)

The interposition layer (`tests/mpi_interposition.hpp`) intercepts MPI calls
at link time, allowing deterministic verification of MPI interactions without
a live multi-rank runtime.

Requirements: 15.2, 15.3

---

### Stage 5: Sanitizer Builds (ASan + UBSan)

```bash
cd libs/logs
cmake -B build-asan -G Ninja \
  -DCMAKE_CXX_STANDARD=20 \
  -DBUILD_TESTING=ON \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
cmake --build build-asan --parallel $(nproc)
cd build-asan
ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 \
  ctest -L property --output-on-failure
```

A separate build with AddressSanitizer and UndefinedBehaviorSanitizer flags
enabled. This catches:

- Heap/stack buffer overflows
- Use-after-free
- Double-free
- Memory leaks (when enabled)
- Signed integer overflow
- Null pointer dereference
- Misaligned memory access
- Other undefined behavior

`detect_leaks=0` suppresses leak reports for process-global objects that
outlive `main()` by design (the MPI interposition layer). ASan still catches
use-after-free/overflow and UBSan still detects undefined behavior.

Requirements: 15.2, 15.3

---

## Docker Container

All stages execute inside the `helm-dev` Docker container (defined by the
project-root `Dockerfile` and `docker-compose.yml`). The container provides:

- GCC 13+ / Clang 16+ with full C++20 support
- OpenMPI with `OMPI_ALLOW_RUN_AS_ROOT` for CI execution
- CMake 3.21+, Ninja
- Google Test 1.14+
- RapidCheck (latest)
- ASan/UBSan runtime libraries

---

## Standalone Build Verification

The standalone build stage proves that **LOGS can be configured, built, and
consumed without any other HELM component source trees present**. Specifically:

1. The `libs/logs/CMakeLists.txt` is self-contained — it requires only CMake,
   a C++20 compiler, and MPI.
2. No `find_package()` or `add_subdirectory()` call references TICK, HALO,
   AXIS, AMIO, SPAN, or DAGR.
3. The exported `HELM::LOGS` target carries transitive MPI include/link
   requirements but no transitive dependency on other HELM libraries.
4. A downstream consumer project can `target_link_libraries(... HELM::LOGS)`
   and build successfully with only the LOGS source tree available.

This isolation guarantee is a core Tier 1 architectural requirement — each
micro-library in the HELM ecosystem must be independently buildable.

Requirements: 15.7

---

## Workflow File

The executable GitHub Actions workflow implementing this pipeline is at:

```text
.github/workflows/logs-ci.yml
```

See that file for the exact commands and job configuration.
