# LOGS — Lightweight Operational Global Status

LOGS is a Tier 1 C++20 micro-library within the [HELM](../../README.md) ecosystem.
It provides thread-safe, MPI-rank-aware operational status, logging, and
error-handling for distributed HPC jobs, replacing the legacy ESMF logging
surface (`ESMF_LogWrite`, `ESMF_LogErr`, `ESMF_LogSet`, and the per-PET
one-log-file-per-rank model).

## Key Capabilities

- **Rank-Stamped Logging** — Every log record is automatically annotated with
  the originating MPI rank for traceability across thousands of processes.
- **Multi-Rank Consolidation** — Identical messages emitted simultaneously
  across thousands of ranks are consolidated into a single representative
  entry with compact rank-range annotations (e.g., "ranks 0-9999"), preventing
  filesystem I/O storms.
- **Synchronized Abort** — A FATAL message deterministically terminates the
  entire distributed job via `MPI_Abort`, preventing zombie processes.
- **Exception Boundary** — All public entry points are `noexcept` from the
  caller's perspective; logging failures degrade gracefully and never crash.
- **RAII Scoped Context** — Nested trace-context labels annotate log records
  automatically and are cleaned up even during stack unwinding.
- **Stack-Trace Formatting** — Pure formatting over caller-supplied frames;
  never reads files or symbol tables.
- **No Input Parsing** — Strictly an output and status mechanism. No YAML,
  no namelist, no configuration-file reading. All behavior configured via
  programmatic API calls.
- **No Kokkos Dependency** — Pure C++20 + MPI status utility with no GPU
  compute or device offload.

## Prerequisites

| Dependency | Minimum Version | Notes |
|---|---|---|
| CMake | 3.21+ | Build system generator |
| C++ Compiler | C++20 support | GCC 13+, Clang 16+ |
| MPI (CXX) | MPI-3.0+ | OpenMPI, MPICH, or Cray MPICH |
| Google Test | 1.14+ | Required when `BUILD_TESTING=ON` |
| RapidCheck | latest | Required for property-based tests |

All prerequisites are pre-installed in the HELM project Docker container.

## Getting Started

### 1. Launch the Development Container

From the HELM project root:

```bash
docker compose up -d --build
docker compose exec helm-dev bash
```

You are now at `/workspace/helm-project` inside the container.

### 2. CMake Configure

```bash
cd /workspace/helm-project/libs/logs
cmake -B build -G Ninja \
  -DCMAKE_CXX_STANDARD=20 \
  -DLOGS_BUILD_TESTING=ON
```

### 3. Build

```bash
cmake --build build --parallel $(nproc)
```

### 4. Run Tests

```bash
ctest --test-dir build --output-on-failure
```

To run only the property-based tests:

```bash
ctest --test-dir build --output-on-failure -R prop
```

## CMake Targets

| Target | Description |
|---|---|
| `logs` | Main library (static or shared) |
| `HELM::LOGS` | Namespace alias for downstream consumption |

## Consuming LOGS in Downstream Projects

```cmake
find_package(LOGS REQUIRED)
target_link_libraries(my_target PRIVATE HELM::LOGS)
```

MPI include paths and link flags propagate transitively.

## API Overview

Include the umbrella header for full access:

```cpp
#include <logs/logs.hpp>
```

### Core Types

| Header | Type | Description |
|---|---|---|
| `logs/severity.hpp` | `logs::Severity_Level` | Ordered enum: DEBUG < INFO < WARNING < ERROR < FATAL |
| `logs/log_record.hpp` | `logs::Log_Record` | Immutable log value object |
| `logs/sink.hpp` | `logs::Sink` | Output-stream-only destination wrapper |
| `logs/scoped_context.hpp` | `logs::Scoped_Context` | RAII thread-local trace-context label |
| `logs/stack_trace.hpp` | `logs::Stack_Frame` | Caller-supplied frame data (POD) |
| `logs/logger.hpp` | `logs::Logger` | Central thread-safe entry point |

### Quick Usage

```cpp
#include <logs/logs.hpp>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    logs::Logger logger;
    logger.configure_communicator(MPI_COMM_WORLD);
    logger.set_threshold(logs::Severity_Level::INFO);

    // Basic logging
    logger.log(logs::Severity_Level::INFO, "Initialization complete");

    // Scoped context for traceability
    {
        logs::Scoped_Context ctx("time_step_loop");
        logger.log(logs::Severity_Level::DEBUG, "Entering main loop");
    }

    // Consolidation across ranks
    logger.consolidate();

    MPI_Finalize();
    return 0;
}
```

## Project Structure

```text
libs/logs/
├── CMakeLists.txt              # Standalone build (produces HELM::LOGS)
├── cmake/
│   ├── LOGSConfig.cmake.in     # Exported config template
│   └── LOGSConfigVersion.cmake.in
├── include/logs/               # Public C++ headers
│   ├── logs.hpp                # Umbrella header
│   ├── severity.hpp            # Severity_Level enum + to_string
│   ├── source_location.hpp     # Source_Location annotation type
│   ├── log_record.hpp          # Log_Record immutable value object
│   ├── sink.hpp                # Sink output-stream wrapper
│   ├── scoped_context.hpp      # Scoped_Context RAII type
│   ├── stack_trace.hpp         # Stack_Frame + format_stack_trace()
│   ├── logger.hpp              # Logger central entry point
│   └── detail/                 # Internal implementation headers
│       ├── mpi_environment.hpp
│       ├── serialized_mpi_guard.hpp
│       ├── consolidation.hpp
│       └── context_stack.hpp
├── src/                        # Implementation files
│   ├── logger.cpp
│   ├── sink.cpp
│   ├── scoped_context.cpp
│   ├── stack_trace.cpp
│   └── detail/
│       ├── mpi_environment.cpp
│       └── consolidation.cpp
├── tests/                      # Unit + property-based tests
│   ├── CMakeLists.txt
│   ├── test_severity.cpp
│   ├── test_log_record.cpp
│   ├── test_rank_stamping.cpp
│   ├── test_filtering.cpp
│   ├── test_consolidation.cpp
│   ├── test_synchronized_abort.cpp
│   ├── test_sinks.cpp
│   ├── test_stack_trace.cpp
│   ├── test_scoped_context.cpp
│   ├── test_exception_boundary.cpp
│   ├── test_thread_safety.cpp
│   ├── prop_severity_ordering.cpp
│   ├── prop_rank_stamping.cpp
│   ├── prop_filtering.cpp
│   ├── prop_consolidation.cpp
│   ├── prop_rank_range.cpp
│   ├── prop_stack_trace.cpp
│   ├── prop_scoped_context.cpp
│   └── prop_format_round_trip.cpp
├── README.md
└── .gitignore
```

## License

See [LICENSE](../../LICENSE) and [DISCLAIMER](../../DISCLAIMER) in the HELM project root.
