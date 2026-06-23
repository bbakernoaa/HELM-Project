# DAGR — Directed Acyclic Graph Router

**Tier 3 C++20 compiled micro-library** within the [HELM](../../README.md) ecosystem providing
YAML-driven asynchronous event-loop DAG orchestration for Earth-system model coupling.

DAGR performs **zero mathematical calculations** — it routes data pointers between
lower-tier HELM engines (TICK, AMIO, BLEND, HALO) and manages execution dependencies
via a directed acyclic graph. Pipeline topology is defined in YAML configuration and
resolved at startup, enabling runtime coupling graph changes without recompilation.

## Features

- **YAML-driven pipeline topology** — DAG nodes, edges, and execution order are declared
  in `pipeline_config.yml`; no recompilation needed to rewire coupling.
- **Asynchronous event loop** — each stage runs as a non-blocking task dispatched through
  an internal event loop; completion tokens notify downstream nodes.
- **Zero computation** — DAGR routes pointers and schedules calls; all arithmetic,
  interpolation, and I/O is delegated to TICK, BLEND, AXIS, HALO, and AMIO.
- **Rank pool management** — assigns MPI ranks to pipeline stages and enforces rank
  budget constraints at initialization time.
- **Static zero-computation enforcement** — CI scans DAGR source for math headers and
  raw MPI calls to prevent accidental computation from entering the orchestration layer.

## Prerequisites

| Dependency | Version | Notes |
|---|---|---|
| C++20 compiler | GCC ≥ 13, Clang ≥ 16 | Required |
| CMake | ≥ 3.21 | Required |
| yaml-cpp | ≥ 0.8 | Required for pipeline config parsing |
| GTest | any recent | Required when `BUILD_TESTING=ON` |
| RapidCheck | latest | Required when `BUILD_TESTING=ON` |

All prerequisites are pre-installed in the HELM Docker development container.

## Docker Container Launch

Build and start the HELM development container from the project root:

```bash
# From the HELM project root (where docker-compose.yml lives)
docker compose up -d --build

# Attach to the running container
docker compose exec helm-dev bash
```

Inside the container the workspace is mounted at `/workspace/helm-project` and
all toolchain dependencies (GCC-13, yaml-cpp, GTest, RapidCheck) are available
system-wide.

## CMake Configure and Build

```bash
# Inside the container, navigate to the DAGR library
cd /workspace/helm-project/libs/dagr

# Configure (tests enabled)
cmake -B build \
  -DCMAKE_CXX_STANDARD=20 \
  -DBUILD_TESTING=ON

# Build
cmake --build build --parallel $(nproc)
```

### CMake Options

| Option | Default | Description |
|---|---|---|
| `BUILD_TESTING` | `OFF` | Build the GTest + RapidCheck test suite |

### Consuming DAGR from a downstream project

```cmake
# Add the DAGR source tree as a subdirectory
add_subdirectory(/path/to/libs/dagr ${CMAKE_BINARY_DIR}/dagr_build)

# Or install DAGR and use find_package
find_package(DAGR REQUIRED)

# Link against the exported alias
target_link_libraries(my_target PRIVATE HELM::DAGR)
```

Include the pipeline configuration header:

```cpp
#include <dagr/pipeline_config.hpp>
```

## Running Tests

```bash
# After building with BUILD_TESTING=ON
cd build
ctest --output-on-failure

# Run only unit tests
ctest -L unit --output-on-failure

# Run only property tests
ctest -L property --output-on-failure
```

The test suite includes:

- **Unit tests** (GTest) — pipeline config parsing, DAG topology validation, completion
  token sequencing, rank pool assignment, and event loop scheduling.
- **Property-based tests** (RapidCheck) — DAG invariants (no cycles, valid edge
  ordering) over randomized pipeline topologies.

## License

This project is part of the NOAA-EMC Ecosystem.

See [LICENSE](../../LICENSE) and [DISCLAIMER](../../DISCLAIMER) for details.
