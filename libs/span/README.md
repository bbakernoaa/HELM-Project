# SPAN — Shared Pointer & Array Network

**Tier 2 C++20 header-only micro-library** within the [HELM](../../README.md) ecosystem providing
zero-copy Fortran/C++ boundary views with coherency-state management and triple-buffered
I/O protection.

SPAN wraps native Fortran pointers in `std::mdspan`-compatible views, maintains a coherency
state machine for host/device synchronization, and provides triple-buffered I/O protection
for asynchronous background writes. It bridges legacy Fortran domain models to the modern
C++ HELM infrastructure without copying field data.

## Features

- **Zero-copy boundary layer** — wraps Fortran-owned memory as non-owning Kokkos views;
  SPAN never allocates or copies field data.
- **Coherency state machine** — tracks host/device synchronization state to prevent
  stale reads across asynchronous dispatch.
- **Triple-buffered I/O protection** — shields Fortran domain arrays from overwrite
  during concurrent background writes.
- **Hardware-portable** — all parallelism via Kokkos; supports CPU, CUDA, and HIP.
- **Strict tier isolation** — no HALO, AMIO, AXIS, DAGR, CONF, TICK, LOGS, or BLEND
  cross-library includes.

## Prerequisites

| Dependency | Version | Notes |
|---|---|---|
| C++20 compiler | GCC ≥ 13, Clang ≥ 16 | Required |
| CMake | ≥ 3.21 | Required |
| Kokkos | ≥ 5.1 | Required (`find_package(Kokkos REQUIRED)`) |
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
all toolchain dependencies (GCC-13, Kokkos, GTest, RapidCheck) are available
system-wide.

## CMake Configure and Build

```bash
# Inside the container, navigate to the SPAN library
cd /workspace/helm-project/libs/span

# Configure (tests enabled)
cmake -B build \
  -DCMAKE_CXX_STANDARD=20 \
  -DSPAN_BUILD_TESTING=ON

# Build
cmake --build build --parallel $(nproc)
```

### CMake Options

| Option | Default | Description |
|---|---|---|
| `BUILD_TESTING` | `OFF` | Build the GTest + RapidCheck test suite |

### Consuming SPAN from a downstream project

```cmake
# Add the SPAN source tree as a subdirectory
add_subdirectory(/path/to/libs/span ${CMAKE_BINARY_DIR}/span_build)

# Or install SPAN and use find_package
find_package(SPAN REQUIRED)

# Link against the exported alias
target_link_libraries(my_target PRIVATE HELM::SPAN)
```

Include the primary header:

```cpp
#include <span/span.hpp>
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

- **Unit tests** (GTest) — field view wrapping, coherency state transitions, triple-buffer
  rotation, and extent validation.
- **Property-based tests** (RapidCheck) — universal correctness properties over randomized
  field sizes and access patterns.

## License

This project is part of the NOAA-EMC Ecosystem.

See [LICENSE](../../LICENSE) and [DISCLAIMER](../../DISCLAIMER) for details.
