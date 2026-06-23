# CONF — Configuration Object & Notation Framework

**Tier 1 C++20 compiled micro-library** within the [HELM](../../README.md) ecosystem providing
stateless RAII YAML parsing for HELM configuration management.

CONF replaces manual yaml-cpp boilerplate with a clean, exception-safe C++ API
(`conf::Config`, `conf::Value`) and an optional Fortran C-interop layer (`conf_mod`)
for legacy Fortran domain models. All parsing is stateless and thread-safe; each
`Config` object owns its parsed tree independently.

## Features

- **Stateless RAII parser** — `conf::Config` owns its yaml-cpp node tree; no global
  state, no singletons, safe for multi-threaded use.
- **Type-safe value access** — `conf::Value` provides `.as<T>()` with clear error
  messages on type mismatch or missing key.
- **Fortran interop** — optional `conf_mod` Fortran module via `iso_c_binding` for
  reading YAML from Fortran domain models (controlled by `BUILD_FORTRAN=ON`).
- **Strict tier isolation** — no HALO, AMIO, AXIS, DAGR, TICK, LOGS, SPAN, or BLEND
  cross-library includes.
- **Single dependency** — links only yaml-cpp (fetched automatically via CMake
  FetchContent or a system install).

## Prerequisites

| Dependency | Version | Notes |
|---|---|---|
| C++20 compiler | GCC ≥ 13, Clang ≥ 16 | Required |
| CMake | ≥ 3.21 | Required |
| yaml-cpp | ≥ 0.8 | Required (auto-fetched or system via `CONF_USE_SYSTEM_YAMLCPP`) |
| GTest | any recent | Required when `BUILD_TESTING=ON` |
| RapidCheck | latest | Required when `BUILD_TESTING=ON` |
| gfortran | ≥ 13 | Required when `BUILD_FORTRAN=ON` |

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
all toolchain dependencies (GCC-13, yaml-cpp, GTest, RapidCheck, gfortran) are
available system-wide.

## CMake Configure and Build

```bash
# Inside the container, navigate to the CONF library
cd /workspace/helm-project/libs/conf

# Configure (tests and Fortran layer enabled)
cmake -B build \
  -DCMAKE_CXX_STANDARD=20 \
  -DBUILD_TESTING=ON \
  -DBUILD_FORTRAN=ON

# Build
cmake --build build --parallel $(nproc)
```

### CMake Options

| Option | Default | Description |
|---|---|---|
| `BUILD_TESTING` | `OFF` | Build the GTest + RapidCheck test suite |
| `BUILD_FORTRAN` | `ON` | Build the Fortran iso_c_binding interop layer (`conf_mod`) |
| `CONF_USE_SYSTEM_YAMLCPP` | `OFF` | Use system yaml-cpp instead of CMake FetchContent |

### Consuming CONF from a downstream project

```cmake
# Add the CONF source tree as a subdirectory
add_subdirectory(/path/to/libs/conf ${CMAKE_BINARY_DIR}/conf_build)

# Or install CONF and use find_package
find_package(CONF REQUIRED)

# Link against the exported alias
target_link_libraries(my_target PRIVATE HELM::CONF)
```

Include the umbrella header:

```cpp
#include <conf/conf.hpp>
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

# Run only Fortran integration tests (requires BUILD_FORTRAN=ON)
ctest -L fortran --output-on-failure
```

The test suite includes:

- **Unit tests** (GTest) — YAML parsing correctness, type-safe value access,
  exception safety on malformed input and missing keys.
- **Property-based tests** (RapidCheck) — round-trip invariants and value accessor
  correctness over randomized YAML structures.
- **Fortran integration tests** — end-to-end verification of the `conf_mod` C-interop
  layer from Fortran source.

## License

This project is part of the NOAA-EMC Ecosystem.

See [LICENSE](../../LICENSE) and [DISCLAIMER](../../DISCLAIMER) for details.
