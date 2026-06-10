# HALO: Hardware-Abstracted Link Operations

HALO is a Tier 1 C++20 micro-library within the [HELM](../../README.md) ecosystem. It provides RAII-wrapped MPI communication primitives with compile-time GPU-aware halo exchange via Kokkos, replacing the legacy ESMF VM class and halo exchange infrastructure.

## Features

- **RAII Resource Safety** — All MPI handles (`MPI_Comm`, `MPI_Request`, `MPI_Win`) are scope-bound C++ objects guaranteeing cleanup on all exit paths, including exceptions.
- **Compile-Time Memory Dispatch** — Template metaprogramming detects the Kokkos memory space at compile time and selects GPU-direct MPI or host-staged transfer paths.
- **Precomputed Exchange Plans** — Neighbor topology and buffer geometry are computed once and reused across timesteps.
- **Fortran C-Interop** — Full `iso_c_binding` interface for incremental adoption by legacy NUOPC/ESMF Fortran models.

## Prerequisites

| Dependency | Minimum Version | Notes |
|---|---|---|
| CMake | 3.21+ | Build system generator |
| C++ Compiler | C++20 support | GCC 13+, Clang 16+, or NVHPC 23.1+ |
| MPI | Any with CXX component | OpenMPI, MPICH, or Cray MPICH |
| Kokkos | 4.3+ | Hardware portability layer |
| Google Test | 1.14+ | Optional — required when `BUILD_TESTING=ON` |
| RapidCheck | latest | Optional — required for property-based tests |
| Fortran Compiler | Fortran 2008 | Optional — gfortran 9+, ifort 2021+, or nvfortran 21.1+ |

All prerequisites are pre-installed in the HELM project Docker container.

## Docker Container Launch

The recommended development environment is the HELM Docker container built from the project root:

```bash
# From the HELM project root directory:

# Build and start the container (detached)
docker compose up -d --build

# Enter the running container
docker compose exec helm-dev bash

# You are now at /workspace/helm-project inside the container
# HALO source is at /workspace/helm-project/libs/halo/
```

## CMake Configure and Build

All commands below assume you are inside the Docker container at `/workspace/helm-project/libs/halo/`.

### Basic Build

```bash
cmake -B build -G Ninja \
  -DCMAKE_CXX_STANDARD=20
cmake --build build --parallel $(nproc)
```

### Build with Tests Enabled

```bash
cmake -B build -G Ninja \
  -DCMAKE_CXX_STANDARD=20 \
  -DBUILD_TESTING=ON
cmake --build build --parallel $(nproc)
```

### Build with GPU-Aware MPI

```bash
cmake -B build -G Ninja \
  -DCMAKE_CXX_STANDARD=20 \
  -DHALO_GPU_AWARE_MPI=ON
cmake --build build --parallel $(nproc)
```

### Build with Fortran Interface

```bash
cmake -B build -G Ninja \
  -DCMAKE_CXX_STANDARD=20 \
  -DBUILD_FORTRAN=ON
cmake --build build --parallel $(nproc)
```

### Full Development Build (All Options)

```bash
cmake -B build -G Ninja \
  -DCMAKE_CXX_STANDARD=20 \
  -DBUILD_TESTING=ON \
  -DBUILD_FORTRAN=ON \
  -DHALO_GPU_AWARE_MPI=OFF
cmake --build build --parallel $(nproc)
```

## Running Tests

Tests require MPI and are executed via `ctest` with `mpirun`:

```bash
cd build
ctest --output-on-failure
```

To run tests with a specific number of MPI ranks:

```bash
mpirun -np 4 ./tests/halo_tests
```

Property-based tests (RapidCheck) run in single-rank mode with mocked MPI:

```bash
ctest --output-on-failure -R property
```

## CMake Targets

| Target | Description |
|---|---|
| `halo` | Main shared library |
| `HELM::HALO` | Namespace alias for downstream consumption |
| `HELM::HALO_Fortran` | Fortran interface library (when `BUILD_FORTRAN=ON`) |

## Consuming HALO in Downstream Projects

```cmake
find_package(HALO REQUIRED)
target_link_libraries(my_target PRIVATE HELM::HALO)
```

MPI and Kokkos include paths and link flags propagate transitively.

## Project Structure

```
libs/halo/
├── CMakeLists.txt              # Standalone build (produces HELM::HALO)
├── cmake/                      # Exported CMake config templates
├── include/halo/              # Public C++ headers
│   ├── halo.hpp               # Umbrella header
│   ├── communicator.hpp
│   ├── request_guard.hpp
│   ├── window_guard.hpp
│   ├── halo_plan.hpp
│   ├── halo_handle.hpp
│   ├── exchange.hpp
│   ├── environment.hpp
│   └── detail/                # Internal implementation headers
├── src/                       # Implementation files
│   ├── detail/
│   └── fortran/               # C-interop layer
├── fortran/                   # Fortran halo_mod module
├── tests/                     # C++ unit and property tests
└── tests_fortran/             # Fortran integration tests
```

## License

See [LICENSE](../../LICENSE) and [DISCLAIMER](../../DISCLAIMER) in the HELM project root.
