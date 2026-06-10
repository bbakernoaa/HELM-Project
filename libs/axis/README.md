# AXIS — Arbitrary eXgrid Interpolation Solver

**Tier 1 C++20 micro-library** within the [HELM](../../README.md) ecosystem providing
stateless spatial interpolation (regridding) for Earth-system fields.

AXIS replaces the legacy ESMF spatial-discretization stack (`ESMF_Mesh`,
`ESMF_Grid`, `ESMF_LocStream`, `ESMF_Regrid`, and the offline
`ESMF_RegridWeightGen` application) with a decentralized, zero-copy,
hardware-portable toolkit built on Kokkos.

## Features

- **Zero-copy memory interface** — all field data crosses the boundary as
  non-owning `std::mdspan<layout_left>` views (no copies, preserves Fortran
  column-major layout).
- **Hardware-portable** — all parallelism via Kokkos; runs on CPU and GPU
  (CUDA/HIP) from a single code path.
- **Plain-data ingest contract** (`GridDescriptor`) — any producer (AMIO today,
  Python/numpy tomorrow) populates the same descriptor; AXIS is producer-agnostic.
- **Interpolation methods** — bilinear and first-order conservative weight
  generation with sparse-matrix apply (SpMV).
- **Distributed regridding seam** — publishes `HaloPattern` (plain data, no MPI
  types) for a caller to hand directly to HALO.
- **No file-format libraries** — AXIS links no NetCDF, HDF5, ecCodes, or
  yaml-cpp; all byte-level I/O is delegated to AMIO via the descriptor contract.

## Prerequisites

| Dependency | Version | Notes |
|---|---|---|
| C++20 compiler | GCC ≥ 13, Clang ≥ 16 | Required |
| CMake | ≥ 3.21 | Required |
| Kokkos | ≥ 5.1 | Required (`find_package(Kokkos REQUIRED)`) |
| PROJ (libproj) | ≥ 9 | Optional — controlled by `AXIS_ENABLE_PROJ` (default ON) |
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
all toolchain dependencies (GCC-13, Kokkos, GTest, RapidCheck) are available
system-wide.

## CMake Configure and Build

```bash
# Inside the container, navigate to the AXIS library
cd /workspace/helm-project/libs/axis

# Configure (Release build, tests enabled)
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DAXIS_ENABLE_PROJ=ON

# Build
cmake --build build --parallel $(nproc)
```

### CMake Options

| Option | Default | Description |
|---|---|---|
| `AXIS_ENABLE_PROJ` | `ON` | Enable PROJ-based coordinate transforms |
| `BUILD_TESTING` | `OFF` | Build the GTest + RapidCheck test suite |
| `BUILD_FORTRAN` | `OFF` | Build the Fortran iso_c_binding interop layer |

### Consuming AXIS from a downstream project

```cmake
find_package(AXIS REQUIRED)
target_link_libraries(my_target PRIVATE HELM::AXIS)
```

## Running Tests

```bash
# After building with BUILD_TESTING=ON
cd build
ctest --output-on-failure
```

The test suite includes:

- **Unit tests** (GTest) — conservation, constant-field preservation, bilinear
  exactness, SpMV correctness, descriptor validation, Gmsh round-trip, RAII
  handle verification.
- **Property-based tests** (RapidCheck) — universal correctness properties
  validated over randomized inputs (≥ 100 iterations per property).
- **Tier 1 isolation scan** — static check verifying no forbidden `#include`
  directives (HALO, AMIO, TICK, LOGS, SPAN, DAGR, eckit, domain headers).

## Library Target

The build produces `HELM::AXIS` — a C++20 library linking Kokkos as a PUBLIC
dependency. PROJ (when enabled) is linked PRIVATE and does not propagate to
consumers.

## License

This project is part of the NOAA-EMC Ecosystem.

See [LICENSE](../../LICENSE) and [DISCLAIMER](../../DISCLAIMER) for details.
