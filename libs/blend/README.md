# BLEND — Stateless Array Blending Kernels

**Tier 1 C++20 header-only micro-library** within the [HELM](../../README.md) ecosystem providing
stateless element-wise array blending kernels dispatched via Kokkos execution spaces.

BLEND replaces ad-hoc interpolation loops scattered across domain models with two
hardware-portable, zero-dependency kernel functions: `LinearBlendKernel` (weighted
interpolation) and `StepBlendKernel` (nearest-neighbor step-select). It is entirely
stateless, has no knowledge of time, calendars, grids, or domain physics, and links
only Kokkos.

## Features

- **Two kernels** — `LinearBlendKernel` (`target[i] = left[i]*(1-α) + right[i]*α`) and
  `StepBlendKernel` (`target[i] = α < 0.5 ? left[i] : right[i]`).
- **Header-only** — zero compiled sources; include `<blend/helm_math_blend.hpp>` and link
  Kokkos; no link step required for BLEND itself.
- **Hardware-portable** — all parallelism via Kokkos; runs on CPU, CUDA, and HIP from a
  single code path.
- **Zero-copy memory interface** — all data crosses the boundary as non-owning
  `Kokkos::View<double*, MemoryUnmanaged>` (no allocation, no copy).
- **Strict tier isolation** — no NetCDF, HDF5, MPI, yaml-cpp, or other HELM siblings.

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
# Inside the container, navigate to the BLEND library
cd /workspace/helm-project/libs/blend

# Configure (tests enabled)
cmake -B build \
  -DCMAKE_CXX_STANDARD=20 \
  -DBLEND_BUILD_TESTING=ON

# Build
cmake --build build --parallel $(nproc)
```

### CMake Options

| Option | Default | Description |
|---|---|---|
| `BLEND_BUILD_TESTING` | `OFF` | Build the GTest + RapidCheck test suite |

### Consuming BLEND from a downstream project

```cmake
# In your CMakeLists.txt — add the BLEND source tree as a subdirectory
add_subdirectory(/path/to/libs/blend ${CMAKE_BINARY_DIR}/blend_build)

# Or install BLEND and use find_package
find_package(BLEND REQUIRED)

# Link against the exported alias
target_link_libraries(my_target PRIVATE HELM::BLEND)
```

Include the umbrella header:

```cpp
#include <blend/helm_math_blend.hpp>
```

## Running Tests

```bash
# After building with BLEND_BUILD_TESTING=ON
cd build
ctest --output-on-failure

# Run only unit tests
ctest -L unit --output-on-failure

# Run only property tests
ctest -L property --output-on-failure
```

The test suite includes:

- **Unit tests** (GTest) — fixed-input formula verification for `LinearBlendKernel`
  and `StepBlendKernel`, extent-mismatch exception checks, and empty-array correctness.
- **Property-based tests** (RapidCheck) — identity property (α=0 → left, α=1 → right),
  step-select correctness (output == one of the two inputs), and interpolation boundedness
  (output ∈ [min(left,right), max(left,right)] for α∈[0,1]).

## License

This project is part of the NOAA-EMC Ecosystem.

See [LICENSE](../../LICENSE) and [DISCLAIMER](../../DISCLAIMER) for details.
