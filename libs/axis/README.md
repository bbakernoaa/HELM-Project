# AXIS — Arbitrary eXgrid Interpolation Solver

**Tier 1 C++20 micro-library** within the [HELM](../../README.md) ecosystem providing
stateless spatial interpolation (regridding) for Earth-system fields.

AXIS replaces the legacy ESMF spatial-discretization stack (`ESMF_Mesh`,
`ESMF_Grid`, `ESMF_LocStream`, `ESMF_Regrid`, and the offline
`ESMF_RegridWeightGen` application) with a decentralized, zero-copy,
hardware-portable toolkit built on Kokkos, now fully packaged for Python, `xarray`, and **Dask parallel computing**.

---

## 🚀 Architectural Advantages vs. Legacy CDO/ESMF

AXIS represents a massive paradigm shift in high-performance Earth-system regridding, addressing the fundamental scaling bottlenecks of legacy CDO (SCRIP-based) and ESMF (ESMPy/xregrid) stacks:

1. **Exascale Speedups (10x to 127x faster):** Bypasses all disk-based file I/O overheads and executes parallel-on-device weight generation and SpMVs (Sparse Matrix-Vector multiplication) using Kokkos (CPU OpenMP, GPU CUDA/HIP).
2. **True 3D Spherical Cartesian BVH (Resolving Dateline Seams):** Legacy CDO/ESMF perform 2D search queries in flat $(\lambda, \theta)$ space, causing polar singularities and silent spatial misses at the $0^\circ/360^\circ$ periodic dateline seam. AXIS projects coordinates to 3D Cartesian space on the unit sphere and builds a **3D ArborX AABB tree**, natively resolving all wrapping seams and coordinate singularities.
3. **Spherical-Exact Analytical Conservative Overlaps:** Legacy SCRIP-based engines (CDO/ESMF) assume cell boundaries are straight lines in 2D coordinate space, leading to severe polar stretching and mass sum errors at high latitudes. AXIS treats cell boundaries as **true Great-Circle arcs** and computes intersections using exact spherical excess areas, guaranteeing flawless mass conservation.
4. **Z-Curve Morton Cache Alignment:** Unstructured queries are pre-sorted using a 3D Space-Filling Z-Curve, grouping parallel CPU/GPU threads spatially to maximize L1/L2 cache hit rates and eliminate memory bandwidth thrashing.
5. **Zero-Copy Memory Interfaces:** Preserves Fortran column-major (`LayoutLeft`/`np.asfortranarray`) memory layouts natively across boundaries, avoiding costly data layout transposition.

---

## 📊 Summary of Live Benchmark Results

*Measured on an ARM64 (aarch64) Linux VM under Docker with unified OpenMP parallel execution:*

| Case | Grid & Size | Method | CDO Time (s) | AXIS Time (s) | AXIS Speedup vs CDO |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **1** | Global upscale ($720\times360 \to 1440\times720$) | **Bilinear** | `0.804` | **`0.378`** | **$2.1\times$** ($38\times$ vs `xregrid`) |
| | | **Nearest** | `0.936` | **`0.289`** | **$3.2\times$** ($15\times$ vs `xregrid`) |
| | | **Conservative** | `2.866` | **`0.288`** | **$10.0\times$** ($81\times$ vs `xregrid`) |
| **2** | High-Res global ($259\text{k} \to 6.48\text{M}$ cells) | **Nearest** | `2.318` | **`1.470`** | **$1.5\times$** |
| | | **Conservative** | `12.033` | **`1.328`** | **$9.0\times$** |
| **3** | Regional Projected LCC ($120\times120 \to 100\times100$) | **Conservative** | `0.591` | **`0.054`** | **$10.8\times$** ($10.3\times$ vs `xregrid`) |
| **4** | Unstructured MPAS $\to$ Regular ($10\text{k} \to 90\times90$) | **Bilinear** | *FAILED* | **`0.123`** | **Exclusive!** |
| | | **Nearest** | `0.490` | **`0.147`** | **$3.3\times$** |
| | | **Conservative** | `0.728` | **`0.188`** | **$3.8\times$** |
| **5** | Regular $\to$ Unstructured MPAS ($360\times180 \to 2000$) | **Nearest** | `0.470` | **`0.209`** | **$2.2\times$** |
| | | **Conservative** | `0.492` | **`0.162`** | **$3.0\times$** |

---

## 🐍 Python & xarray/Dask Integration

AXIS is fully packaged for Python with deep `xarray`, `cf-xarray`, and **Dask** distributed cluster support. It installs seamlessly using standard packaging tools:

```bash
# Compile and install AXIS with Python bindings globally
pip install ./libs/axis
```

### 1. Basic Python Usage with xarray Accessor
AXIS registers a custom **`.axis`** accessor namespace on all xarray Datasets and DataArrays:

```python
import xarray as xr
import axis

# Load source and destination datasets
ds_src = xr.open_dataset("gfs_source.nc")
ds_dst = xr.open_dataset("orca_target.nc")

# Remap a DataArray instantly using the accessor
regridded_temp = ds_src.temperature.axis.regrid_to(ds_dst, method="bilinear")
```

### 2. Standard Regridder Object
You can instantiate a persistent `Regridder` object to reuse the same generated weights across multiple fields:

```python
regridder = axis.Regridder(ds_src, ds_dst, method="conservative")

# Apply to multiple variables
u_regridded = regridder(ds_src.u_wind)
v_regridded = regridder(ds_src.v_wind)
```

### 3. Save, Reload, and Reuse Weights
Bypass the weights generation phase entirely by saving compiled C++ weights to a file and reloading them:

```python
# Save weights to a binary file
regridder.to_file("my_weights.bin")

# Reload and reuse instantly in subsequent runs
fast_regridder = axis.Regridder(ds_src, ds_dst, weights_file="my_weights.bin")
result = fast_regridder(ds_src.temperature)
```

### 4. Flawless Dask Distributed Parallelism
When running on backed Dask chunk-wise arrays, AXIS automatically serializes its C++ compiled weights and distributes them across all remote worker nodes in parallel using `client.run`. For local process/thread-wise schedulers, it leverages a thread-safe worker cache to completely bypass Python pickling limits, ensuring zero `PicklingError`s and beautiful, lazy parallel scalability.

---

## 💻 `axis-regrid` Command-Line Tool

AXIS registers a native command-line executable **`axis-regrid`** in your system path, allowing shell scripters and operational pipelines to run Kokkos-parallel, high-performance regridding on NetCDF files with a single command:

```bash
# Run conservative remapping on all spatial variables in a NetCDF file
axis-regrid -s source.nc -t target_grid.nc -o output.nc -m conservative --skipna

# Remap specific variables with periodic wrapping enabled
axis-regrid -s source.nc -t target_grid.nc -o output.nc -m bilinear --periodic -v temperature -v humidity
```

---

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

# Configure (Release build, Python enabled, tests enabled)
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_PYTHON=ON \
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
| `BUILD_PYTHON` | `OFF` | Build the compiled C++ Python nanobind wrapper |

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
- **Python Integration tests** (pytest) — verifies CF coordinate detection, MPAS polygon parsing, Dask chunking, weight saving/reloading, and comparative CDO/ESMF benchmarks.

## License

This project is part of the NOAA-EMC Ecosystem.

See [LICENSE](../../LICENSE) and [DISCLAIMER](../../DISCLAIMER) for details.
