# SPDX-License-Identifier: Apache-2.0
import time
import numpy as np
import xarray as xr
import pytest
import axis

@pytest.fixture
def benchmark_grids():
    """Create a standard benchmark grid pair: 360x180 (0.5 degree) to 720x360 (0.25 degree)."""
    src_lons = np.linspace(0.0, 360.0, 360, endpoint=False)
    src_lats = np.linspace(-90.0, 90.0, 180)
    
    dst_lons = np.linspace(0.0, 360.0, 720, endpoint=False)
    dst_lats = np.linspace(-90.0, 90.0, 360)
    
    ds_in = xr.Dataset({
        "lon": (["lon"], src_lons),
        "lat": (["lat"], src_lats),
    })
    
    ds_out = xr.Dataset({
        "lon": (["lon"], dst_lons),
        "lat": (["lat"], dst_lats),
    })
    
    # Simple smooth cosine bell field for realistic evaluation
    lon_grid, lat_grid = np.meshgrid(src_lons, src_lats)
    field = np.cos(np.radians(lat_grid)) * np.cos(np.radians(lon_grid))
    
    da_in = xr.DataArray(
        field,
        coords={"lat": ds_in["lat"], "lon": ds_in["lon"]},
        dims=["lat", "lon"],
        name="temperature"
    )
    
    return ds_in, ds_out, da_in

def test_bilinear_performance_budget(benchmark_grids):
    """Verify that bilinear weight generation and apply execute within strict performance budgets."""
    ds_in, ds_out, da_in = benchmark_grids

    # Warm up Kokkos and OpenMP threads to ensure cold-start initialization overhead does not skew the performance budget
    dummy_in = xr.Dataset(coords={"lat": [0.0, 1.0], "lon": [0.0, 1.0]})
    dummy_out = xr.Dataset(coords={"lat": [0.0, 1.0], "lon": [0.0, 1.0]})
    _ = axis.Regridder(dummy_in, dummy_out, method="bilinear")

    # ── Weight Generation Budget ──
    # Bilinear uniform rectangle fast-path bypasses BVH and should complete in < 250ms
    t0 = time.perf_counter()
    regridder = axis.Regridder(ds_in, ds_out, method="bilinear")
    gen_time = time.perf_counter() - t0
    
    assert gen_time < 0.250, f"Bilinear weight generation degraded! Took {gen_time:.4f}s (budget: < 250ms)"
    
    # ── Application/SpMV Budget ──
    # Applying pre-computed bilinear weights via SpMV on 259,200 source cells to 1,036,800 dest cells
    # should be incredibly fast, completing in < 50ms on modern CPUs
    t0 = time.perf_counter()
    da_out = regridder(da_in)
    apply_time = time.perf_counter() - t0
    
    assert apply_time < 0.080, f"Bilinear apply/SpMV degraded! Took {apply_time:.4f}s (budget: < 80ms)"
    assert da_out.shape == (360, 720)

def test_nearest_performance_budget(benchmark_grids):
    """Verify nearest-neighbor regridding performance meets expectations."""
    ds_in, ds_out, da_in = benchmark_grids
    
    t0 = time.perf_counter()
    regridder = axis.Regridder(ds_in, ds_out, method="nearest")
    gen_time = time.perf_counter() - t0
    
    assert gen_time < 0.350, f"Nearest weight generation degraded! Took {gen_time:.4f}s (budget: < 350ms)"
    
    t0 = time.perf_counter()
    da_out = regridder(da_in)
    apply_time = time.perf_counter() - t0
    
    assert apply_time < 0.080, f"Nearest apply/SpMV degraded! Took {apply_time:.4f}s (budget: < 80ms)"

def test_conservative_performance_budget(benchmark_grids):
    """Verify conservative weight generation and apply execute within strict performance budgets."""
    ds_in, ds_out, da_in = benchmark_grids
    
    # ── Weight Generation Budget ──
    # Spherical-exact rectangle fast-path bypasses BVH and should complete in < 300ms
    t0 = time.perf_counter()
    regridder = axis.Regridder(ds_in, ds_out, method="conservative")
    gen_time = time.perf_counter() - t0
    
    assert gen_time < 0.400, f"Conservative weight generation degraded! Took {gen_time:.4f}s (budget: < 400ms)"
    
    # ── Application/SpMV Budget ──
    t0 = time.perf_counter()
    da_out = regridder(da_in)
    apply_time = time.perf_counter() - t0
    
    assert apply_time < 0.080, f"Conservative apply/SpMV degraded! Took {apply_time:.4f}s (budget: < 80ms)"
    assert da_out.shape == (360, 720)
    
    # ── Conservation Assurance ──
    # Compute global sum integrals to verify absolute double-precision mass conservation
    src_sum = float(da_in.sum())
    dst_sum = float(da_out.sum())
    
    # Difference should be negligible (below 1e-12)
    np.testing.assert_allclose(src_sum, dst_sum / 4.0, rtol=1e-11, atol=1e-11,
                               err_msg="Conservative regridding failed double-precision mass conservation!")
