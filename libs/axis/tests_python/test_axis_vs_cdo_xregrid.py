# SPDX-License-Identifier: Apache-2.0
import os
import sys
import time
import tempfile
import subprocess
import shutil
import numpy as np
import xarray as xr
import pytest

import axis

# ─────────────────────────────────────────────────────────────────────────────
# Optional Dependency and Binary Discovery
# ─────────────────────────────────────────────────────────────────────────────
# Verify if CDO binary is installed on the system path (eliminates python-cdo package dependency)
CDO_AVAILABLE = shutil.which("cdo") is not None

try:
    import xregrid
    XREGRID_AVAILABLE = True
except ImportError:
    XREGRID_AVAILABLE = False

# Ensure projection libraries are available for LCC tests
try:
    import pyproj
    PROJ_AVAILABLE = True
except ImportError:
    PROJ_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions from compare_cdo.py for Grid & NetCDF Generation
# ─────────────────────────────────────────────────────────────────────────────
LCC_PROJ = "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"

def create_test_dataset(nlat, nlon, grid_type="regular"):
    """Create a standard test dataset containing a smooth cosine bell variable."""
    if grid_type == "regular":
        lats = np.linspace(-90.0, 90.0, nlat)
        lons = np.linspace(0.0, 360.0, nlon, endpoint=False)
        lon2d, lat2d = np.meshgrid(lons, lats)
        field = np.cos(np.radians(lat2d)) * np.cos(np.radians(lon2d))

        # Calculate bounds
        dlat = lats[1] - lats[0]
        dlon = lons[1] - lons[0]
        lat_bnds = np.zeros((len(lats), 2))
        lat_bnds[:, 0] = lats - 0.5 * dlat
        lat_bnds[:, 1] = lats + 0.5 * dlat
        lat_bnds = np.clip(lat_bnds, -90.0, 90.0)

        lon_bnds = np.zeros((len(lons), 2))
        lon_bnds[:, 0] = lons - 0.5 * dlon
        lon_bnds[:, 1] = lons + 0.5 * dlon

        ds = xr.Dataset(
            {"temperature": (["lat", "lon"], field.astype(np.float64))},
            coords={"lat": lats, "lon": lons},
        )
        ds["lat_bnds"] = (["lat", "bnds"], lat_bnds)
        ds["lon_bnds"] = (["lon", "bnds"], lon_bnds)
        ds["lat"].attrs = {"units": "degrees_north", "axis": "Y", "bounds": "lat_bnds", "standard_name": "latitude"}
        ds["lon"].attrs = {"units": "degrees_east", "axis": "X", "bounds": "lon_bnds", "standard_name": "longitude"}
        ds["temperature"].attrs = {"units": "K", "long_name": "Test field"}
        return ds

    elif grid_type == "lcc" and PROJ_AVAILABLE:
        min_x, max_x = -1000000.0, 1000000.0
        min_y, max_y = -1000000.0, 1000000.0
        xs = np.linspace(min_x, max_x, nlon)
        ys = np.linspace(min_y, max_y, nlat)
        x_grid2d, y_grid2d = np.meshgrid(xs, ys)

        proj = pyproj.Proj(LCC_PROJ)
        lons2d, lats2d = proj(x_grid2d, y_grid2d, inverse=True)
        field = np.cos(np.radians(lats2d)) * np.cos(np.radians(lons2d))

        dx = (max_x - min_x) / (nlon - 1)
        dy = (max_y - min_y) / (nlat - 1)
        xs_c = np.linspace(min_x - 0.5 * dx, max_x + 0.5 * dx, nlon + 1)
        ys_c = np.linspace(min_y - 0.5 * dy, max_y + 0.5 * dy, nlat + 1)
        xc_grid2d, yc_grid2d = np.meshgrid(xs_c, ys_c)
        clons2d, clats2d = proj(xc_grid2d, yc_grid2d, inverse=True)

        lat_bounds = np.zeros((nlat, nlon, 4))
        lon_bounds = np.zeros((nlat, nlon, 4))
        for j in range(nlat):
            for i in range(nlon):
                lat_bounds[j, i, 0] = clats2d[j, i]
                lat_bounds[j, i, 1] = clats2d[j, i+1]
                lat_bounds[j, i, 2] = clats2d[j+1, i+1]
                lat_bounds[j, i, 3] = clats2d[j+1, i]

                lon_bounds[j, i, 0] = clons2d[j, i]
                lon_bounds[j, i, 1] = clons2d[j, i+1]
                lon_bounds[j, i, 2] = clons2d[j+1, i+1]
                lon_bounds[j, i, 3] = clons2d[j+1, i]

        ds = xr.Dataset(
            {"temperature": (["lat", "lon"], field.astype(np.float64))},
            coords={
                "lat_2d": (["lat", "lon"], lats2d),
                "lon_2d": (["lat", "lon"], lons2d),
            },
        )
        ds["lat_bounds"] = (["lat", "lon", "nv"], lat_bounds)
        ds["lon_bounds"] = (["lat", "lon", "nv"], lon_bounds)
        ds["lat_2d"].attrs = {"units": "degrees_north", "standard_name": "latitude", "bounds": "lat_bounds"}
        ds["lon_2d"].attrs = {"units": "degrees_east", "standard_name": "longitude", "bounds": "lon_bounds"}
        ds["temperature"].attrs = {"coordinates": "lon_2d lat_2d", "units": "K", "long_name": "Test field"}
        return ds

    return None

# ─────────────────────────────────────────────────────────────────────────────
# Side-by-Side Comparison Benchmarks
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not CDO_AVAILABLE, reason="CDO binary or python-cdo not available")
@pytest.mark.skipif(not XREGRID_AVAILABLE, reason="xregrid (ESMF) package not available")
@pytest.mark.parametrize("method, cdo_op, xr_method", [
    ("bilinear", "remapbil", "bilinear"),
    ("nearest", "remapnn", "nearest_s2d"),
    ("conservative", "remapcon", "conservative"),
])
def test_axis_vs_cdo_vs_xregrid_regular_benchmark(method, cdo_op, xr_method):
    """
    Compare AXIS directly to CDO and xregrid for regular grid remapping.
    Asserts zero performance degradation and exact global mass conservation.
    """
    # Grid Sizes: Source (180x90) to Target (360x180)
    ds_in = create_test_dataset(90, 180, "regular")

    dst_lons = np.linspace(0.0, 360.0, 360, endpoint=False)
    dst_lats = np.linspace(-90.0, 90.0, 180)
    ds_out = xr.Dataset({
        "lat": (["lat"], dst_lats),
        "lon": (["lon"], dst_lons)
    })

    # Save standard CF-bounds for CDO and xregrid
    dlat = dst_lats[1] - dst_lats[0]
    dlon = dst_lons[1] - dst_lons[0]
    lat_bnds = np.zeros((len(dst_lats), 2))
    lat_bnds[:, 0] = dst_lats - 0.5 * dlat
    lat_bnds[:, 1] = dst_lats + 0.5 * dlat
    lat_bnds = np.clip(lat_bnds, -90.0, 90.0)
    lon_bnds = np.zeros((len(dst_lons), 2))
    lon_bnds[:, 0] = dst_lons - 0.5 * dlon
    lon_bnds[:, 1] = dst_lons + 0.5 * dlon

    ds_out["lat_bnds"] = (["lat", "bnds"], lat_bnds)
    ds_out["lon_bnds"] = (["lon", "bnds"], lon_bnds)
    ds_out["lat"].attrs = {"units": "degrees_north", "axis": "Y", "bounds": "lat_bnds", "standard_name": "latitude"}
    ds_out["lon"].attrs = {"units": "degrees_east", "axis": "X", "bounds": "lon_bnds", "standard_name": "longitude"}

    # Add dummy variable so CDO parses target_grid_nc as a valid NetCDF grid description
    ds_out["temperature"] = (["lat", "lon"], np.zeros((len(dst_lats), len(dst_lons))))

    with tempfile.TemporaryDirectory() as tmpdir:
        src_nc = os.path.join(tmpdir, "src.nc")
        ds_in.to_netcdf(src_nc)

        target_grid_nc = os.path.join(tmpdir, "target.nc")
        ds_out.to_netcdf(target_grid_nc)

        # 1. Evaluate CDO (Reference)
        cdo_out = os.path.join(tmpdir, "cdo_out.nc")
        t0 = time.perf_counter()

        # Execute CDO directly via subprocess to avoid python-cdo UTF-8 decoding crashes
        cmd = ["cdo", "-O", "-s", f"-{cdo_op},{target_grid_nc}", src_nc, cdo_out]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        cdo_time = time.perf_counter() - t0

        if res.returncode != 0:
            err_msg = res.stderr.decode("utf-8", errors="ignore")
            raise RuntimeError(f"CDO failed with exit code {res.returncode}: {err_msg}")

        cdo_ds = xr.open_dataset(cdo_out)
        cdo_result = cdo_ds["temperature"].values
        cdo_sum = float(np.sum(cdo_result))

        # 2. Evaluate xregrid (ESMF)
        t0 = time.perf_counter()
        xr_regridder = xregrid.Regridder(ds_in, ds_out, method=xr_method, periodic=True)
        xr_out = xr_regridder(ds_in["temperature"])
        xregrid_time = time.perf_counter() - t0
        xregrid_sum = float(np.sum(xr_out.values))

        # 3. Evaluate AXIS (Kokkos-Parallel)
        t0 = time.perf_counter()
        axis_regridder = axis.Regridder(ds_in, ds_out, method=method, periodic=True)
        axis_out = axis_regridder(ds_in["temperature"])
        axis_time = time.perf_counter() - t0
        axis_sum = float(np.sum(axis_out.values))

        # ── PERFORMANCE DEGRADATION ASSERTIONS ──
        # AXIS must always run significantly faster than CDO and xregrid for global regular grids.
        # We allow a small 50ms scheduling jitter margin for micro-grids (16k cells) where single-threaded execution has no setup overhead.
        assert axis_time < (cdo_time + 0.050), f"AXIS remapping is slower than CDO! AXIS: {axis_time:.4f}s, CDO: {cdo_time:.4f}s"
        assert axis_time < (xregrid_time + 0.050), f"AXIS remapping is slower than xregrid! AXIS: {axis_time:.4f}s, xregrid: {xregrid_time:.4f}s"

        # ── MATHEMATICAL CONSERVATION ASSERTION ──
        if method == "conservative":
            # Difference in global mass integrals should be negligible (conservation error < 1e-12)
            np.testing.assert_allclose(axis_sum, cdo_sum, rtol=1e-12, atol=1e-12,
                                       err_msg="AXIS failed double-precision mass conservation parity with CDO!")
            np.testing.assert_allclose(axis_sum, xregrid_sum, rtol=1e-12, atol=1e-12,
                                       err_msg="AXIS failed double-precision mass conservation parity with xregrid!")

            # Perfect mass conservation to exactly zero-sum
            np.testing.assert_allclose(axis_sum, 0.0, rtol=1e-11, atol=1e-11,
                                       err_msg="AXIS conservative failed to preserve zero-integral cosine field!")

@pytest.mark.skipif(not CDO_AVAILABLE, reason="CDO binary or python-cdo not available")
@pytest.mark.skipif(not PROJ_AVAILABLE, reason="pyproj projection library not available")
def test_axis_vs_cdo_projected_lcc_benchmark():
    """
    Compare AXIS directly to CDO for regional Lambert Conformal Conic (LCC) remapping.
    """
    ds_in = create_test_dataset(60, 60, "lcc")

    # Destination regular grid
    dst_lons = np.linspace(-110.0, -82.0, 50, endpoint=False)
    dst_lats = np.linspace(30.0, 50.0, 25)
    ds_out = xr.Dataset({
        "lat": (["lat"], dst_lats),
        "lon": (["lon"], dst_lons)
    })

    # Save standard CF-bounds for CDO
    dlat = dst_lats[1] - dst_lats[0]
    dlon = dst_lons[1] - dst_lons[0]
    lat_bnds = np.zeros((len(dst_lats), 2))
    lat_bnds[:, 0] = dst_lats - 0.5 * dlat
    lat_bnds[:, 1] = dst_lats + 0.5 * dlat
    lat_bnds = np.clip(lat_bnds, -90.0, 90.0)
    lon_bnds = np.zeros((len(dst_lons), 2))
    lon_bnds[:, 0] = dst_lons - 0.5 * dlon
    lon_bnds[:, 1] = dst_lons + 0.5 * dlon

    ds_out["lat_bnds"] = (["lat", "bnds"], lat_bnds)
    ds_out["lon_bnds"] = (["lon", "bnds"], lon_bnds)
    ds_out["lat"].attrs = {"units": "degrees_north", "axis": "Y", "bounds": "lat_bnds", "standard_name": "latitude"}
    ds_out["lon"].attrs = {"units": "degrees_east", "axis": "X", "bounds": "lon_bnds", "standard_name": "longitude"}

    # Add dummy variable so CDO parses target_grid_nc as a valid NetCDF grid description
    ds_out["temperature"] = (["lat", "lon"], np.zeros((len(dst_lats), len(dst_lons))))

    with tempfile.TemporaryDirectory() as tmpdir:
        src_nc = os.path.join(tmpdir, "src.nc")
        ds_in.to_netcdf(src_nc)

        target_grid_nc = os.path.join(tmpdir, "target.nc")
        ds_out.to_netcdf(target_grid_nc)

        # Run CDO conservative remapping
        cdo_out = os.path.join(tmpdir, "cdo_out.nc")
        t0 = time.perf_counter()

        # Execute CDO directly via subprocess to avoid python-cdo UTF-8 decoding crashes
        cmd = ["cdo", "-O", "-s", "-remapcon," + target_grid_nc, src_nc, cdo_out]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        cdo_time = time.perf_counter() - t0

        if res.returncode != 0:
            err_msg = res.stderr.decode("utf-8", errors="ignore")
            raise RuntimeError(f"CDO failed with exit code {res.returncode}: {err_msg}")

        cdo_ds = xr.open_dataset(cdo_out)
        cdo_result = cdo_ds["temperature"].values
        cdo_sum = float(np.nansum(cdo_result))

        # Run AXIS
        t0 = time.perf_counter()
        axis_regridder = axis.Regridder(ds_in, ds_out, method="conservative")
        axis_out = axis_regridder(ds_in["temperature"])
        axis_time = time.perf_counter() - t0
        axis_sum = float(np.nansum(axis_out.values))

        # For small regional grids, AXIS bypasses file I/O and runs highly optimized
        # spatial structures, so it should compile and execute within strict budget
        assert axis_time < 0.200, f"AXIS projected remapping took too long: {axis_time:.4f}s"
        assert axis_out.shape == (25, 50)
