# SPDX-License-Identifier: Apache-2.0
import numpy as np
import xarray as xr
import pytest

# Ensure the dask package is optionally importable for lazy tests
try:
    import dask.array as da
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

import axis

@pytest.fixture
def sample_grids():
    """Create simple source (10x5) and destination (20x10) xarray regular grids."""
    src_lons = np.linspace(0.0, 360.0, 10, endpoint=False)
    src_lats = np.linspace(-90.0, 90.0, 5)
    
    dst_lons = np.linspace(0.0, 360.0, 20, endpoint=False)
    dst_lats = np.linspace(-90.0, 90.0, 10)
    
    ds_in = xr.Dataset({
        "lon": (["lon"], src_lons),
        "lat": (["lat"], src_lats),
    })
    
    ds_out = xr.Dataset({
        "lon": (["lon"], dst_lons),
        "lat": (["lat"], dst_lats),
    })
    
    return ds_in, ds_out

def test_regridder_eager_numpy(sample_grids):
    """Verify that axis.Regridder works on eager NumPy-backed DataArrays."""
    ds_in, ds_out = sample_grids
    
    # Create simple cosine field: cos(lat) * cos(lon)
    lon_grid, lat_grid = np.meshgrid(ds_in["lon"], ds_in["lat"])
    field = np.cos(np.radians(lat_grid)) * np.cos(np.radians(lon_grid))
    
    da_in = xr.DataArray(
        field,
        coords={"lat": ds_in["lat"], "lon": ds_in["lon"]},
        dims=["lat", "lon"],
        name="temperature"
    )
    
    # Initialize and call regridder
    regridder = axis.Regridder(ds_in, ds_out, method="bilinear")
    da_out = regridder(da_in)
    
    assert da_out.dims == ("lat", "lon")
    assert da_out.shape == (10, 20)
    assert da_out.name == "temperature"
    assert "lat" in da_out.coords
    assert "lon" in da_out.coords
    
    # Verify constant field is preserved (partition of unity)
    const_da = xr.DataArray(
        np.full(da_in.shape, 5.0),
        coords=da_in.coords,
        dims=da_in.dims
    )
    const_out = regridder(const_da)
    np.testing.assert_allclose(const_out.values, 5.0, rtol=1e-12)

@pytest.mark.skipif(not DASK_AVAILABLE, reason="Dask array not available")
def test_regridder_lazy_dask(sample_grids):
    """Verify that axis.Regridder works on lazy Dask-backed DataArrays."""
    ds_in, ds_out = sample_grids
    
    # Create a lazy Dask-backed DataArray with 2 time steps
    lazy_data = da.ones((2, 5, 10), chunks=(1, 5, 10))
    da_in = xr.DataArray(
        lazy_data,
        coords={"time": [0, 1], "lat": ds_in["lat"], "lon": ds_in["lon"]},
        dims=["time", "lat", "lon"],
        name="precipitation"
    )
    
    regridder = axis.Regridder(ds_in, ds_out, method="bilinear")
    da_out = regridder(da_in)
    
    # Verify that the output array is ALSO a lazy Dask array (has not been eagerly computed)
    assert hasattr(da_out.data, "dask")
    assert da_out.dims == ("time", "lat", "lon")
    assert da_out.shape == (2, 10, 20)
    
    # Eagerly compute the lazy array and verify values
    computed = da_out.compute()
    assert isinstance(computed.data, np.ndarray)
    np.testing.assert_allclose(computed.values, 1.0, rtol=1e-12)

def test_regridder_accessor(sample_grids):
    """Verify that xarray datasets and dataarrays can be regridded via the .regrid accessor."""
    ds_in, ds_out = sample_grids
    
    da_in = xr.DataArray(
        np.ones((5, 10)),
        coords={"lat": ds_in["lat"], "lon": ds_in["lon"]},
        dims=["lat", "lon"],
        name="temperature"
    )
    
    # 1. Test DataArray accessor: da.regrid.to(ds_out)
    da_out = da_in.regrid.to(ds_out, method="bilinear")
    assert da_out.shape == (10, 20)
    assert da_out.name == "temperature"
    np.testing.assert_allclose(da_out.values, 1.0, rtol=1e-12)
    
    # 2. Test Dataset accessor: ds.regrid.to(ds_out)
    ds_in_vars = xr.Dataset({"temperature": da_in})
    ds_out_vars = ds_in_vars.regrid.to(ds_out, method="bilinear")
    assert "temperature" in ds_out_vars.data_vars
    assert ds_out_vars["temperature"].shape == (10, 20)
    np.testing.assert_allclose(ds_out_vars["temperature"].values, 1.0, rtol=1e-12)
