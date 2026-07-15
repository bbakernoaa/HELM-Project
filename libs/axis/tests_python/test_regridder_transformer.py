# test_regridder_transformer.py
#
# Verifies standard Scikit-Learn transformer API (fit/transform) for AXIS.

import numpy as np
import pytest
import xarray as xr
from axis import RectilinearGrid, Regridder


@pytest.fixture
def sample_data():
    # 10x10 input regular lat-lon dataset
    lons = np.linspace(-180, 180, 10)
    lats = np.linspace(-90, 90, 10)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    da = xr.DataArray(
        np.sin(np.radians(lon_grid)) * np.cos(np.radians(lat_grid)),
        coords={"lat": lats, "lon": lons},
        dims=["lat", "lon"],
    )
    ds = da.to_dataset(name="sst")
    return ds, da


def test_sklearn_transformer_pipeline(sample_data):
    ds_in, da_in = sample_data

    # Target 5x5 grid
    lons_out = np.linspace(-180, 180, 5)
    lats_out = np.linspace(-90, 90, 5)
    ds_out = xr.Dataset(coords={"lat": lats_out, "lon": lons_out})

    # 1. Instantiate-then-fit (Sklearn style)
    regridder = Regridder(method="bilinear")
    regridder.fit(ds_in, ds_out)

    res_transformer = regridder.transform(da_in)

    # 2. Classic construction (backward compatible style)
    classic_regridder = Regridder(ds_in, ds_out, method="bilinear")
    res_classic = classic_regridder(da_in)

    # Verify results are identical within machine float precision
    xr.testing.assert_allclose(res_transformer, res_classic)


def test_fit_with_explicit_geometries(sample_data):
    ds_in, da_in = sample_data

    # Use RectilinearGrid directly
    lons_out = np.linspace(-180, 180, 5)
    lats_out = np.linspace(-90, 90, 5)
    dst_geom = RectilinearGrid(lons_out, lats_out)

    regridder = Regridder(method="bilinear")
    regridder.fit(ds_in, dst_geom)

    res = regridder.transform(da_in)
    assert res.shape == (5, 5)
    assert "lat" in res.coords
    assert "lon" in res.coords
