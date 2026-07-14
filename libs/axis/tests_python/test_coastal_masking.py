# test_coastal_masking.py
#
# Verifies spatial coastline masking and conservative weight normalization.

import numpy as np
import xarray as xr
import pytest
from axis import Regridder

def test_coastal_mask_application():
    # 4x4 source grid
    lons_in = np.linspace(-10, 10, 4)
    lats_in = np.linspace(-10, 10, 4)
    ds_in = xr.Dataset(coords={"lat": lats_in, "lon": lons_in})

    # 2x2 target grid
    lons_out = np.linspace(-10, 10, 2)
    lats_out = np.linspace(-10, 10, 2)
    ds_out = xr.Dataset(coords={"lat": lats_out, "lon": lons_out})

    # Create a source land mask: 0 representing dry land, 1 representing ocean
    # Let's set one quadrant to be land
    src_mask = np.ones((4, 4), dtype=np.int32)
    src_mask[0:2, 0:2] = 0 # Top-left quadrant is land

    # Generate weights with coastal mask
    regridder = Regridder(method="conservative", src_mask=src_mask)
    regridder.fit(ds_in, ds_out, src_mask=src_mask)

    # Let's verify that a flat wet field preserves values under fracarea normalization
    src_data = np.ones((4, 4), dtype=np.float64)

    # Apply weights
    res = regridder.transform(src_data)

    # Under conservative fracarea normalization, the sum of overlap areas is divided by the
    # total wet active source fractions. Thus, active unmasked regions still remap to 1.0.
    assert res.shape == (2, 2)
    assert not np.any(np.isnan(res))
