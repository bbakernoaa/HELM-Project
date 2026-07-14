# test_vector_regridder.py
#
# Verifies coupled wind vector interpolation and local grid-frame rotation.

import numpy as np
from axis import VectorRegridder, RectilinearGrid


def test_vector_regridder_numerical_precision():
    # Source grid lons/lats well within bounds
    lons_in = np.linspace(-170, 170, 4)
    lats_in = np.linspace(-80, 80, 4)

    # Target grid lons/lats nested inside source domain
    lons_out = np.linspace(-90, 90, 2)
    lats_out = np.linspace(-45, 45, 2)

    # Generate flat eastward and northward wind components
    u_in = np.ones((4, 4), dtype=np.float64) * 10.0
    v_in = np.ones((4, 4), dtype=np.float64) * 5.0

    # Instantiate coupled vector regridder (with zero grid rotation angles)
    regridder = VectorRegridder(
        RectilinearGrid(lons_in, lats_in),
        RectilinearGrid(lons_out, lats_out),
        method="bilinear",
    )

    u_out, v_out = regridder.transform(u_in, v_in)

    assert u_out.shape == (2, 2)
    assert v_out.shape == (2, 2)
    np.testing.assert_allclose(u_out, 10.0, rtol=1e-12)
    np.testing.assert_allclose(v_out, 5.0, rtol=1e-12)
