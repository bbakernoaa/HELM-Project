# SPDX-License-Identifier: Apache-2.0
"""
Pytest configuration for AXIS Python API integration tests.

Ensures Kokkos is initialized before any test that imports axis_py.
Kokkos initialization is handled internally by the axis_py module
(via ensure_kokkos()), so this conftest primarily provides shared
fixtures for mesh construction used across multiple tests.
"""

import pytest
import numpy as np


@pytest.fixture
def small_src_mesh():
    """Create a small 8x4 regular lat-lon source mesh."""
    import axis_py

    ni, nj = 8, 4
    lon_start, lat_start = 0.0, -90.0
    dlon = 360.0 / ni
    dlat = 180.0 / nj
    return axis_py.make_regular_mesh(ni, nj, lon_start, lat_start, dlon, dlat)


@pytest.fixture
def small_dst_mesh():
    """Create a small 4x2 regular lat-lon destination mesh."""
    import axis_py

    ni, nj = 4, 2
    lon_start, lat_start = 0.0, -90.0
    dlon = 360.0 / ni
    dlat = 180.0 / nj
    return axis_py.make_regular_mesh(ni, nj, lon_start, lat_start, dlon, dlat)


@pytest.fixture
def bilinear_matrix(small_src_mesh, small_dst_mesh):
    """Generate bilinear interpolation weights between src and dst meshes."""
    import axis_py

    return axis_py.generate_weights(
        small_src_mesh, small_dst_mesh, axis_py.Method.Bilinear
    )
