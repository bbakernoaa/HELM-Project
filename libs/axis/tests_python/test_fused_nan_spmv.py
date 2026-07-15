import pytest
import numpy as np
from axis import axis_py

def test_fused_nan_batch_apply():
    # Construct a simple regular mesh
    src = axis_py.make_regular_mesh(4, 4, 0.0, -90.0, 90.0, 45.0)
    dst = axis_py.make_regular_mesh(2, 2, 0.0, -90.0, 180.0, 90.0)

    config = {"method": "bilinear", "unmapped": "ignore"}
    matrix = axis_py.generate_weights(src, dst, config)
    matrix.to_csr()

    # Create source data with a NaN value
    src_data = np.ones((matrix.n_src, 2), dtype=np.float64)
    src_data[0, 0] = np.nan  # Inject NaN in variable 0

    # Ensure python binding works
    res = axis_py.nan_batch_apply(matrix, src_data, 1.0)
    assert res.shape == (matrix.n_dst, 2)
