#!/usr/bin/env python3
"""
Example: Complete AXIS regridding workflow in Python.

Demonstrates:
1. Named grid generation
2. Weight generation (bilinear and conservative)
3. Single-field apply
4. Batch apply for multiple variables
5. Weight caching (serialize/deserialize)

Requires: axis Python module (built with -DAXIS_BUILD_PYTHON=ON)
"""

import numpy as np


def main():
    import axis

    # =========================================================================
    # 1. Create source and destination grids
    # =========================================================================
    print("Creating grids...")
    src_mesh = axis.NamedGridRegistry.generate("O48")
    dst_mesh = axis.NamedGridRegistry.generate("O96")
    print(f"  Source: {src_mesh.n_cells} cells")
    print(f"  Destination: {dst_mesh.n_cells} cells")

    # =========================================================================
    # 2. Generate bilinear interpolation weights
    # =========================================================================
    print("\nGenerating bilinear weights...")
    config = {"method": "bilinear"}
    matrix = axis.WeightGenerator.generate(src_mesh, dst_mesh, config)
    print(f"  Matrix nnz: {matrix.nnz}")

    # =========================================================================
    # 3. Apply to a constant field (partition of unity test)
    # =========================================================================
    print("\nTesting partition of unity...")
    src_const = np.ones(src_mesh.n_cells, dtype=np.float64)
    dst_const = axis.apply(matrix, src_const)
    max_err = np.max(np.abs(dst_const - 1.0))
    print(f"  Max error from 1.0: {max_err:.2e}")
    assert max_err < 1e-12, "Partition of unity violated!"

    # =========================================================================
    # 4. Batch apply for multiple variables
    # =========================================================================
    print("\nBatch apply (5 variables)...")
    n_vars = 5
    src_state = np.random.default_rng(42).standard_normal(
        (src_mesh.n_cells, n_vars)
    )
    # Use Fortran order for zero-copy
    src_state = np.asfortranarray(src_state)

    dst_state = axis.batch_apply(matrix, src_state)
    print(f"  Input shape:  {src_state.shape}")
    print(f"  Output shape: {dst_state.shape}")
    assert dst_state.shape == (dst_mesh.n_cells, n_vars)

    # Verify batch matches individual
    for v in range(n_vars):
        dst_v = axis.apply(matrix, src_state[:, v])
        err = np.max(np.abs(dst_state[:, v] - dst_v))
        assert err < 1e-14, f"Batch/individual mismatch for var {v}: {err}"
    print("  Batch matches individual apply: OK")

    # =========================================================================
    # 5. Weight caching round-trip
    # =========================================================================
    print("\nWeight caching...")
    blob = matrix.to_bytes()
    print(f"  Serialized: {len(blob)} bytes ({len(blob) / 1024:.1f} KB)")

    matrix2 = axis.InterpolationMatrix.from_bytes(blob)
    dst_cached = axis.apply(matrix2, src_const)
    assert np.array_equal(dst_const, dst_cached), "Cache round-trip failed!"
    print("  Round-trip verified: bitwise identical")

    # =========================================================================
    # 6. Conservative remapping
    # =========================================================================
    print("\nConservative remapping...")
    config_cons = {"method": "conservative_1st", "normalization": "dst_area"}
    matrix_cons = axis.WeightGenerator.generate(src_mesh, dst_mesh, config_cons)

    src_field = np.random.default_rng(123).standard_normal(src_mesh.n_cells)
    dst_field = axis.apply(matrix_cons, src_field)

    # Note: full conservation check requires area arrays from mesh
    print(f"  Source mean: {np.mean(src_field):.6f}")
    print(f"  Dest mean:   {np.mean(dst_field):.6f}")

    print("\nAll examples passed successfully!")


if __name__ == "__main__":
    main()
