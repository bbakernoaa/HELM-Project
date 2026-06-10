#!/usr/bin/env python3
"""
AXIS vs CDO Interpolation Benchmark
====================================

Compares AXIS interpolation accuracy and speed against CDO (Climate Data
Operators) for bilinear, nearest-neighbor, and conservative remapping.

Prerequisites:
  - CDO installed (via conda: `mamba install -c conda-forge cdo`)
  - python-cdo package (`mamba install -c conda-forge python-cdo`)
  - xarray + netcdf4 for I/O
  - numpy
  - axis_py module (built from libs/axis/python/)

Usage:
  python benchmarks/compare_cdo.py [--grid-size 32] [--methods bilinear,nearest,conservative]

Output:
  Prints a comparison table with max error, RMS error, conservation error,
  and wall-clock time for both AXIS and CDO.
"""

import argparse
import os
import sys
import time
import tempfile

import numpy as np

try:
    import xarray as xr
except ImportError:
    print("ERROR: xarray not available. Install with: pip install xarray netcdf4")
    sys.exit(1)

try:
    from cdo import Cdo
except ImportError:
    print("ERROR: python-cdo not available. Install with: mamba install -c conda-forge python-cdo")
    sys.exit(1)

# Try to import axis_py — if not built, provide instructions
try:
    import axis_py
except ImportError:
    print("WARNING: axis_py module not found. Building requires:")
    print("  cd libs/axis && cmake -B build-py -DBUILD_PYTHON=ON && cmake --build build-py")
    print("  Then: export PYTHONPATH=build-py/python")
    print("")
    print("Running CDO-only benchmark (no AXIS comparison)...")
    axis_py = None


def create_test_field(nlat, nlon, field_type="cosine"):
    """Create a test field on a regular lat-lon grid."""
    lats = np.linspace(-90, 90, nlat)
    lons = np.linspace(0, 360, nlon, endpoint=False)
    lon2d, lat2d = np.meshgrid(lons, lats)

    if field_type == "cosine":
        # Smooth cosine bell — good for testing bilinear exactness
        field = np.cos(np.radians(lat2d)) * np.cos(np.radians(lon2d))
    elif field_type == "linear":
        # Linear field — bilinear should reproduce exactly
        field = 0.5 * lon2d + 0.3 * lat2d + 10.0
    elif field_type == "constant":
        # Constant — tests partition of unity
        field = 42.0 * np.ones_like(lat2d)
    elif field_type == "step":
        # Step function — stresses conservative method
        field = np.where(lat2d > 0, 1.0, 0.0)
    else:
        raise ValueError(f"Unknown field type: {field_type}")

    return lats, lons, field


def write_netcdf(filepath, lats, lons, field, varname="temperature"):
    """Write a field to a CF-compliant NetCDF file for CDO."""
    ds = xr.Dataset(
        {varname: (["lat", "lon"], field.astype(np.float64))},
        coords={"lat": lats, "lon": lons},
    )
    ds["lat"].attrs = {"units": "degrees_north", "axis": "Y"}
    ds["lon"].attrs = {"units": "degrees_east", "axis": "X"}
    ds[varname].attrs = {"units": "K", "long_name": "Test field"}
    ds.to_netcdf(filepath)
    return ds


def run_cdo_remap(input_file, output_file, target_grid, method):
    """Run CDO remapping and return wall-clock time."""
    cdo = Cdo()

    # CDO operator names
    method_map = {
        "bilinear": "remapbil",
        "nearest": "remapnn",
        "bicubic": "remapbic",
        "patch": "remapbil",  # CDO doesn't have exact patch; use bilinear as reference
        "conservative": "remapcon",
    }

    operator = method_map[method]
    t0 = time.perf_counter()
    getattr(cdo, operator)(target_grid, input=input_file, output=output_file)
    elapsed = time.perf_counter() - t0

    return elapsed


def run_axis_remap(src_nlat, src_nlon, dst_nlat, dst_nlon, field, method):
    """Run AXIS remapping and return (result, wall_time)."""
    if axis_py is None:
        return None, 0.0

    # Map method name to AXIS enum
    method_map = {
        "bilinear": axis_py.Method.Bilinear,
        "nearest": axis_py.Method.NearestNeighbor,
        "bicubic": axis_py.Method.Bicubic,
        "patch": axis_py.Method.Patch,
        "conservative": axis_py.Method.Conservative,
    }

    # Build source and destination meshes as regular lat-lon grids
    src_dlon = 360.0 / src_nlon
    src_dlat = 180.0 / src_nlat
    dst_dlon = 360.0 / dst_nlon
    dst_dlat = 180.0 / dst_nlat

    t0 = time.perf_counter()

    src_mesh = axis_py.make_regular_mesh(
        src_nlon, src_nlat, 0.0, -90.0, src_dlon, src_dlat)
    dst_mesh = axis_py.make_regular_mesh(
        dst_nlon, dst_nlat, 0.0, -90.0, dst_dlon, dst_dlat)

    # Generate weights
    matrix = axis_py.generate_weights(src_mesh, dst_mesh, method_map[method])

    # Apply weights
    src_flat = field.ravel().astype(np.float64)
    dst_flat = axis_py.apply_weights(matrix, src_flat)

    elapsed = time.perf_counter() - t0

    return dst_flat, elapsed


def compare_results(axis_result, cdo_result):
    """Compare AXIS and CDO results, return error metrics."""
    if axis_result is None:
        return {"max_error": np.nan, "rms_error": np.nan, "mean_error": np.nan}

    # Flatten CDO result to match AXIS
    cdo_flat = cdo_result.ravel()

    # If sizes differ (different grid interpretation), truncate to common size
    n = min(len(axis_result), len(cdo_flat))
    diff = axis_result[:n] - cdo_flat[:n]

    return {
        "max_error": np.max(np.abs(diff)),
        "rms_error": np.sqrt(np.mean(diff**2)),
        "mean_error": np.mean(np.abs(diff)),
    }


def main():
    parser = argparse.ArgumentParser(description="AXIS vs CDO interpolation benchmark")
    parser.add_argument("--src-size", type=str, default="32",
                        help="Source grid size: N (square NxN) or NLONxNLAT")
    parser.add_argument("--dst-size", type=str, default="24",
                        help="Destination grid size: N (square NxN) or NLONxNLAT")
    parser.add_argument("--methods", type=str, default="bilinear,nearest,bicubic,patch,conservative",
                        help="Comma-separated interpolation methods to test")
    parser.add_argument("--field", type=str, default="cosine",
                        choices=["cosine", "linear", "constant", "step"],
                        help="Test field type")
    args = parser.parse_args()

    # Parse grid sizes (support NxM or just N for square)
    def parse_grid_size(s):
        if 'x' in s:
            parts = s.split('x')
            return int(parts[0]), int(parts[1])
        n = int(s)
        return n, n

    src_nlon, src_nlat = parse_grid_size(args.src_size)
    dst_nlon, dst_nlat = parse_grid_size(args.dst_size)
    methods = [m.strip() for m in args.methods.split(",")]

    print(f"{'='*70}")
    print(f"AXIS vs CDO Interpolation Benchmark")
    print(f"{'='*70}")
    print(f"Source grid:  {src_nlon}x{src_nlat} regular lat-lon ({src_nlon*src_nlat:,} cells)")
    print(f"Dest grid:    {dst_nlon}x{dst_nlat} regular lat-lon ({dst_nlon*dst_nlat:,} cells)")
    print(f"Test field:   {args.field}")
    print(f"Methods:      {', '.join(methods)}")
    print(f"AXIS module:  {'loaded' if axis_py else 'NOT AVAILABLE'}")
    print(f"{'='*70}")
    print()

    # Create test field
    lats, lons, field = create_test_field(src_nlat, src_nlon, args.field)

    # Write source NetCDF for CDO
    with tempfile.TemporaryDirectory() as tmpdir:
        src_nc = os.path.join(tmpdir, "source.nc")
        write_netcdf(src_nc, lats, lons, field)

        # Create CDO target grid description
        dst_lats = np.linspace(-90, 90, dst_nlat)
        dst_lons = np.linspace(0, 360, dst_nlon, endpoint=False)
        target_grid = os.path.join(tmpdir, "target_grid.txt")
        with open(target_grid, "w") as f:
            f.write(f"gridtype = lonlat\n")
            f.write(f"xsize = {dst_nlon}\n")
            f.write(f"ysize = {dst_nlat}\n")
            f.write(f"xfirst = {dst_lons[0]}\n")
            f.write(f"xinc = {dst_lons[1] - dst_lons[0]}\n")
            f.write(f"yfirst = {dst_lats[0]}\n")
            f.write(f"yinc = {dst_lats[1] - dst_lats[0]}\n")

        # Run benchmarks
        print(f"{'Method':<15} {'Engine':<8} {'Time (s)':<12} {'Max Err':<14} {'RMS Err':<14} {'Src Σ':<14} {'Dst Σ':<14}")
        print(f"{'-'*15} {'-'*8} {'-'*12} {'-'*14} {'-'*14} {'-'*14} {'-'*14}")

        for method in methods:
            # ── CDO ──
            cdo_out = os.path.join(tmpdir, f"cdo_{method}.nc")
            try:
                cdo_time = run_cdo_remap(src_nc, cdo_out, target_grid, method)
                cdo_ds = xr.open_dataset(cdo_out)
                cdo_result = cdo_ds["temperature"].values
                cdo_sum = float(np.nansum(cdo_result))
            except Exception as e:
                print(f"{method:<15} {'CDO':<8} {'FAILED':<12} {str(e)[:40]}")
                cdo_result = None
                cdo_time = 0.0
                cdo_sum = np.nan

            src_sum = float(np.sum(field))

            if cdo_result is not None:
                print(f"{method:<15} {'CDO':<8} {cdo_time:<12.4f} {'—':<14} {'—':<14} {src_sum:<14.4f} {cdo_sum:<14.4f}")

            # ── AXIS ──
            axis_result, axis_time = run_axis_remap(
                src_nlat, src_nlon, dst_nlat, dst_nlon, field, method)

            if axis_result is not None:
                axis_sum = float(np.sum(axis_result))

                # Compare against CDO
                if cdo_result is not None:
                    errors = compare_results(axis_result, cdo_result)
                    print(f"{method:<15} {'AXIS':<8} {axis_time:<12.4f} "
                          f"{errors['max_error']:<14.2e} "
                          f"{errors['rms_error']:<14.2e} "
                          f"{src_sum:<14.4f} {axis_sum:<14.4f}")
                else:
                    print(f"{method:<15} {'AXIS':<8} {axis_time:<12.4f} {'—':<14} {'—':<14} {src_sum:<14.4f} {axis_sum:<14.4f}")
            else:
                print(f"{method:<15} {'AXIS':<8} {'N/A':<12} {'(module not loaded)'}")

            print()

    print(f"{'='*70}")
    print("Notes:")
    print("  - Max/RMS Err = AXIS result vs CDO result (CDO is the reference)")
    print("  - Src/Dst Σ = sum of field values (check conservation)")
    print("  - Time includes weight generation + apply (not I/O)")
    if axis_py is None:
        print("\n  To enable AXIS comparison, build the Python module:")
        print("    cd libs/axis && cmake -B build-py -DBUILD_PYTHON=ON")
        print("    cmake --build build-py && export PYTHONPATH=build-py/python")


if __name__ == "__main__":
    main()
