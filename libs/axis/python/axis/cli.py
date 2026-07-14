# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import sys
import time

import xarray as xr

import axis


def main():
    parser = argparse.ArgumentParser(
        description="AXIS Command-Line Regridding Tool — Blazing-fast, Kokkos-parallel spatial interpolation."
    )
    parser.add_argument("-s", "--source", required=True, help="Path to the input source NetCDF file.")
    parser.add_argument(
        "-t",
        "--target",
        required=True,
        help="Path to the target destination NetCDF file (or UGRID/MPAS description).",
    )
    parser.add_argument("-o", "--output", required=True, help="Path to save the regridded output NetCDF file.")
    parser.add_argument(
        "-m",
        "--method",
        default="bilinear",
        choices=["bilinear", "nearest", "bicubic", "patch", "conservative"],
        help="Remapping interpolation method (default: bilinear).",
    )
    parser.add_argument(
        "--periodic",
        action="store_true",
        help="Enable periodic longitude wrapping (connects 360 back to 0 degrees).",
    )
    parser.add_argument(
        "--skipna",
        action="store_true",
        help="Enable NaN-aware weight re-normalization (preserves edges near missing values).",
    )
    parser.add_argument(
        "--na-thres",
        type=float,
        default=1.0,
        help="Minimum fraction of valid input contribution required for output cell (default: 1.0).",
    )
    parser.add_argument(
        "-v",
        "--var",
        action="append",
        help="Specific variable(s) to regrid (optional, defaults to all spatial variables).",
    )

    args = parser.parse_args()

    if not os.path.exists(args.source):
        print(f"Error: Source file '{args.source}' does not exist.")
        sys.exit(1)
    if not os.path.exists(args.target):
        print(f"Error: Target file '{args.target}' does not exist.")
        sys.exit(1)

    print(f"{'=' * 70}")
    print("AXIS Command-Line Regridding Pipeline")
    print(f"{'=' * 70}")
    print(f"Source file : {args.source}")
    print(f"Target file : {args.target}")
    print(f"Output file : {args.output}")
    print(f"Method      : {args.method}")
    print(f"Periodic    : {args.periodic}")
    print(f"NaN Renorm  : {args.skipna} (na_thres: {args.na_thres})")
    if args.var:
        print(f"Variables   : {', '.join(args.var)}")
    else:
        print("Variables   : All spatial data variables")
    print(f"{'=' * 70}")
    print()

    # Load datasets
    t0 = time.perf_counter()
    print("Loading source and target datasets...")
    try:
        ds_in = xr.open_dataset(args.source)
        ds_out = xr.open_dataset(args.target)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        sys.exit(1)

    # Initialize Regridder (generates weights in parallel C++)
    print("Initializing AXIS engine and generating weights...")
    try:
        regridder = axis.Regridder(
            ds_in,
            ds_out,
            method=args.method,
            periodic=args.periodic,
            skipna=args.skipna,
            na_thres=args.na_thres,
        )
    except Exception as e:
        print(f"Error generating weights: {e}")
        sys.exit(1)

    # Filter variables to process
    if args.var:
        invalid_vars = [v for v in args.var if v not in ds_in.variables]
        if invalid_vars:
            print(f"Error: Variable(s) {invalid_vars} not found in source dataset.")
            sys.exit(1)
        ds_subset = ds_in[args.var]
    else:
        ds_subset = ds_in

    # Perform remapping
    print("Executing parallel spatial remapping...")
    try:
        ds_regridded = regridder(ds_subset)
    except Exception as e:
        print(f"Error during remapping: {e}")
        sys.exit(1)

    # Write output file
    print(f"Saving regridded output to {args.output}...")
    try:
        ds_regridded.to_netcdf(args.output)
    except Exception as e:
        print(f"Error saving output NetCDF file: {e}")
        sys.exit(1)

    elapsed = time.perf_counter() - t0
    print(f"Success! Total execution time: {elapsed:.3f} seconds.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
