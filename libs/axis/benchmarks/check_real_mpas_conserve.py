#!/usr/bin/env python3
"""Real MPAS x1.2562: AXIS vs CDO vs xregrid conservation with a CONSTANT field.

cos(lat)cos(lon) integrates to ~0 over the globe, which masks conservation
errors. A constant field makes the mass integral an absolute test.
"""

import os
import sys
import tempfile

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import axis  # noqa: E402
from compare_cdo import create_test_field, write_netcdf  # noqa: E402
from fetch_mpas import fetch_mpas_grid  # noqa: E402

# Downloaded + cached under libs/axis/data/mpas/ on first run.
MPAS = str(fetch_mpas_grid("x1.2562"))
ds = xr.open_dataset(MPAS)
n = ds.sizes["nCells"]
voc = ds["verticesOnCell"].values
nec = ds["nEdgesOnCell"].values
lon_v = np.degrees(ds["lonVertex"].values) % 360.0
lat_v = np.degrees(ds["latVertex"].values)
lat_c = np.degrees(ds["latCell"].values)
lon_c = np.degrees(ds["lonCell"].values) % 360.0
maxnv = int(nec.max())
lb = np.zeros((n, maxnv))
ln = np.zeros((n, maxnv))
for c in range(n):
    m = int(nec[c])
    vids = voc[c, :m] - 1
    lb[c, :m] = lat_v[vids]
    ln[c, :m] = lon_v[vids]
    lb[c, m:] = lb[c, m - 1]
    ln[c, m:] = ln[c, m - 1]

with tempfile.TemporaryDirectory() as tmpdir:
    lats, lons, sfield = create_test_field(180, 360, "constant", "regular")
    src_nc = os.path.join(tmpdir, "source.nc")
    write_netcdf(src_nc, lats, lons, sfield, "regular")
    ds_mp = xr.Dataset(
        {
            "temperature": (["cell"], np.full(n, 42.0)),
            "lat": (["cell"], lat_c),
            "lon": (["cell"], lon_c),
            "lat_bnds": (["cell", "nv"], lb),
            "lon_bnds": (["cell", "nv"], ln),
        }
    )
    ds_mp["lat"].attrs = {"units": "degrees_north", "bounds": "lat_bnds", "standard_name": "latitude"}
    ds_mp["lon"].attrs = {"units": "degrees_east", "bounds": "lon_bnds", "standard_name": "longitude"}
    ds_mp["temperature"].attrs = {"coordinates": "lon lat", "units": "K"}
    dst_nc = os.path.join(tmpdir, "mpas_dst.nc")
    ds_mp.to_netcdf(dst_nc)

    # exact area-weighted mass of the source on the unit sphere
    dlat = np.radians(lats[1] - lats[0])
    dlon = np.radians(lons[1] - lons[0])
    b = np.clip(np.radians(lats) + dlat / 2, -np.pi / 2, np.pi / 2)
    a = np.clip(np.radians(lats) - dlat / 2, -np.pi / 2, np.pi / 2)
    sr_area = dlon * (np.sin(b) - np.sin(a))
    src_mass = float(np.sum(sfield * np.repeat(sr_area[:, None], len(lons), axis=1)))
    print(f"source: {len(lons)}x{len(lats)} regular, value=42.0")
    print(f"source area-weighted mass = {src_mass:.4f}   (42 * 4pi = {42 * 4 * np.pi:.4f})")

    # CDO via subprocess (python-cdo chokes on CDO's non-UTF8 stderr)
    import subprocess

    cdo_out = os.path.join(tmpdir, "cdo.nc")
    p = subprocess.run(["cdo", "-s", "-f", "nc4", f"remapcon,{dst_nc}", src_nc, cdo_out], capture_output=True)
    assert p.returncode == 0, p.stderr[-400:]
    cdo_v = xr.open_dataset(cdo_out)["temperature"].values
    print(f"\nCDO remapcon        : sum={np.nansum(cdo_v):+.4f}  nan={int(np.isnan(cdo_v).sum())}/{cdo_v.size}")

    ds_in = xr.open_dataset(src_nc)
    ds_out = xr.open_dataset(dst_nc)
    for lt in ("great_circle", "cartesian"):
        r = axis.Regridder(ds_in, ds_out, method="conservative", periodic=True, norm_type="fracarea", line_type=lt)
        v = r(ds_in["temperature"]).values
        print(f"AXIS {lt:<13} : sum={np.nansum(v):+.4f}  nan={int(np.isnan(v).sum())}/{v.size}  mean={np.nanmean(v):.4f}")
    print(
        "\n(A constant field remapped conservatively should stay ~42.0 everywhere;"
        "\n a sum well below 42*n_cells indicates dropped/under-covered cells.)"
    )
