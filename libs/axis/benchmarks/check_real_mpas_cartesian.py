#!/usr/bin/env python3
"""Characterize the cartesian-mode coverage drop on the real MPAS x1.2562 grid.

Cross-tabulates AXIS cartesian zero-coverage against:
  - planar shoelace winding sign (in lon/lat degrees)
  - dateline straddle (vertex lon jump > 180)
  - cell latitude
and compares with great_circle coverage.
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

planar_cw = np.zeros(n, bool)
straddle = np.zeros(n, bool)
lat_c = np.degrees(ds["latCell"].values)
lon_c = np.degrees(ds["lonCell"].values) % 360.0
maxnv = int(nec.max())
lb = np.zeros((n, maxnv))
ln = np.zeros((n, maxnv))
for c in range(n):
    m = int(nec[c])
    vids = voc[c, :m] - 1
    lo = lon_v[vids]
    la = lat_v[vids]
    x = np.radians(lo)
    y = np.radians(la)
    sl = 0.5 * (np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + x[-1] * y[0] - x[0] * y[-1])
    planar_cw[c] = sl < 0
    d = np.abs(np.diff(np.concatenate([lo, [lo[0]]])))
    straddle[c] = d.max() > 180
    lb[c, :m] = la
    ln[c, :m] = lo
    lb[c, m:] = lb[c, m - 1]
    ln[c, m:] = ln[c, m - 1]

with tempfile.TemporaryDirectory() as tmpdir:
    lats, lons, sfield = create_test_field(180, 360, "cosine", "regular")
    src_nc = os.path.join(tmpdir, "source.nc")
    write_netcdf(src_nc, lats, lons, sfield, "regular")
    ds_mp = xr.Dataset(
        {"lat": (["cell"], lat_c), "lon": (["cell"], lon_c), "lat_bnds": (["cell", "nv"], lb), "lon_bnds": (["cell", "nv"], ln)}
    )
    ds_mp["lat"].attrs = {"units": "degrees_north", "bounds": "lat_bnds"}
    ds_mp["lon"].attrs = {"units": "degrees_east", "bounds": "lon_bnds"}
    dst_nc = os.path.join(tmpdir, "mpas_dst.nc")
    ds_mp.to_netcdf(dst_nc)

    ds_in = xr.open_dataset(src_nc)
    ds_out = xr.open_dataset(dst_nc)
    ones = xr.DataArray(np.ones(sfield.shape), dims=ds_in["temperature"].dims, coords=ds_in["temperature"].coords)
    cov = {}
    for lt in ("great_circle", "cartesian"):
        r = axis.Regridder(ds_in, ds_out, method="conservative", periodic=True, norm_type="dstarea", line_type=lt)
        cov[lt] = np.nan_to_num(r(ones).values, nan=0.0)

gc_zero = cov["great_circle"] < 1e-9
cc_zero = cov["cartesian"] < 1e-9
print(f"cells={n}   GC zero-cov={int(gc_zero.sum())}   CART zero-cov={int(cc_zero.sum())}")
print(f"\n{'subset':<28} {'n':>5} {'CART zero':>10} {'rate':>7}")
for name, m in (
    ("all", np.ones(n, bool)),
    ("planar CCW", ~planar_cw),
    ("planar CW", planar_cw),
    ("dateline straddle", straddle),
    ("no straddle", ~straddle),
    ("|lat| < 30", np.abs(lat_c) < 30),
    ("30<=|lat|<60", (np.abs(lat_c) >= 30) & (np.abs(lat_c) < 60)),
    ("|lat| >= 60", np.abs(lat_c) >= 60),
):
    z = int((cc_zero & m).sum())
    print(f"{name:<28} {int(m.sum()):>5} {z:>10} {z / max(int(m.sum()), 1):>7.2f}")

print(f"\nstraddle & planar CW: {int((straddle & planar_cw).sum())}")
print(f"CART zero among straddle: {int((cc_zero & straddle).sum())}/{int(straddle.sum())}")
print(f"CART zero among non-straddle: {int((cc_zero & ~straddle).sum())}/{int((~straddle).sum())}")
print(f"\nlat range of CART zero cells: {lat_c[cc_zero].min():.1f} .. {lat_c[cc_zero].max():.1f}")
print(f"lat range of CART covered   : {lat_c[~cc_zero].min():.1f} .. {lat_c[~cc_zero].max():.1f}")
