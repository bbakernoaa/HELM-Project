#!/usr/bin/env python3
"""Conservation check on a REAL MPAS atmosphere mesh (x1.2562).

Two questions:
  1. What is the vertex winding convention of real MPAS cells? (The synthetic
     scipy-Voronoi grid in compare_cdo.py has arbitrary winding, which is what
     exposed the clipper's CCW assumption.)
  2. Does AXIS conserve mass into the real MPAS grid the way CDO does?

Read-only w.r.t. library code.
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


def to_xyz(lon, lat):
    lr = np.radians(lon)
    ar = np.radians(lat)
    c = np.cos(ar)
    return np.stack([c * np.cos(lr), c * np.sin(lr), np.sin(ar)], axis=-1)


def signed_sph_area(v):
    """Signed spherical excess via triangle fan (clipper's effective criterion)."""
    n = len(v)
    if n < 3:
        return 0.0
    total = 0.0
    a = v[0]
    for i in range(1, n - 1):
        b, c = v[i], v[i + 1]
        num = np.dot(a, np.cross(b, c))
        den = 1.0 + np.dot(a, b) + np.dot(a, c) + np.dot(b, c)
        total += 2.0 * np.arctan2(num, den)
    return total


def cell_polygons(ds):
    """Yield (cell_index, n_verts, lon[], lat[]) from native MPAS connectivity."""
    voc = ds["verticesOnCell"].values  # (nCells, maxEdges), 1-based, 0=pad
    nec = ds["nEdgesOnCell"].values  # (nCells,)
    lon_v = np.degrees(ds["lonVertex"].values) % 360.0
    lat_v = np.degrees(ds["latVertex"].values)
    for c in range(ds.sizes["nCells"]):
        m = int(nec[c])
        vids = voc[c, :m] - 1
        yield c, m, lon_v[vids], lat_v[vids]


ds = xr.open_dataset(MPAS)
n_cells = ds.sizes["nCells"]

# ── 1. winding census on the real grid ───────────────────────────────────────
areas = np.zeros(n_cells)
nv = np.zeros(n_cells, int)
for c, m, lo, la in cell_polygons(ds):
    nv[c] = m
    areas[c] = signed_sph_area(to_xyz(lo, la))

print(f"REAL MPAS x1.2562: nCells={n_cells}")
print(f"  vertices/cell: min={nv.min()} max={nv.max()} (histogram {np.bincount(nv)[nv.min() :].tolist()})")
print(f"  signed spherical excess:  CCW(>0)={int((areas > 0).sum())}  CW(<0)={int((areas < 0).sum())}")
print(f"  => real MPAS winding is {'CONSISTENT (all same sign)' if (areas > 0).all() or (areas < 0).all() else 'MIXED'}")
print(f"  |area| range: {np.abs(areas).min():.3e} .. {np.abs(areas).max():.3e} sr")

# ── 2. AXIS vs CDO conservation into the real MPAS grid ──────────────────────
with tempfile.TemporaryDirectory() as tmpdir:
    lats, lons, sfield = create_test_field(180, 360, "cosine", "regular")
    src_nc = os.path.join(tmpdir, "source.nc")
    write_netcdf(src_nc, lats, lons, sfield, "regular")

    # Present the MPAS grid to AXIS/CDO in SCRIP-style bounds form (what the
    # benchmark harness writes) so both engines see identical geometry.
    lb = np.full((n_cells, int(nv.max())), np.nan)
    ln = np.full((n_cells, int(nv.max())), np.nan)
    for c, m, lo, la in cell_polygons(ds):
        lb[c, :m] = la
        ln[c, :m] = lo
    # pad by repeating the last corner (standard SCRIP padding)
    for c in range(n_cells):
        m = int(nv[c])
        lb[c, m:] = lb[c, m - 1]
        ln[c, m:] = ln[c, m - 1]

    lat_c = np.degrees(ds["latCell"].values)
    lon_c = np.degrees(ds["lonCell"].values) % 360.0

    ds_mp = xr.Dataset(
        {"lat": (["cell"], lat_c), "lon": (["cell"], lon_c), "lat_bnds": (["cell", "nv"], lb), "lon_bnds": (["cell", "nv"], ln)},
    )
    ds_mp["lat"].attrs = {"units": "degrees_north", "bounds": "lat_bnds"}
    ds_mp["lon"].attrs = {"units": "degrees_east", "bounds": "lon_bnds"}
    dst_nc = os.path.join(tmpdir, "mpas_dst.nc")
    ds_mp.to_netcdf(dst_nc)

    ds_in = xr.open_dataset(src_nc)
    ds_out = xr.open_dataset(dst_nc)
    ones = xr.DataArray(np.ones(sfield.shape), dims=ds_in["temperature"].dims, coords=ds_in["temperature"].coords)

    for lt in ("great_circle", "cartesian"):
        r = axis.Regridder(ds_in, ds_out, method="conservative", periodic=True, norm_type="dstarea", line_type=lt)
        fd = np.nan_to_num(r(ones).values, nan=0.0)
        s = float(np.nansum(r(ds_in["temperature"]).values))
        print(f"\n[AXIS {lt:<12}] covered={int((fd > 1e-9).sum())}/{fd.size}  mean_cov={fd.mean():.4f}  sum={s:+.4f}")
        # correlate zero-coverage with winding
        z = fd < 1e-9
        if z.any():
            print(f"    zero-cov cells: CCW={int((areas[z] > 0).sum())} CW={int((areas[z] < 0).sum())}")
