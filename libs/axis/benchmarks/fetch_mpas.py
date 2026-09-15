#!/usr/bin/env python3
"""Fetch (and cache) real MPAS atmosphere meshes on demand.

Instead of committing multi-megabyte grid files to the repository, benchmark
and investigation scripts call :func:`fetch_mpas_grid`, which downloads the
tarball from the UCAR MPAS mesh archive and extracts it under
``libs/axis/data/mpas/`` (git-ignored). Subsequent calls reuse the cache.

Usage from Python::

    from fetch_mpas import fetch_mpas_grid
    grid_nc = fetch_mpas_grid("x1.2562")   # -> .../data/mpas/x1.2562.grid.nc

Usage from the shell (prints the cached path)::

    python benchmarks/fetch_mpas.py x1.2562
"""

import tarfile
import urllib.request
from pathlib import Path

# UCAR MPAS atmosphere mesh archive (https://www2.mmm.ucar.edu/projects/mpas/atmosphere_meshes/)
BASE_URL = "https://www2.mmm.ucar.edu/projects/mpas/atmosphere_meshes"

# libs/axis/data/mpas — computed from this file so scripts work from any cwd.
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "mpas"

# Meshes available for on-demand download: name -> list of tarball names.
MESHES = {
    "x1.2562": ["x1.2562.tar.gz", "x1.2562_static.tar.gz"],
}


def fetch_mpas_grid(name="x1.2562", static=False):
    """Return the path to the extracted MPAS ``.nc`` grid file.

    Downloads and extracts the tarball(s) into ``data/mpas/`` on first use;
    later calls hit the cache. Pass ``static=True`` for the static fields file.
    """
    if name not in MESHES:
        raise KeyError(f"Unknown MPAS mesh {name!r}; available: {sorted(MESHES)}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    wanted = [f"{name}_static.tar.gz"] if static else [f"{name}.tar.gz"]

    for tar_name in wanted:
        marker = CACHE_DIR / tar_name.replace(".tar.gz", ".done")
        if marker.exists():
            continue
        url = f"{BASE_URL}/{tar_name}"
        print(f"[fetch_mpas] downloading {url} ...", flush=True)
        tmp = CACHE_DIR / (tar_name + ".part")
        urllib.request.urlretrieve(url, tmp)
        with tarfile.open(tmp) as tf:
            # extractall filter for Python 3.12+ data-safe extraction
            try:
                tf.extractall(CACHE_DIR, filter="data")
            except TypeError:  # older tarfile without filter kwarg
                tf.extractall(CACHE_DIR)
        tmp.unlink()
        marker.touch()

    nc = CACHE_DIR / f"{name}.static.nc" if static else CACHE_DIR / f"{name}.grid.nc"
    if not nc.exists():
        raise FileNotFoundError(f"Expected {nc} after extracting {wanted}; cache may be corrupt — delete {CACHE_DIR} and retry")
    return nc


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mesh", nargs="?", default="x1.2562", help="Mesh name (default: x1.2562)")
    p.add_argument("--static", action="store_true", help="Fetch the static-fields file instead")
    args = p.parse_args()
    print(fetch_mpas_grid(args.mesh, static=args.static))
