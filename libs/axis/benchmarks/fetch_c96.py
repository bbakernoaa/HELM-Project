#!/usr/bin/env python3
"""Fetch (and cache) the FV3 C96 cubed-sphere grid tiles on demand.

Instead of committing multi-megabyte grid files to the repository, the C96
benchmark and test scripts call :func:`fetch_c96_tiles`, which downloads the
six ``C96_grid.tileN.nc`` files from the NOAA EMC public fix directory and
caches them under ``libs/axis/data/c96/`` (git-ignored). Subsequent calls
reuse the cache.

Usage from Python::

    from fetch_c96 import fetch_c96_tiles
    tile_files = fetch_c96_tiles()   # -> [.../data/c96/C96_grid.tile1.nc, ...]

Usage from the shell (prints the cached paths, one per line)::

    python benchmarks/fetch_c96.py
"""

import socket
import urllib.request
from pathlib import Path

# NOAA EMC public FV3 fix files for the C96 cubed-sphere grid.
BASE_URL = "https://ftp.emc.ncep.noaa.gov/static_files/public/UFS/GFS/fix/fix_fv3/C96"

TILES = [f"C96_grid.tile{i}.nc" for i in range(1, 7)]

# libs/axis/data/c96 — computed from this file so scripts work from any cwd.
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "c96"


def fetch_c96_tiles(cache_dir=None):
    """Return the paths to the six cached C96 tile NetCDF files.

    Downloads any missing tile into ``data/c96/`` on first use; later calls
    hit the cache. Raises ``RuntimeError`` if a download fails.
    """
    cache = Path(cache_dir) if cache_dir else CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)

    paths = []
    for name in TILES:
        dest = cache / name
        if not dest.exists():
            url = f"{BASE_URL}/{name}"
            print(f"[fetch_c96] downloading {url} ...", flush=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            old_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(60)  # bound stalled reads so CI cannot hang
            try:
                urllib.request.urlretrieve(url, tmp)
            except OSError as exc:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"failed to fetch {url}: {exc}") from exc
            finally:
                socket.setdefaulttimeout(old_timeout)
            tmp.rename(dest)
        paths.append(dest)
    return [str(p) for p in paths]


if __name__ == "__main__":
    for p in fetch_c96_tiles():
        print(p)
