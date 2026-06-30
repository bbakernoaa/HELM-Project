# SPDX-License-Identifier: Apache-2.0
"""
AXIS Python package — stateless, high-performance spatial regridding for xarray.
"""
from .regridder import Regridder
from . import accessors  # Automatically registers the custom .regrid xarray accessor!

__all__ = ["Regridder"]
