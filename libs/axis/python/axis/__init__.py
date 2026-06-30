# SPDX-License-Identifier: Apache-2.0
"""
AXIS Python package — stateless, high-performance spatial regridding for xarray.
"""
import sys

# Import axis_py FIRST using relative import to completely avoid partially initialized circular issues
from . import axis_py

# Transparent backwards-compatibility bridge (HELM Law #3):
sys.modules["axis_py"] = axis_py

from .regridder import Regridder
from . import accessors  # Automatically registers the custom .axis xarray accessor!

# Expose C++ Mesh construction and Matrix serialization APIs directly on the axis package
Mesh = axis_py.Mesh
Matrix = axis_py.Matrix
make_regular_mesh = axis_py.make_regular_mesh
make_projected_mesh = axis_py.make_projected_mesh
make_ugrid_mesh = axis_py.make_ugrid_mesh
make_named_mesh = axis_py.make_named_mesh
apply_weights = axis_py.apply_weights
batch_apply = axis_py.batch_apply
Method = axis_py.Method
NormType = axis_py.NormType
UnmappedAction = axis_py.UnmappedAction
LineType = axis_py.LineType

__all__ = [
    "Regridder",
    "Mesh",
    "Matrix",
    "make_regular_mesh",
    "make_projected_mesh",
    "make_ugrid_mesh",
    "make_named_mesh",
    "apply_weights",
    "batch_apply",
    "Method",
    "NormType",
    "UnmappedAction",
    "LineType",
]
