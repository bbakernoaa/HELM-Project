# SPDX-License-Identifier: Apache-2.0
"""
AXIS Python package — stateless, high-performance spatial regridding for xarray.
"""
# ruff: noqa: I001

import sys

# Import axis_py FIRST using relative import to completely avoid partially initialized circular issues
from . import axis_py

# Register the custom .axis xarray accessor
from . import accessors  # noqa: F401
from .grid import CurvilinearGrid, Geometry, GridFactory, RectilinearGrid, UnstructuredMesh, RuleGeometry
from .regridder import Regridder
from .vector import VectorRegridder
from .vertical import VerticalRegridder, regrid_3d
from . import spherical

# Transparent backwards-compatibility bridge (HELM Law #3):
sys.modules["axis_py"] = axis_py

# Expose C++ Mesh construction and Matrix serialization APIs directly on the axis package
Mesh = axis_py.Mesh
Matrix = axis_py.Matrix
make_regular_mesh = axis_py.make_regular_mesh
make_projected_mesh = axis_py.make_projected_mesh
make_ugrid_mesh = axis_py.make_ugrid_mesh
make_named_mesh = axis_py.make_named_mesh
apply_weights = axis_py.apply_weights
batch_apply = axis_py.batch_apply
write_gmsh = axis_py.write_gmsh
reconstruct_gradient = axis_py.reconstruct_gradient
detect_tripolar_grid = axis_py.detect_tripolar_grid
detect_regular_grid = axis_py.detect_regular_grid
detect_rectilinear_grid = axis_py.detect_rectilinear_grid
adjust_by_fraction = axis_py.adjust_by_fraction
generate_vector_weights = axis_py.generate_vector_weights
Method = axis_py.Method
NormType = axis_py.NormType
UnmappedAction = axis_py.UnmappedAction
LineType = axis_py.LineType

__all__ = [
    "Regridder",
    "Geometry",
    "RectilinearGrid",
    "CurvilinearGrid",
    "UnstructuredMesh",
    "RuleGeometry",
    "GridFactory",
    "VectorRegridder",
    "VerticalRegridder",
    "regrid_3d",
    "Mesh",
    "Matrix",
    "make_regular_mesh",
    "make_projected_mesh",
    "make_ugrid_mesh",
    "make_named_mesh",
    "apply_weights",
    "batch_apply",
    "write_gmsh",
    "reconstruct_gradient",
    "detect_tripolar_grid",
    "detect_regular_grid",
    "detect_rectilinear_grid",
    "adjust_by_fraction",
    "generate_vector_weights",
    "Method",
    "NormType",
    "UnmappedAction",
    "LineType",
    "spherical",
]
