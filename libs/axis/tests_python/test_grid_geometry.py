# test_grid_geometry.py
#
# Verifies explicit coordinate layouts (Rectilinear, Curvilinear, Unstructured).

import numpy as np
from axis.grid import RectilinearGrid, CurvilinearGrid, UnstructuredMesh


def test_rectilinear_grid_generation():
    lons = np.linspace(-180, 180, 36)
    lats = np.linspace(-90, 90, 18)

    grid = RectilinearGrid(lons, lats)
    mesh = grid.to_mesh()

    # 36 * 18 cell mesh has (36+1)*(18+1) = 37 * 19 = 703 nodes, and 36 * 18 = 648 cells
    assert mesh.n_nodes == 703
    assert mesh.n_cells == 648


def test_curvilinear_grid_generation():
    # 5x5 grid centers
    x = np.arange(5)
    y = np.arange(5)
    lons, lats = np.meshgrid(x, y)

    grid = CurvilinearGrid(lons, lats)
    mesh = grid.to_mesh()

    # Curvilinear corner synthesis pads (ny, nx) cells to (ny+1, nx+1) corners
    # For a 5x5 center array: NY=5, NX=5 -> Corners: (NY+1)*(NX+1) = 36 nodes, NY*NX = 25 cells
    assert mesh.n_nodes == 36
    assert mesh.n_cells == 25


def test_unstructured_mesh_generation():
    # Define a single triangular cell with 3 nodes
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float64)
    offsets = np.array([0, 3], dtype=np.int64)
    indices = np.array([0, 1, 2], dtype=np.int64)

    mesh_geom = UnstructuredMesh(coords, offsets, indices)
    mesh = mesh_geom.to_mesh()

    assert mesh.n_nodes == 3
    assert mesh.n_cells == 1
