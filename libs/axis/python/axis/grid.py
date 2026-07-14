# SPDX-License-Identifier: Apache-2.0

import numpy as np
import xarray as xr

from . import axis_py

# Unstructured spatial dimension tags commonly used in climate datasets
UNSTRUCTURED_DIMS = {
    "ncol",
    "grid_size",
    "nCells",
    "nVertices",
    "nNodes",
    "nFaces",
    "nEdges",
    "n_node",
    "n_face",
    "n_edge",
    "n_cells",
    "n_vertices",
    "node",
    "face",
    "vertex",
    "cell",
    "n_pts",
}


def _get_non_spatial_dims(ds: xr.Dataset) -> set[str]:
    """Identify and filter out non-spatial dimensions (Time, Z, Member)."""
    spatial_keywords = {
        "lat",
        "lon",
        "x",
        "y",
        "node",
        "face",
        "element",
        "cell",
        "n_pts",
        "ncol",
        "ncells",
        "grid_size",
        "vert",
        "vertex",
        "vertices",
        "tile",
    }
    non_spatial = set()
    for d in ds.dims:
        d_lower = str(d).lower()
        if not any(kw in d_lower for kw in spatial_keywords):
            non_spatial.add(str(d))
    return non_spatial


def _find_coord(ds: xr.Dataset, name: str) -> xr.DataArray | None:
    """Find a coordinate array based on standard_name, axis, or name heuristics."""
    for c in ds.coords:
        da = ds[c]
        if da.attrs.get("standard_name") == name:
            return da
        if name == "latitude" and da.attrs.get("axis") == "Y":
            return da
        if name == "longitude" and da.attrs.get("axis") == "X":
            return da
    return None


def _get_mesh_info(
    ds: xr.Dataset,
    method: str | None = None,
) -> tuple[xr.DataArray, xr.DataArray, tuple[int, ...], tuple[str, ...], bool]:
    """Detect grid type and extract coordinate DataArrays and dimensions."""
    non_spatial_dims = _get_non_spatial_dims(ds)

    lat = None
    lon = None

    # CF search priority
    lat = _find_coord(ds, "latitude")
    if lat is None:
        for v in ["lat", "latCell", "lat_face", "lat_node", "latitude"]:
            if v in ds:
                lat = ds[v]
                break

    if lat is None:
        raise KeyError("Could not find latitude coordinates in dataset.")

    lon_name = lat.name.replace("lat", "lon").replace("LAT", "LON").replace("latitude", "longitude")
    if lon_name in ds:
        lon = ds[lon_name]
    else:
        lon = _find_coord(ds, "longitude")

    if lon is None:
        for v in ["lon", "lonCell", "lon_face", "lon_node", "longitude"]:
            if v in ds:
                lon = ds[v]
                break

    if lon is None:
        raise KeyError("Could not find matching longitude coordinates.")

    # Drop non-spatial dimensions if present
    lat_isel = {d: 0 for d in non_spatial_dims if d in lat.dims}
    lon_isel = {d: 0 for d in non_spatial_dims if d in lon.dims}
    if lat_isel:
        lat = lat.isel(lat_isel, drop=True)
    if lon_isel:
        lon = lon.isel(lon_isel, drop=True)

    # Detect if unstructured
    is_unstructured = False
    if "mesh" in lat.attrs and "location" in lat.attrs:
        is_unstructured = True
    elif any(d in lat.dims for d in UNSTRUCTURED_DIMS):
        is_unstructured = True
    else:
        for var in ds.variables:
            if ds[var].attrs.get("cf_role") == "mesh_topology":
                is_unstructured = True
                break

    if not is_unstructured:
        if lat.ndim == 3:
            is_unstructured = True
        elif lat.ndim == 2:
            # If 2D curvilinear with NO lambert conformal conic mapping, treat as unstructured quad mesh
            has_lcc = False
            for v in ds.variables:
                if ds[v].attrs.get("grid_mapping_name") == "lambert_conformal_conic":
                    has_lcc = True
                    break
            if not has_lcc:
                is_unstructured = True

    if is_unstructured:
        return lon, lat, lat.shape, lat.dims, True
    else:
        # Structured grid: both latitude and longitude are spatial dimensions!
        if lat.ndim == 1:
            # Regular grid with 1D lat and 1D lon.
            # Return full 2D spatial dims and shape
            dims = (lat.name, lon.name)
            shape = (len(lat), len(lon))
            return lon, lat, shape, dims, False
        else:
            # Curvilinear grid with 2D lat and 2D lon
            return lon, lat, lat.shape, lat.dims, False


def _synthesize_curvilinear_corners(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize (ny+1, nx+1) corner coordinates from (ny, nx) cell centers via fast 2D slicing."""
    ny, nx = lon.shape
    pad_lon = np.pad(lon, 1, mode="edge")
    pad_lat = np.pad(lat, 1, mode="edge")

    # Sum the 4 surrounding padded elements
    sum_lon = pad_lon[:-1, :-1] + pad_lon[:-1, 1:] + pad_lon[1:, :-1] + pad_lon[1:, 1:]
    sum_lat = pad_lat[:-1, :-1] + pad_lat[:-1, 1:] + pad_lat[1:, :-1] + pad_lat[1:, 1:]

    clon = sum_lon / 4.0
    clat = sum_lat / 4.0
    return clon, clat


def _triangulate_mpas_mesh(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Triangulate arbitrary polygon cells (like MPAS Voronoi cells) into triangles."""
    non_spatial_dims = _get_non_spatial_dims(ds)

    v_lat = ds["latVertex"]
    v_lon = ds["lonVertex"]
    v_conn = ds["verticesOnCell"]

    # Filter non-spatial dimensions individually to prevent mismatched dimension indexing
    isel_lat = {d: 0 for d in non_spatial_dims if d in v_lat.dims}
    if isel_lat:
        v_lat = v_lat.isel(isel_lat, drop=True)

    isel_lon = {d: 0 for d in non_spatial_dims if d in v_lon.dims}
    if isel_lon:
        v_lon = v_lon.isel(isel_lon, drop=True)

    isel_conn = {d: 0 for d in non_spatial_dims if d in v_conn.dims}
    if isel_conn:
        v_conn = v_conn.isel(isel_conn, drop=True)

    # Normalize longitudes and latitudes to degrees
    node_lat = v_lat.values
    node_lon = v_lon.values
    if np.any(np.abs(node_lat) > 2.0 * np.pi):
        pass  # Already degrees
    else:
        node_lat = np.degrees(node_lat)
        node_lon = np.degrees(node_lon)

    # Wrap longitudes to [0, 360]
    node_lon = np.mod(node_lon, 360.0)

    conn_raw = v_conn.values
    n_edges = ds["nEdgesOnCell"].values if "nEdgesOnCell" in ds else np.full(conn_raw.shape[0], conn_raw.shape[1])

    n_cells, max_edges = conn_raw.shape
    max_tris = max_edges - 2
    j = np.arange(1, max_tris + 1)
    mask = j[None, :] < (n_edges[:, None] - 1)

    v0 = np.repeat(conn_raw[:, 0:1], max_tris, axis=1) - 1
    v1 = conn_raw[:, 1:-1] - 1
    v2 = conn_raw[:, 2:] - 1

    element_conn = np.stack([v0[mask], v1[mask], v2[mask]], axis=1).flatten()
    np.repeat(np.arange(n_cells), max_tris)[mask.flatten()]

    return node_lon, node_lat, element_conn.astype(np.int32)


def _parse_scrip_bounds(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse unstructured SCRIP-style cell centers and 2D bounds into general polygon nodes/connectivity offsets/indices."""
    # Find longitude/latitude coordinates
    lat = None
    for v in ["lat", "latCell", "latitude"]:
        if v in ds:
            lat = ds[v]
            break
    if lat is None:
        raise KeyError("Could not find latitude coordinates in dataset.")

    lon_name = lat.name.replace("lat", "lon").replace("LAT", "LON").replace("latitude", "longitude")
    if lon_name in ds:
        lon = ds[lon_name]
    else:
        for v in ["lon", "lonCell", "longitude"]:
            if v in ds:
                lon = ds[v]
                break
    if lon is None:
        raise KeyError("Could not find longitude coordinates.")

    lat_bnds_name = lat.attrs.get("bounds", "lat_bnds")
    lon_bnds_name = lon.attrs.get("bounds", "lon_bnds")

    lat_bnds = ds[lat_bnds_name].values
    lon_bnds = ds[lon_bnds_name].values

    n_cells, nv = lat_bnds.shape

    node_lons = []
    node_lats = []
    conn_offsets = [0]
    conn_indices = []

    node_counter = 0
    for idx in range(n_cells):
        lats_c = lat_bnds[idx]
        lons_c = lon_bnds[idx]

        # Filter out repeated padded corners
        cell_vertices = []
        for v in range(nv):
            # Skip repeated padded corners (standard CDO SCRIP padding)
            if v > 0 and lats_c[v] == lats_c[v - 1] and lons_c[v] == lons_c[v - 1]:
                continue
            cell_vertices.append((lons_c[v], lats_c[v]))

        n_vertices = len(cell_vertices)
        if n_vertices < 3:
            # Fallback: if too many repeated, just use the first 3
            cell_vertices = [(lons_c[0], lats_c[0]), (lons_c[1], lats_c[1]), (lons_c[2], lats_c[2])]
            n_vertices = 3

        for lon_val, lat_val in cell_vertices:
            node_lons.append(lon_val)
            node_lats.append(lat_val)
            conn_indices.append(node_counter)
            node_counter += 1

        conn_offsets.append(len(conn_indices))

    return (
        np.mod(np.array(node_lons), 360.0),
        np.array(node_lats),
        np.array(conn_offsets, dtype=np.int32),
        np.array(conn_indices, dtype=np.int32),
    )


def _get_ugrid_info(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract standard UGRID mesh connectivity and node coordinates."""
    mesh_var = None
    for var in ds.variables:
        if ds[var].attrs.get("cf_role") == "mesh_topology":
            mesh_var = var
            break

    if mesh_var is None:
        raise KeyError("Could not find CF UGRID mesh_topology variable.")

    attrs = ds[mesh_var].attrs
    node_coords_names = attrs.get("node_coordinates", "").split()
    face_conn_name = attrs.get("face_node_connectivity", "")

    node_lon = ds[node_coords_names[0]].values
    node_lat = ds[node_coords_names[1]].values
    face_conn = ds[face_conn_name].values

    # Adjust for 1-based indexing in some UGRID files
    start_index = ds[face_conn_name].attrs.get("start_index", 0)
    if start_index == 1:
        face_conn = face_conn - 1

    return node_lon, node_lat, face_conn.astype(np.int32).flatten()


def create_axis_mesh(ds: xr.Dataset, method: str | None = None) -> axis_py.Mesh:
    """Build a native C++ Kokkos-parallel AXIS Mesh from an xarray dataset."""
    lon, lat, shape, dims, is_unstructured = _get_mesh_info(ds, method)

    if is_unstructured:
        # 1. MPAS (Arbitrary polygonal cells)
        if "verticesOnCell" in ds and "latVertex" in ds:
            # Check if method is cell-centered (nearest/conservative/conservative2nd)
            cell_centered = (method is not None) and (method.lower() in ["nearest", "conservative", "conservative2nd"])

            # Normalize and wrap coordinates
            v_lat = ds["latVertex"]
            v_lon = ds["lonVertex"]
            non_spatial_dims = _get_non_spatial_dims(ds)
            isel_lat = {d: 0 for d in non_spatial_dims if d in v_lat.dims}
            if isel_lat:
                v_lat = v_lat.isel(isel_lat, drop=True)
            isel_lon = {d: 0 for d in non_spatial_dims if d in v_lon.dims}
            if isel_lon:
                v_lon = v_lon.isel(isel_lon, drop=True)

            node_lat = v_lat.values
            node_lon = v_lon.values
            if not np.any(np.abs(node_lat) > 2.0 * np.pi):
                node_lat = np.degrees(node_lat)
                node_lon = np.degrees(node_lon)
            node_lon = np.mod(node_lon, 360.0)
            node_coords = np.asfortranarray(np.column_stack([node_lon, node_lat]))

            if cell_centered:
                # Raw polygonal MPAS mesh directly (no triangulation!)
                v_conn = ds["verticesOnCell"]
                isel_conn = {d: 0 for d in non_spatial_dims if d in v_conn.dims}
                if isel_conn:
                    v_conn = v_conn.isel(isel_conn, drop=True)

                conn_raw = v_conn.values
                n_edges = ds["nEdgesOnCell"].values if "nEdgesOnCell" in ds else np.full(conn_raw.shape[0], conn_raw.shape[1])

                conn_offsets = np.zeros(len(n_edges) + 1, dtype=np.int32)
                conn_offsets[1:] = np.cumsum(n_edges)

                n_cells, max_edges = conn_raw.shape
                row_indices = np.arange(max_edges)
                mask = row_indices[None, :] < n_edges[:, None]
                conn_indices = (conn_raw[mask] - 1).astype(np.int32)

                return axis_py.make_ugrid_mesh(node_coords, conn_offsets, conn_indices)
            else:
                # Triangulated MPAS (for bilinear/bicubic/patch)
                node_lon, node_lat, element_conn = _triangulate_mpas_mesh(ds)
                node_coords = np.asfortranarray(np.column_stack([node_lon, node_lat]))
                conn_offsets = np.arange(0, len(element_conn) + 1, 3, dtype=np.int32)
                conn_indices = element_conn.astype(np.int32)
                return axis_py.make_ugrid_mesh(node_coords, conn_offsets, conn_indices)
        # 2. SCRIP 2D Bounds format
        elif "lat_bnds" in ds or any("bounds" in ds[v].attrs for v in ds.variables if v in ["lat", "lon"]):
            node_lon, node_lat, conn_offsets, conn_indices = _parse_scrip_bounds(ds)
            node_coords = np.asfortranarray(np.column_stack([node_lon, node_lat]))
            return axis_py.make_ugrid_mesh(node_coords, conn_offsets, conn_indices)
        # 3. Curvilinear (2D) or Cubed-Sphere (3D) coordinate arrays fallback
        elif lat.ndim in [2, 3]:
            if lat.ndim == 2:
                # 2D Curvilinear grid
                clon, clat = _synthesize_curvilinear_corners(lon.values, lat.values)
                node_coords = np.asfortranarray(np.column_stack([clon.ravel(), clat.ravel()]))

                ni, nj = lon.shape[1], lon.shape[0]
                nip1 = ni + 1
                n_cells = ni * nj
                conn_offsets = np.arange(0, n_cells * 4 + 1, 4, dtype=np.int32)

                i_grid, j_grid = np.meshgrid(np.arange(ni), np.arange(nj))
                bl = (i_grid + j_grid * nip1).ravel()
                br = ((i_grid + 1) + j_grid * nip1).ravel()
                tr = ((i_grid + 1) + (j_grid + 1) * nip1).ravel()
                tl = (i_grid + (j_grid + 1) * nip1).ravel()

                conn_indices = np.column_stack([bl, br, tr, tl]).astype(np.int32).ravel()
                return axis_py.make_ugrid_mesh(node_coords, conn_offsets, conn_indices)
            else:
                # 3D Cubed-Sphere grid (ntiles, ny, nx)
                ntiles, ny, nx = lon.shape
                node_coords_list = []
                conn_indices_list = []
                node_offset = 0

                for t in range(ntiles):
                    clon_t, clat_t = _synthesize_curvilinear_corners(lon[t].values, lat[t].values)
                    coords_t = np.column_stack([clon_t.ravel(), clat_t.ravel()])
                    node_coords_list.append(coords_t)

                    ni, nj = nx, ny
                    nip1 = ni + 1
                    i_grid, j_grid = np.meshgrid(np.arange(ni), np.arange(nj))
                    bl = (i_grid + j_grid * nip1 + node_offset).ravel()
                    br = ((i_grid + 1) + j_grid * nip1 + node_offset).ravel()
                    tr = ((i_grid + 1) + (j_grid + 1) * nip1 + node_offset).ravel()
                    tl = (i_grid + (j_grid + 1) * nip1 + node_offset).ravel()

                    indices_t = np.column_stack([bl, br, tr, tl]).astype(np.int32).ravel()
                    conn_indices_list.append(indices_t)

                    node_offset += len(coords_t)

                node_coords = np.asfortranarray(np.concatenate(node_coords_list))
                conn_indices = np.concatenate(conn_indices_list)
                conn_offsets = np.arange(0, len(conn_indices) + 1, 4, dtype=np.int32)

                return axis_py.make_ugrid_mesh(node_coords, conn_offsets, conn_indices)
        # 4. CF-UGRID standard
        else:
            node_lon, node_lat, element_conn = _get_ugrid_info(ds)
            node_coords = np.asfortranarray(np.column_stack([node_lon, node_lat]))
            conn_offsets = np.arange(0, len(element_conn) + 1, 3, dtype=np.int32)
            conn_indices = element_conn.astype(np.int32)
            return axis_py.make_ugrid_mesh(node_coords, conn_offsets, conn_indices)
    else:
        # Structured: regular or rectilinear/projected
        if lon.ndim == 1 and lat.ndim == 1:
            # Regular grid
            ni = len(lon)
            nj = len(lat)
            lon_start = float(lon[0])
            lat_start = float(lat[0])
            dlon = float(lon[1] - lon[0]) if ni > 1 else 1.0
            dlat = float(lat[1] - lat[0]) if nj > 1 else 1.0

            return axis_py.make_regular_mesh(ni, nj, lon_start, lat_start, dlon, dlat)
        else:
            # 2D Curvilinear or Projected grid
            # If coordinates have a grid_mapping or PROJ metadata, build a projected mesh
            grid_mapping = None
            for v in ds.variables:
                if "grid_mapping_name" in ds[v].attrs:
                    grid_mapping = ds[v].attrs["grid_mapping_name"]
                    break

            if grid_mapping == "lambert_conformal_conic":
                # Generate local projection string and coordinates
                # Host models can configure custom LCC params here, otherwise fallback to CONUS standard
                if not getattr(axis_py, "HAVE_PROJ", False):
                    raise RuntimeError(
                        f"This dataset uses a '{grid_mapping}' grid_mapping, which "
                        "requires projected-coordinate support, but axis_py was built without "
                        "PROJ (AXIS_ENABLE_PROJ=OFF). Rebuild AXIS with -DAXIS_ENABLE_PROJ=ON "
                        "to regrid projected grids."
                    ) from None
                proj_string = "+proj=lcc +lat_1=25 +lat_2=25 +lat_0=25 +lon_0=-95 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
                ni, nj = shape[1], shape[0]
                center_x = lon.values.ravel()
                center_y = lat.values.ravel()
                return axis_py.make_projected_mesh(ni, nj, proj_string, center_x, center_y)
            else:
                # Flat regular 2D fallback
                ni, nj = shape[1], shape[0]
                lon_start = float(lon[0, 0])
                lat_start = float(lat[0, 0])
                dlon = float(lon[0, 1] - lon[0, 0]) if ni > 1 else 1.0
                dlat = float(lat[1, 0] - lat[0, 0]) if nj > 1 else 1.0

                return axis_py.make_regular_mesh(ni, nj, lon_start, lat_start, dlon, dlat)


# ==============================================================================
# Low-Level Explicit Geometry API
# ==============================================================================


class Geometry:
    """Base abstract class representing any physical coordinate layout in AXIS."""

    def to_mesh(self, method: str | None = None) -> axis_py.Mesh:
        """Convert this geometry to a unified C++ UnstructuredMesh."""
        raise NotImplementedError()


class XarrayGeometry(Geometry):
    """Wraps an xarray Dataset or DataArray to defer mesh construction."""

    def __init__(self, ds: xr.Dataset | xr.DataArray, method: str | None = None):
        self.ds = ds
        self.method = method

    def to_mesh(self, method: str | None = None) -> axis_py.Mesh:
        # Keep 100% of existing, verified auto-detection and triangulation/centering logic!
        ds_normalized = self.ds.to_dataset(name="_tmp_data") if isinstance(self.ds, xr.DataArray) else self.ds
        return create_axis_mesh(ds_normalized, method or self.method)


class RectilinearGrid(Geometry):
    """
    Represent a standard 2D lat-lon grid with 1-D coordinate vectors.
    """

    def __init__(self, lons: np.ndarray, lats: np.ndarray):
        self.lons = np.asarray(lons, dtype=np.float64)
        self.lats = np.asarray(lats, dtype=np.float64)

    def to_mesh(self, method: str | None = None) -> axis_py.Mesh:
        ni = len(self.lons)
        nj = len(self.lats)
        dlon = float((self.lons[-1] - self.lons[0]) / (ni - 1)) if ni > 1 else 1.0
        dlat = float((self.lats[-1] - self.lats[0]) / (nj - 1)) if nj > 1 else 1.0
        return axis_py.make_regular_mesh(ni, nj, float(self.lons[0]), float(self.lats[0]), dlon, dlat)


class CurvilinearGrid(Geometry):
    """
    Represent a 2D curvilinear grid with 2-D coordinate matrices.
    """

    def __init__(self, lons: np.ndarray, lats: np.ndarray, proj_string: str | None = None):
        self.lons = np.asarray(lons, dtype=np.float64)
        self.lats = np.asarray(lats, dtype=np.float64)
        self.proj_string = proj_string

    def to_mesh(self, method: str | None = None) -> axis_py.Mesh:
        if self.proj_string:
            return axis_py.make_projected_mesh(
                self.lons.shape[1],
                self.lons.shape[0],
                self.proj_string,
                self.lons.ravel(),
                self.lats.ravel(),
            )
        else:
            clon, clat = _synthesize_curvilinear_corners(self.lons, self.lats)
            ny, nx = self.lons.shape
            n_cells = nx * ny

            node_coords = np.asfortranarray(np.column_stack([clon.ravel(), clat.ravel()]))
            conn_offsets = np.arange(0, (n_cells + 1) * 4, 4, dtype=np.int64)

            conn_indices = np.zeros(n_cells * 4, dtype=np.int64)
            cell_idx = 0
            ncol = nx + 1
            for j in range(ny):
                for i in range(nx):
                    bl = i + j * ncol
                    br = (i + 1) + j * ncol
                    tr = (i + 1) + (j + 1) * ncol
                    tl = i + (j + 1) * ncol

                    base = cell_idx * 4
                    conn_indices[base + 0] = bl
                    conn_indices[base + 1] = br
                    conn_indices[base + 2] = tr
                    conn_indices[base + 3] = tl
                    cell_idx += 1

            return axis_py.make_ugrid_mesh(node_coords, conn_offsets, conn_indices)


class UnstructuredMesh(Geometry):
    """
    Represent an arbitrary unstructured polygon grid (e.g. MPAS, FVCOM, SCRIP).
    """

    def __init__(
        self,
        node_coords: np.ndarray,
        connectivity_offsets: np.ndarray,
        connectivity_indices: np.ndarray,
    ):
        self.coords = np.asfortranarray(node_coords, dtype=np.float64)
        self.offsets = np.asarray(connectivity_offsets, dtype=np.int64)
        self.indices = np.asarray(connectivity_indices, dtype=np.int64)

    def to_mesh(self, method: str | None = None) -> axis_py.Mesh:
        return axis_py.make_ugrid_mesh(self.coords, self.offsets, self.indices)


class GridFactory:
    """
    Convenience factory to convert high-level containers to Geometry classes.
    """

    @staticmethod
    def from_xarray(
        ds: xr.Dataset | xr.DataArray,
        lon_var: str | None = None,
        lat_var: str | None = None,
        method: str | None = None,
    ) -> Geometry:
        """
        Build an explicit AXIS geometry from an xarray container.
        """
        if lon_var and lat_var:
            lons = ds[lon_var].values
            lats = ds[lat_var].values
            if lons.ndim == 1:
                return RectilinearGrid(lons, lats)
            return CurvilinearGrid(lons, lats)
        return XarrayGeometry(ds, method=method)
