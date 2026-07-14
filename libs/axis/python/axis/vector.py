# SPDX-License-Identifier: Apache-2.0
from typing import Any

import numpy as np
import xarray as xr

from . import axis_py
from .grid import CurvilinearGrid, Geometry, RectilinearGrid


class VectorRegridder:
    """
    Coupled horizontal vector interpolator (e.g. for winds and currents)
    accounting for grid-frame rotation.
    """

    def __init__(
        self,
        src: Geometry | xr.Dataset | dict,
        dst: Geometry | xr.Dataset | dict,
        method: str = "bilinear",
        src_alpha: np.ndarray | None = None,
        dst_alpha: np.ndarray | None = None,
        **kwargs: Any,
    ):
        from .grid import GridFactory

        self._src_geom = src if isinstance(src, Geometry) else GridFactory.from_xarray(src)
        self._dst_geom = dst if isinstance(dst, Geometry) else GridFactory.from_xarray(dst)

        src_mesh = self._src_geom.to_mesh()
        dst_mesh = self._dst_geom.to_mesh()

        n_src_cells = src_mesh.n_cells
        n_dst_cells = dst_mesh.n_cells

        if src_alpha is None:
            src_alpha = np.zeros(n_src_cells, dtype=np.float64)
        else:
            src_alpha = np.asarray(src_alpha, dtype=np.float64)

        if dst_alpha is None:
            dst_alpha = np.zeros(n_dst_cells, dtype=np.float64)
        else:
            dst_alpha = np.asarray(dst_alpha, dtype=np.float64)

        method_map = {
            "bilinear": axis_py.Method.Bilinear,
            "nearest": axis_py.Method.NearestNeighbor,
        }
        method_enum = method_map.get(method.lower(), axis_py.Method.Bilinear)

        config = {
            "method": method_enum,
            "periodic": kwargs.get("periodic", False),
            "line_type": axis_py.LineType.GreatCircle
            if kwargs.get("line_type") == "great_circle"
            else axis_py.LineType.Cartesian,
        }

        self._W_u, self._W_v = axis_py.generate_vector_weights(
            src_mesh, dst_mesh, src_alpha, dst_alpha, config
        )

    def transform(
        self, u: np.ndarray | xr.DataArray, v: np.ndarray | xr.DataArray
    ) -> tuple[np.ndarray | xr.DataArray, np.ndarray | xr.DataArray]:
        """
        Remap and rotate (u, v) wind or current components simultaneously.
        """
        is_xarray = isinstance(u, xr.DataArray) and isinstance(v, xr.DataArray)

        u_arr = np.asarray(u.values if is_xarray else u, dtype=np.float64)
        v_arr = np.asarray(v.values if is_xarray else v, dtype=np.float64)

        # Determine source and target shapes
        if isinstance(self._src_geom, RectilinearGrid):
            src_shape = (len(self._src_geom.lats), len(self._src_geom.lons))
        elif isinstance(self._src_geom, CurvilinearGrid):
            src_shape = self._src_geom.lons.shape
        else:
            src_shape = (self._W_u.n_src // 2,)

        if isinstance(self._dst_geom, RectilinearGrid):
            dst_shape = (len(self._dst_geom.lats), len(self._dst_geom.lons))
        elif isinstance(self._dst_geom, CurvilinearGrid):
            dst_shape = self._dst_geom.lons.shape
        else:
            dst_shape = (self._W_u.n_dst,)

        n_spatial = self._W_u.n_src // 2

        if u_arr.ndim == len(src_shape) and u_arr.shape == src_shape:
            # 2D Spatial Grid Case
            uv_flat = np.concatenate([u_arr.ravel(), v_arr.ravel()])
            u_out = np.array(axis_py.apply_weights(self._W_u, uv_flat)).reshape(dst_shape)
            v_out = np.array(axis_py.apply_weights(self._W_v, uv_flat)).reshape(dst_shape)
        elif u_arr.ndim == 1:
            uv_flat = np.concatenate([u_arr, v_arr])
            u_out = np.array(axis_py.apply_weights(self._W_u, uv_flat))
            v_out = np.array(axis_py.apply_weights(self._W_v, uv_flat))
            if len(dst_shape) > 1:
                u_out = u_out.reshape(dst_shape)
                v_out = v_out.reshape(dst_shape)
        else:
            # Multidimensional Batch Case
            original_shape = u_arr.shape

            n_spatial_dims = len(src_shape)
            other_dims_shape = original_shape[:-n_spatial_dims]
            n_other = int(np.prod(other_dims_shape)) if other_dims_shape else 1

            flat_u = u_arr.reshape(n_other, n_spatial)
            flat_v = v_arr.reshape(n_other, n_spatial)
            flat_uv = np.concatenate([flat_u, flat_v], axis=1)  # shape: (n_other, 2 * n_spatial)

            flat_uv_t = np.asfortranarray(flat_uv.T)

            u_out_t = axis_py.batch_apply(self._W_u, flat_uv_t)
            v_out_t = axis_py.batch_apply(self._W_v, flat_uv_t)

            u_out = u_out_t.T.reshape(other_dims_shape + dst_shape)
            v_out = v_out_t.T.reshape(other_dims_shape + dst_shape)

        if is_xarray:
            target_coords = {}
            if hasattr(self._dst_geom, "lons"):
                if self._dst_geom.lons.ndim == 1:
                    target_coords["lat"] = self._dst_geom.lats
                    target_coords["lon"] = self._dst_geom.lons
                    dims = ("lat", "lon")
                else:
                    target_coords["lat"] = (("y", "x"), self._dst_geom.lats)
                    target_coords["lon"] = (("y", "x"), self._dst_geom.lons)
                    dims = ("y", "x")
            else:
                dims = ("ncol",)

            u_da = xr.DataArray(u_out, coords=target_coords, dims=dims, name=u.name, attrs=u.attrs)
            v_da = xr.DataArray(v_out, coords=target_coords, dims=dims, name=v.name, attrs=v.attrs)
            return u_da, v_da

        return u_out, v_out
