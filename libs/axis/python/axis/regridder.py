# SPDX-License-Identifier: Apache-2.0
import numpy as np
import xarray as xr
import uuid
import warnings
from typing import Union, Optional, Tuple, Any

from . import axis_py
from .grid import create_axis_mesh, _get_mesh_info, _get_non_spatial_dims
from .core import _apply_weights_core, _setup_worker_cache, _sync_cache_from_worker_data

# Client-side cache on the driver node to avoid redundant worker cache syncs
_DRIVER_CACHE = {}

class Regridder:
    """
    Exascale-ready spatial regridder wrapping AXIS's compiled Kokkos-parallel engine.
    
    Seamlessly supports NumPy eager arrays, xarray Datasets/DataArrays, and Dask 
    distributed lazy arrays.
    """
    
    def __init__(
        self,
        ds_in: xr.Dataset,
        ds_out: xr.Dataset,
        method: str = "bilinear",
        periodic: bool = False,
        unmapped: str = "ignore",
        skipna: bool = False,
        na_thres: float = 1.0,
        line_type: str = "great_circle",
        weights_file: Optional[str] = None,
    ):
        """
        Initialize the regridder by generating spatial interpolation weights in C++,
        or by reloading pre-computed weights from a file.
        
        Parameters
        ----------
        ds_in : xarray.Dataset
            The source grid dataset.
        ds_out : xarray.Dataset
            The destination grid dataset.
        method : str, default "bilinear"
            Remapping method: "bilinear", "nearest", "bicubic", "patch", "conservative".
        periodic : bool, default False
            Whether the grid has periodic longitude boundaries wrapping 360 to 0.
        unmapped : str, default "ignore"
            Action on unmapped destination cells: "ignore" or "error".
        skipna : bool, default False
            If True, dynamically re-normalizes weights to handle NaNs.
        na_thres : float, default 1.0
            Minimum fraction of valid input contribution required to not mask output.
        line_type : str, default "great_circle"
            Line geometry: "great_circle" (spherical exact) or "cartesian" (Sutherland flat-clipping).
        weights_file : str, optional
            Path to a binary file containing pre-computed weights to reload.
        """
        self.method = method
        self.periodic = periodic
        self.unmapped = unmapped
        self.line_type = line_type
        self.skipna = skipna
        self.na_thres = na_thres
        self._uid = str(uuid.uuid4())

        # Extract coordinate information and dimensions
        _, _, self._shape_source, self._dims_source, self._is_unstructured_src = _get_mesh_info(ds_in, method)
        _, _, self._shape_target, self._dims_target, _ = _get_mesh_info(ds_out, method)
        
        # Save original datasets for coordinate matching
        self.source_grid_ds = ds_in
        self.target_grid_ds = ds_out

        if weights_file is not None:
            # Load pre-computed weights from file, completely bypassing weight generation
            with open(weights_file, "rb") as f:
                self._serialized_weights = f.read()
            self._weights_matrix = axis_py.Matrix.from_bytes(self._serialized_weights)
        else:
            # Map string method name to AXIS enum
            method_map = {
                "bilinear": axis_py.Method.Bilinear,
                "nearest": axis_py.Method.NearestNeighbor,
                "bicubic": axis_py.Method.Bicubic,
                "patch": axis_py.Method.Patch,
                "conservative": axis_py.Method.Conservative,
                "conservative2nd": axis_py.Method.Conservative2ndOrder,
            }
            if method.lower() not in method_map:
                raise ValueError(f"Unknown interpolation method: {method}. Choose from {list(method_map.keys())}")
            
            self._axis_method = method_map[method.lower()]

            # Map line_type to AXIS LineType enum
            line_type_map = {
                "great_circle": axis_py.LineType.GreatCircle,
                "cartesian": axis_py.LineType.Cartesian,
            }
            if line_type.lower() not in line_type_map:
                raise ValueError(f"Unknown line type: {line_type}. Choose from {list(line_type_map.keys())}")
            self._axis_line_type = line_type_map[line_type.lower()]

            # Generate on-device meshes and weights using C++ AXIS
            self._src_mesh = create_axis_mesh(ds_in, method)
            self._dst_mesh = create_axis_mesh(ds_out, method)

            config = {
                "method": self._axis_method,
                "periodic": periodic,
                "line_type": self._axis_line_type,
                "unmapped": axis_py.UnmappedAction.Ignore if unmapped == "ignore" else axis_py.UnmappedAction.Error
            }

            # Build sparse matrix weights
            self._weights_matrix = axis_py.generate_weights(self._src_mesh, self._dst_mesh, config)
            
            # Serialize C++ weights to bytes so they can be securely copied to remote Dask workers
            self._serialized_weights = self._weights_matrix.to_bytes()

        # Compute total weights sum for NaN-aware re-normalization
        if self.skipna:
            self._total_weights = np.array(axis_py.apply_weights(self._weights_matrix, np.ones(self._weights_matrix.n_src))).flatten()
        else:
            self._total_weights = None

    def to_file(self, filename: str) -> None:
        """
        Save the compiled sparse regridding weights to a binary file for future reuse.
        
        Parameters
        ----------
        filename : str
            Path where the weights binary file will be saved.
        """
        with open(filename, "wb") as f:
            f.write(self._serialized_weights)

    def __call__(
        self,
        obj: Union[xr.DataArray, xr.Dataset],
        keep_attrs: bool = True,
    ) -> Union[xr.DataArray, xr.Dataset]:
        """
        Apply spatial remapping to the input xarray object.
        """
        if isinstance(obj, xr.Dataset):
            res = self._regrid_dataset(obj)
            if keep_attrs:
                res.attrs = {**obj.attrs, **res.attrs}
            return res
        elif isinstance(obj, xr.DataArray):
            res = self._regrid_dataarray(obj)
            if keep_attrs:
                res.attrs = {**obj.attrs, **res.attrs}
            return res
        else:
            raise TypeError("Input object must be an xarray.DataArray or xarray.Dataset")

    def _regrid_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Remap all spatial variables inside a Dataset."""
        regridded_vars = {}
        for var in ds.data_vars:
            da = ds[var]
            if all(dim in da.dims for dim in self._dims_source):
                regridded_vars[var] = self._regrid_dataarray(da)
            else:
                regridded_vars[var] = da

        res = xr.Dataset(regridded_vars)
        # Inherit non-spatial global coordinates
        for c in ds.coords:
            if c not in res.coords and set(ds[c].dims).intersection(set(self._dims_source)) == set():
                res = res.assign_coords({c: ds[c]})
        return res

    def _regrid_dataarray(self, da_in: xr.DataArray) -> xr.DataArray:
        """Remap a single DataArray, including Dask-lazy execution paths."""
        # Detect if input array is backed by Dask
        is_dask = hasattr(da_in.data, "dask")

        input_core_dims = list(self._dims_source)
        temp_output_core_dims = [f"{d}_regridded" for d in self._dims_target]

        weights_arg = self._weights_matrix
        total_weights_arg = self._total_weights
        weights_key_arg = None

        if is_dask:
            # Sync C++ compiled weights to remote Dask workers via serialization
            try:
                import dask.distributed
                client = dask.distributed.get_client()
            except (ImportError, ValueError):
                client = None

            weights_key_arg = f"weights_{self._uid}"

            if client is not None:
                client_id = getattr(client, "id", id(client))

                # If workers are not synchronized, replicate the serialized bytes and deserialize
                if (client_id, weights_key_arg) not in _DRIVER_CACHE:
                    # Deserialize directly inside the worker node's memory space (zero copying)
                    def _deserialize_and_cache(bytes_data, key):
                        from . import axis_py
                        mat = axis_py.Matrix.from_bytes(bytes_data)
                        _setup_worker_cache(key, mat)
                        return True

                    client.run(_deserialize_and_cache, self._serialized_weights, weights_key_arg)
                    _DRIVER_CACHE[(client_id, weights_key_arg)] = True

                weights_arg = weights_key_arg

                if self._total_weights is not None:
                    tw_key = f"tw_{self._uid}_sum"
                    if (client_id, tw_key) not in _DRIVER_CACHE:
                        client.run(_setup_worker_cache, tw_key, self._total_weights)
                        _DRIVER_CACHE[(client_id, tw_key)] = True
                    total_weights_arg = tw_key
            else:
                # Fallback: register locally in our _WORKER_CACHE so that local multiprocessing or thread schedulers
                # can lookup the weights matrix by string key, completely avoiding PicklingErrors!
                _setup_worker_cache(weights_key_arg, self._weights_matrix)
                weights_arg = weights_key_arg

                if self._total_weights is not None:
                    tw_key = f"tw_{self._uid}_sum"
                    _setup_worker_cache(tw_key, self._total_weights)
                    total_weights_arg = tw_key

        # Execute parallelized map-blocks SpMV using apply_ufunc
        out = xr.apply_ufunc(
            _apply_weights_core,
            da_in,
            kwargs={
                "weights_matrix": weights_arg,
                "dims_source": self._dims_source,
                "shape_target": self._shape_target,
                "skipna": self.skipna,
                "total_weights": total_weights_arg,
                "na_thres": self.na_thres,
                "weights_key": weights_key_arg,
            },
            input_core_dims=[input_core_dims],
            output_core_dims=[temp_output_core_dims],
            dask="parallelized",
            vectorize=False,
            output_dtypes=[da_in.dtype],
            dask_gufunc_kwargs={
                "output_sizes": {
                    d: s for d, s in zip(temp_output_core_dims, self._shape_target)
                },
                "allow_rechunk": True,
            },
        )

        # Rename temporary output dimensions to target dimension names
        rename_dict = {temp: orig for temp, orig in zip(temp_output_core_dims, self._dims_target) if temp in out.dims}
        if rename_dict:
            out = out.rename(rename_dict)

        # Assign coordinates from target grid
        target_coords = {}
        for c in self.target_grid_ds.coords:
            c_dims = set(self.target_grid_ds.coords[c].dims)
            if c_dims.issubset(set(self._dims_target)):
                target_coords[c] = self.target_grid_ds.coords[c]

        out = out.assign_coords(target_coords)
        return out
