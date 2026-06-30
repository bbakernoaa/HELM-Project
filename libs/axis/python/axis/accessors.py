# SPDX-License-Identifier: Apache-2.0
from typing import Any, Union
import xarray as xr
from .regridder import Regridder

@xr.register_dataarray_accessor("axis")
class RegridDataArrayAccessor:
    """
    Xarray Accessor for regridding DataArrays.
    """
    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj

    def to(
        self, target_grid: Union[xr.Dataset, Regridder], **kwargs: Any
    ) -> xr.DataArray:
        """
        Regrid the DataArray to a target grid or using a pre-computed Regridder.
        """
        if isinstance(target_grid, Regridder):
            return target_grid(self._obj)

        source_ds = self._obj.to_dataset(name="_tmp_data")
        regridder = Regridder(source_ds, target_grid, **kwargs)
        return regridder(self._obj)

    def get_regridder(self, target_grid: xr.Dataset, **kwargs: Any) -> Regridder:
        """
        Create a Regridder instance for this DataArray.
        """
        source_ds = self._obj.to_dataset(name="_tmp_data")
        return Regridder(source_ds, target_grid, **kwargs)


@xr.register_dataset_accessor("axis")
class RegridDatasetAccessor:
    """
    Xarray Accessor for regridding Datasets.
    """
    def __init__(self, xarray_obj: xr.Dataset):
        self._obj = xarray_obj

    def to(
        self, target_grid: Union[xr.Dataset, Regridder], **kwargs: Any
    ) -> xr.Dataset:
        """
        Regrid the Dataset to a target grid or using a pre-computed Regridder.
        """
        if isinstance(target_grid, Regridder):
            return target_grid(self._obj)

        regridder = Regridder(self._obj, target_grid, **kwargs)
        return regridder(self._obj)

    def get_regridder(self, target_grid: xr.Dataset, **kwargs: Any) -> Regridder:
        """
        Create a Regridder instance for this Dataset.
        """
        return Regridder(self._obj, target_grid, **kwargs)
