# Spec: AXIS Fractional & Categorical Regridding Support

## 1. Overview & Motivation
In Earth System Modeling (ESM) and land-surface modeling, it is extremely common to require remapping of high-resolution categorical datasets (such as a $1\text{ km}$ land-use classification grid with IDs representing Forest, Water, Urban, etc.) to a coarser destination grid, producing the **exact fractional coverage of each category** inside each coarse-resolution grid cell.

This specification defines the implementation of a new Python-level feature `regrid_categorical` in the AXIS `Regridder` class. This automatically builds binary category indicator masks, remaps them using AXIS's exact area-conservative clipping, and compiles the resulting fractional coverage fields into a single, unified `xarray.Dataset`.

---

## 2. Design & Python API

We will add a new method `regrid_categorical` to the `Regridder` class in `libs/axis/python/axis/regridder.py`:

```python
    def regrid_categorical(
        self,
        da_in: xr.DataArray,
        categories: Optional[Union[list, dict]] = None,
        prefix: str = "fraction_",
    ) -> xr.Dataset:
        """
        Remap high-resolution categorical data (e.g., land-use classes) to coarse grid fractions.

        Parameters
        ----------
        da_in : xr.DataArray
            Categorical input DataArray containing integer class codes or IDs.
        categories : list or dict, optional
            If a list, computes fractions only for the specified category IDs.
            If a dict, maps category ID (key) to its corresponding string name (value)
            for the output variables.
            If None (default), automatically discovers all unique non-NaN category IDs.
        prefix : str, default "fraction_"
            Prefix to prepend to the output variable names (e.g. "fraction_forest").

        Returns
        -------
        xr.Dataset
            A dataset containing the fractional coverage for each category as separate variables.
        """
```

### Ingestion Flow:
1.  **Category Identification:**
    *   If `categories` is None, discover unique values:
        `unique_vals = np.unique(da_in.values)` (excluding NaNs and padded indices).
    *   If `categories` is a list, use its elements.
    *   If `categories` is a dict, use its keys.
2.  **Fractional Remapping Loop:**
    For each category $c$:
    *   Construct binary indicator: `indicator = (da_in == c).astype(float)`
    *   Remap using conservative (or nearest) interpolation: `fraction_da = self(indicator)`
    *   Name the variable:
        *   If `categories` is a dict: `var_name = f"{prefix}{categories[c]}"`
        *   Otherwise: `var_name = f"{prefix}{c}"`
    *   Store in output map.
3.  **Return Dataset:**
    Combine all fractional variables into a single `xr.Dataset`.

---

## 3. Verification & Testing Strategy

*   **Unit Tests:**
    *   Add a test `test_regridder_categorical` in `libs/axis/tests_python/test_axis_regridder.py`.
    *   Create a categorical source array representing forest and water.
    *   Remap using `regrid_categorical` and verify that the output variables contain the correct area fractions (summing to 1.0).
