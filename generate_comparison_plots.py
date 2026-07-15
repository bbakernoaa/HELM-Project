import os
import sys
import numpy as np
import xarray as xr
import subprocess
import tempfile
import matplotlib.pyplot as plt
import axis

def create_test_dataset(nlat, nlon):
    lats = np.linspace(-90.0, 90.0, nlat)
    lons = np.linspace(0.0, 360.0, nlon, endpoint=False)
    lon2d, lat2d = np.meshgrid(lons, lats)
    field = np.cos(np.radians(lat2d)) * np.cos(np.radians(lon2d))

    # Calculate bounds
    dlat = lats[1] - lats[0]
    dlon = lons[1] - lons[0]
    lat_bnds = np.zeros((len(lats), 2))
    lat_bnds[:, 0] = lats - 0.5 * dlat
    lat_bnds[:, 1] = lats + 0.5 * dlat
    lat_bnds = np.clip(lat_bnds, -90.0, 90.0)

    lon_bnds = np.zeros((len(lons), 2))
    lon_bnds[:, 0] = lons - 0.5 * dlon
    lon_bnds[:, 1] = lons + 0.5 * dlon

    ds = xr.Dataset(
        {"temperature": (["lat", "lon"], field.astype(np.float64))},
        coords={"lat": lats, "lon": lons},
    )
    ds["lat_bnds"] = (["lat", "bnds"], lat_bnds)
    ds["lon_bnds"] = (["lon", "bnds"], lon_bnds)
    ds["lat"].attrs = {"units": "degrees_north", "axis": "Y", "bounds": "lat_bnds", "standard_name": "latitude"}
    ds["lon"].attrs = {"units": "degrees_east", "axis": "X", "bounds": "lon_bnds", "standard_name": "longitude"}
    ds["temperature"].attrs = {"units": "K", "long_name": "Test field"}
    return ds

def generate_plots():
    ds_in = create_test_dataset(90, 180)

    dst_lons = np.linspace(0.0, 360.0, 360, endpoint=False)
    dst_lats = np.linspace(-90.0, 90.0, 180)
    ds_out = xr.Dataset({
        "lat": (["lat"], dst_lats),
        "lon": (["lon"], dst_lons)
    })

    dlat = dst_lats[1] - dst_lats[0]
    dlon = dst_lons[1] - dst_lons[0]
    lat_bnds = np.zeros((len(dst_lats), 2))
    lat_bnds[:, 0] = dst_lats - 0.5 * dlat
    lat_bnds[:, 1] = dst_lats + 0.5 * dlat
    lat_bnds = np.clip(lat_bnds, -90.0, 90.0)
    lon_bnds = np.zeros((len(dst_lons), 2))
    lon_bnds[:, 0] = dst_lons - 0.5 * dlon
    lon_bnds[:, 1] = dst_lons + 0.5 * dlon

    ds_out["lat_bnds"] = (["lat", "bnds"], lat_bnds)
    ds_out["lon_bnds"] = (["lon", "bnds"], lon_bnds)
    ds_out["lat"].attrs = {"units": "degrees_north", "axis": "Y", "bounds": "lat_bnds", "standard_name": "latitude"}
    ds_out["lon"].attrs = {"units": "degrees_east", "axis": "X", "bounds": "lon_bnds", "standard_name": "longitude"}
    ds_out["temperature"] = (["lat", "lon"], np.zeros((len(dst_lats), len(dst_lons))))

    with tempfile.TemporaryDirectory() as tmpdir:
        src_nc = os.path.join(tmpdir, "src.nc")
        ds_in.to_netcdf(src_nc)

        target_grid_nc = os.path.join(tmpdir, "target.nc")
        ds_out.to_netcdf(target_grid_nc)

        methods = [("conservative", "remapcon"), ("bilinear", "remapbil"), ("nearest", "remapnn")]

        for method, cdo_op in methods:
            cdo_out = os.path.join(tmpdir, f"cdo_{method}.nc")
            cmd = ["cdo", "-O", "-s", f"-{cdo_op},{target_grid_nc}", src_nc, cdo_out]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            cdo_ds = xr.open_dataset(cdo_out)
            cdo_result = cdo_ds["temperature"].values

            axis_regridder = axis.Regridder(ds_in, ds_out, method=method, periodic=True)
            axis_out = axis_regridder(ds_in["temperature"])
            axis_result = axis_out.values

            diff = axis_result - cdo_result

            # Create side-by-side plots
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # AXIS plot
            im0 = axes[0].imshow(axis_result, extent=[0, 360, -90, 90], origin='lower', cmap='viridis')
            axes[0].set_title(f"AXIS {method.capitalize()}")
            axes[0].set_xlabel("Longitude")
            axes[0].set_ylabel("Latitude")
            fig.colorbar(im0, ax=axes[0], orientation='horizontal', label="Value")

            # CDO plot
            im1 = axes[1].imshow(cdo_result, extent=[0, 360, -90, 90], origin='lower', cmap='viridis')
            axes[1].set_title(f"CDO {method.capitalize()}")
            axes[1].set_xlabel("Longitude")
            axes[1].set_ylabel("Latitude")
            fig.colorbar(im1, ax=axes[1], orientation='horizontal', label="Value")

            # Difference plot
            im2 = axes[2].imshow(diff, extent=[0, 360, -90, 90], origin='lower', cmap='bwr')
            axes[2].set_title(f"Difference (AXIS - CDO)")
            axes[2].set_xlabel("Longitude")
            axes[2].set_ylabel("Latitude")
            fig.colorbar(im2, ax=axes[2], orientation='horizontal', label="Difference")

            plt.suptitle(f"Interpolation Comparison: {method.capitalize()} Method", fontsize=16)
            plt.tight_layout()

            # Save the plot
            plot_path = f"{method}_comparison.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Generated comparison plot for {method}: {plot_path}")

if __name__ == "__main__":
    generate_plots()
