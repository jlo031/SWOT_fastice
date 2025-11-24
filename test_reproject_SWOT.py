import pathlib
import os
from loguru import logger

import netCDF4
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

import geocoding.generic_geocoding as gen_geo
import geocoding.geocoding_utils as geo_utils

from config.load_config import *


##L2_D_Unsmoothed_cropped


loglevel = "DEBUG"
tie_points = 21
epsg = 3031
pixel_spacing = 250


swot_dir = SWOT_L2_D_Unsmoothed_cropped

swot_files_cropped   = [ f for f in os.listdir(SWOT_L2_D_Unsmoothed_cropped) if f.endswith(".nc")]
swot_files_uncropped = [ f for f in os.listdir(SWOT_L2_D_Unsmoothed) if f.endswith(".nc")]





swot_file = swot_files_cropped[0]









good_swot_dir = pathlib.Path("/g/data/jk72/jw2777/SWOT_ICE/DATA/SWOT/swot_swaths_perpass/ascending_v2")
swot_file_list = [ f for f in os.listdir(good_swot_dir) if f.endswith(".nc")]
swot_file = swot_file_list[3]
swot_path = good_swot_dir / swot_file
swot_base = swot_path.stem
ds = xr.open_dataset(swot_path, engine="netcdf4")

# Read SWOT lat/lon into arrays
lat = ds["latitude"].values
lon = ds["longitude"].values

# Read SWOT ssha and sig0
ssha = ds["ssha_HR"].values
sig0 = ds["sig0_karin_2"].values


# Find the step in lat and lon
lat_mean = np.nanmean(lat,1)
lon_mean = np.nanmean(lon,1)

lat_max_diff     = np.max(np.abs(np.diff(lat_mean)))
lon_max_diff     = np.max(np.abs(np.diff(lon_mean)))
lat_max_diff_idx = np.argmax(np.abs(np.diff(lat_mean)))
lon_max_diff_idx = np.argmax(np.abs(np.diff(lon_mean)))


if lat_max_diff_idx != lon_max_diff_idx:
    logger.error(f"Did not find same index for max step in lat and lon.")
else:
    split_idx = lat_max_diff_idx + 1


lat_1  = lat[0:split_idx,:]
lon_1  = lon[0:split_idx,:]
ssha_1 = ssha[0:split_idx,:]
sig0_1 = sig0[0:split_idx,:]

lat_2  = lat[split_idx:,:]
lon_2  = lon[split_idx:,:]
ssha_2 = ssha[split_idx:,:]
sig0_2 = sig0[split_idx:,:]




# Get GCPs and tie_point_WKT from lat lon bands
gcp_list_1, tie_point_WKT_1 = geo_utils.get_tie_points_from_lat_lon(
    lat_1,
    lon_1,
    tie_points = tie_points,
    loglevel = loglevel
)
gcp_list_2, tie_point_WKT_2 = geo_utils.get_tie_points_from_lat_lon(
    lat_2,
    lon_2,
    tie_points = tie_points,
    loglevel = loglevel
)

# Embed GCPs into temporary tiff file for sig0_1
geo_utils.embed_tie_points_in_array_to_tiff(
    sig0_1,
    gcp_list_1,
    "sig0_1.tiff",
    tie_point_WKT_1,
    loglevel=loglevel,
)

# Embed GCPs into temporary tiff file for sig0_2
geo_utils.embed_tie_points_in_array_to_tiff(
    sig0_2,
    gcp_list_2,
    "sig0_2.tiff",
    tie_point_WKT_2,
    loglevel=loglevel,
)


# Embed GCPs into temporary tiff file for ssha_1
geo_utils.embed_tie_points_in_array_to_tiff(
    ssha_1,
    gcp_list_1,
    "ssha_1.tiff",
    tie_point_WKT_1,
    loglevel=loglevel,
)

# Embed GCPs into temporary tiff file for ssha_2
geo_utils.embed_tie_points_in_array_to_tiff(
    ssha_2,
    gcp_list_2,
    "ssha_2.tiff",
    tie_point_WKT_2,
    loglevel=loglevel,
)



sig0_1_output = f"{swot_base}_sig0_1_epsg{epsg}_pixelspacing{pixel_spacing}.tiff"
sig0_2_output = f"{swot_base}_sig0_2_epsg{epsg}_pixelspacing{pixel_spacing}.tiff"

ssha_1_output = f"{swot_base}_ssha_1_epsg{epsg}_pixelspacing{pixel_spacing}.tiff"
ssha_2_output = f"{swot_base}_ssha_2_epsg{epsg}_pixelspacing{pixel_spacing}.tiff"


# Warp tiff file to final projection
geo_utils.warp_image_to_target_projection(
    "sig0_1.tiff",
    sig0_1_output,
    epsg,
    pixel_spacing,
    srcnodata = 0,
    dstnodata = 0,
    resampling = 'near',
    order = 3,
    loglevel = loglevel,
)
# Warp tiff file to final projection
geo_utils.warp_image_to_target_projection(
    "sig0_2.tiff",
    sig0_2_output,
    epsg,
    pixel_spacing,
    srcnodata = 0,
    dstnodata = 0,
    resampling = 'near',
    order = 3,
    loglevel = loglevel,
)

# Warp tiff file to final projection
geo_utils.warp_image_to_target_projection(
    "ssha_1.tiff",
    ssha_1_output,
    epsg,
    pixel_spacing,
    srcnodata = 0,
    dstnodata = 0,
    resampling = 'near',
    order = 3,
    loglevel = loglevel,
)
# Warp tiff file to final projection
geo_utils.warp_image_to_target_projection(
    "ssha_2.tiff",
    ssha_2_output,
    epsg,
    pixel_spacing,
    srcnodata = 0,
    dstnodata = 0,
    resampling = 'near',
    order = 3,
    loglevel = loglevel,
)















plt.plot(lat_mean,'r')
plt.plot(lon_mean,'g')
plt.show()

plt.plot(np.abs(np.diff(lat_mean),'r')
plt.plot(np.absnp.diff(lon_mean),'g')
plt.show()

idx_step_lat = lat_mean.diff


dims = lat.shape

lat_1 = lat[0:dim
















# Build full SWOT path
swot_path_cropped = SWOT_L2_D_Unsmoothed_cropped / swot_file
swot_path_uncropped = SWOT_L2_D_Unsmoothed / swot_file

# get SWOT basename
swot_base = swot_path.stem


ds_cropped   = xr.open_dataset(swot_path_cropped, engine="netcdf4")
ds_uncropped = xr.open_dataset(swot_path_uncropped, engine="netcdf4")




dset = netCDF4.Dataset(swot_path_uncropped, 'r')



# Read SWOT lat/lon into arrays
lat = ds_cropped["latitude"].values
lon = ds_cropped["longitude"].values

# Read SWOT ssh and sig0
ssh  = ds_cropped["ssh_karin_2"].values
sig0 = ds_cropped["sig0_karin_2"].values

# Read mean ssh
mean_ssh  = ds_cropped["mean_sea_surface_cnescls"].values

# Convert sig0 to dB
sig0_dB = 10*np.log10(sig0)

# Subtract mean from ssh
ssh_anomaly = ssh - mean_ssh



ds_uncropped["left"]





ds_new = 

for swot_file_ in swot_file_list:
    logger.info(f"Processing swot file: {swot_file}")
    
    # Build full SWOT path
    swot_path = swot_dir / swot_file

    # get SWOT basename
    swot_base = swot_path.stem

    # Read SWOT variables
    with xr.open_dataset(swot_path, engine="netcdf4") as ds:
        
        # Read SWOT lat/lon into arrays
        lat = ds["latitude"].values
        lon = ds["longitude"].values

        # Read SWOT ssh and sig0
        ssh  = ds["ssh_karin_2"].values
        sig0 = ds["sig0_karin_2"].values

        # Read mean ssh
        mean_ssh  = ds["mean_sea_surface_cnescls"].values



    # Convert sig0 to dB
    sig0_dB = 10*np.log10(sig0)

    # Subtract mean from ssh
    ssh_anomaly = ssh - mean_ssh


    fig, axes = plt.subplots(1,4,sharex=True,sharey=True)
    axes = axes.ravel()
    axes[0].imshow(ssha, vmin=-0.5,vmax=0.5)
    axes[1].imshow(sig0, vmin=-10, vmax=15)
    axes[2].imshow(lat)
    axes[3].imshow(lon)
    plt.show()
    """



    # Get GCPs and tie_point_WKT from lat lon bands
    gcp_list, tie_point_WKT = geo_utils.get_tie_points_from_lat_lon(
        lat,
        lon,
        tie_points = tie_points,
        loglevel = loglevel
    )

    # Embed GCPs into temporary tiff file for sig
    geo_utils.embed_tie_points_in_array_to_tiff(
        sig0_dB,
        gcp_list,
        "sig0_dB_tmp.tiff",
        tie_point_WKT,
        loglevel=loglevel,
    )

    # Embed GCPs into temporary tiff file for ssh_anomaly
    geo_utils.embed_tie_points_in_array_to_tiff(
        ssh_anomaly,
        gcp_list,
        "ssh_anomaly_tmp.tiff",
        tie_point_WKT,
        loglevel=loglevel,
    )

    sig0_output = WORK_DIR / f"{swot_base}_sig0_dB_epsg{epsg}_pixelspacing{pixel_spacing}.tiff"
    ssh_output = WORK_DIR / f"{swot_base}_ssh_anomaly_epsg{epsg}_pixelspacing{pixel_spacing}.tiff"


    # Warp tiff file to final projection
    geo_utils.warp_image_to_target_projection(
        "sig0_dB_tmp.tiff",
        sig0_output,
        epsg,
        pixel_spacing,
        srcnodata = 0,
        dstnodata = 0,
        resampling = 'near',
        order = 3,
        loglevel = loglevel,
    )

    # Warp tiff file to final projection
    geo_utils.warp_image_to_target_projection(
        "ssh_anomaly_tmp.tiff",
        ssh_output,
        epsg,
        pixel_spacing,
        srcnodata = 0,
        dstnodata = 0,
        resampling = 'near',
        order = 3,
        loglevel = loglevel,
    )

