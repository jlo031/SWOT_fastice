# ---- This is <make_geotiff_from_SWOT.py> ----

"""
Make geotiff files from SWOT input data.
SWOT nc file pre-processed by James Wyatt.

    Default feature to geocode: ssha_HR, sig0_karin_2, total_coherence
    
Changes required if this script is to be adapted to other SWOT data.
"""

import argparse
import sys
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

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

def set_loglevel(level: str):
    level = level.upper()
    if level not in ["TRACE", "DEBUG", "INFO", "SUCCESS" "WARNING", "ERROR", "CRITICAL"]:
        raise ValueError(f"Invalid log level: {level}")
    logger.remove()
    logger.add(sink=sys.stdout, level=level)

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

def main():

    p = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter, description =__doc__)

    p.add_argument("swot_basename", help = "SWOT file basename")
    p.add_argument("-epsg", default="3031", help = "target epsg code (default=3031)")
    p.add_argument("-pixel_spacing", default="250", help = "target pixel spacing (default=250)")
    p.add_argument("-tie_points", default="21", help = "number of GCPs per dimension (default=21)")
    p.add_argument("-overwrite", action = "store_true", help = "overwrite existing files")
    p.add_argument('-loglevel', choices = ["TRACE", "DEBUG", "INFO", "SUCCESS" "WARNING", "ERROR", "CRITICAL"], default = "INFO", help = "loglevel setting (default=INFO)")

    args = p.parse_args()

    # set loglevel
    try:
        set_loglevel(args.loglevel)
    except ValueError as e:
        print(e)
        return

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

    logger.debug(f"args: {args}")

    # Parse inputs
    swot_basename = args.swot_basename
    epsg          = args.epsg
    pixel_spacing = args.pixel_spacing
    tie_points    = int(args.tie_points)
    overwrite     = args.overwrite
    loglevel      = args.loglevel

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

    logger.info(f"Processing swot_file: {swot_basename}")

    swot_path_ascending  = SWOT_SWATHS_ssha_sig0_coherence / "ascending" / f"{swot_basename}.nc"
    swot_path_descending = SWOT_SWATHS_ssha_sig0_coherence / "descending" / f"{swot_basename}.nc"

    if swot_path_ascending.is_file():
        swot_path = swot_path_ascending
        logger.debug("ascending path")
    elif swot_path_descending.is_file():
        swot_path = swot_path_descending
        logger.debug("descending path")
    else:
        logger.error(f"Could not find swot_path for descending or ascending")
        logger.error(f"swot_path_ascending: {swot_path_ascending}")
        logger.error(f"swot_path_descending: {swot_path_ascending}")
        return

    # Build final output file names with date
    sig0_1_output = SWOT_GEOTIFF_DIR / "tmp" / f"{swot_basename}_sig0_1_epsg{epsg}_pixelspacing{pixel_spacing}.tiff"
    sig0_2_output = SWOT_GEOTIFF_DIR / f"{swot_basename}_sig0_2_epsg{epsg}_pixelspacing{pixel_spacing}.tiff"
    ssha_1_output = SWOT_GEOTIFF_DIR / f"{swot_basename}_ssha_1_epsg{epsg}_pixelspacing{pixel_spacing}.tiff"
    ssha_2_output = SWOT_GEOTIFF_DIR / f"{swot_basename}_ssha_2_epsg{epsg}_pixelspacing{pixel_spacing}.tiff"
    coh_1_output  = SWOT_GEOTIFF_DIR / f"{swot_basename}_coh_1_epsg{epsg}_pixelspacing{pixel_spacing}.tiff"
    coh_2_output  = SWOT_GEOTIFF_DIR / f"{swot_basename}_coh_2_epsg{epsg}_pixelspacing{pixel_spacing}.tiff"

    logger.debug(f"swot_path:     {swot_path}")
    logger.debug(f"swot_basename: {swot_basename}")

# --------------------------------------------------------------------------- #

    # Read SWOT variables
    with xr.open_dataset(swot_path, engine="netcdf4") as ds:
        
        # Read SWOT lat/lon, sigma0, ssha, coh
        lat  = ds["latitude"].values
        lon  = ds["longitude"].values
        ssha = ds["ssha_HR"].values
        sig0 = ds["sig0_karin_2"].values
        coh  = ds["total_coherence"].values

    logger.debug("Read data from input file")

# --------------------------------------------------------------------------- #

    # Current data contains left and right beam in the same array
    # For correct geocoding and further processing, left and right are split
    # Splitting index is determined from largest jump in lat/lon

    # Find the step in lat and lon
    lat_mean = np.nanmean(lat,1)
    lon_mean = np.nanmean(lon,1)

    lat_max_diff     = np.max(np.abs(np.diff(lat_mean)))
    lon_max_diff     = np.max(np.abs(np.diff(lon_mean)))
    lat_max_diff_idx = np.argmax(np.abs(np.diff(lat_mean)))
    lon_max_diff_idx = np.argmax(np.abs(np.diff(lon_mean)))

    if lat_max_diff_idx != lon_max_diff_idx:
        logger.error(f"Did not find same index for max step in lat and lon.")
        return

    split_idx = lat_max_diff_idx + 1
    logger.debug(f"lat.shape: {lat.shape}")
    logger.debug(f"split_idx: {split_idx}")

    lat_1  = lat[0:split_idx,:]
    lon_1  = lon[0:split_idx,:]
    ssha_1 = ssha[0:split_idx,:]
    sig0_1 = sig0[0:split_idx,:]
    coh_1  = coh[0:split_idx,:]

    lat_2  = lat[split_idx:,:]
    lon_2  = lon[split_idx:,:]
    ssha_2 = ssha[split_idx:,:]
    sig0_2 = sig0[split_idx:,:]
    coh_2  = coh[split_idx:,:]

# --------------------------------------------------------------------------- #

    # Embed the GCPs and do the warping with gdal
    # Could be implemented better, but this works

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

    # Embed GCPs into temporary tiff files for sig0_1 and sig0_2
    geo_utils.embed_tie_points_in_array_to_tiff(
        sig0_1,
        gcp_list_1,
        "sig0_1.tiff",
        tie_point_WKT_1,
        loglevel=loglevel,
    )
    geo_utils.embed_tie_points_in_array_to_tiff(
        sig0_2,
        gcp_list_2,
        "sig0_2.tiff",
        tie_point_WKT_1,
        loglevel=loglevel,
    )

    # Embed GCPs into temporary tiff files for ssha_1 and ssha_2
    geo_utils.embed_tie_points_in_array_to_tiff(
        ssha_1,
        gcp_list_1,
        "ssha_1.tiff",
        tie_point_WKT_1,
        loglevel=loglevel,
    )
    geo_utils.embed_tie_points_in_array_to_tiff(
        ssha_2,
        gcp_list_2,
        "ssha_2.tiff",
        tie_point_WKT_1,
        loglevel=loglevel,
    )

    # Embed GCPs into temporary tiff files for coh_1 and coh_2
    geo_utils.embed_tie_points_in_array_to_tiff(
        coh_1,
        gcp_list_1,
        "coh_1.tiff",
        tie_point_WKT_1,
        loglevel=loglevel,
    )
    geo_utils.embed_tie_points_in_array_to_tiff(
        coh_2,
        gcp_list_2,
        "coh_2.tiff",
        tie_point_WKT_1,
        loglevel=loglevel,
    )

    # Warp tiff files for sig0_1 and sig0_2 to final projection
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

    # Warp tiff files for ssha_1 and ssha_2 to final projection
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

    # Warp tiff files for coh_1 and coh_2 to final projection
    geo_utils.warp_image_to_target_projection(
        "coh_1.tiff",
        coh_1_output,
        epsg,
        pixel_spacing,
        srcnodata = 0,
        dstnodata = 0,
        resampling = 'near',
        order = 3,
        loglevel = loglevel,
    )
    geo_utils.warp_image_to_target_projection(
        "coh_2.tiff",
        coh_2_output,
        epsg,
        pixel_spacing,
        srcnodata = 0,
        dstnodata = 0,
        resampling = 'near',
        order = 3,
        loglevel = loglevel,
    )

    os.remove("sig0_1.tiff")
    os.remove("sig0_2.tiff")
    os.remove("ssha_1.tiff")
    os.remove("ssha_2.tiff")
    os.remove("coh_1.tiff")
    os.remove("coh_2.tiff")

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    main()
    
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# ---- End of <make_geotiff_from_SWOT.py> ----


"""
plt.figure()
plt.plot(lat_mean,'r')
plt.plot(lon_mean,'g')

plt.figure()
plt.plot(np.abs(np.diff(lat_mean)),'r')
plt.plot(np.abs(np.diff(lon_mean)),'g')

plt.show()
"""

