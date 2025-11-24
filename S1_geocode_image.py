# ---- This is <S1_geocode_image.py> ----

"""
Geocode features from S1 input image.
Folder structure is read from config folder.

    Default feature to geocode: Sigma0_HH_dB, Sigma0_HV_dB, IA
"""

import pathlib
import sys
import argparse

from loguru import logger

import geocoding.generic_geocoding as gen_geo
import geocoding.geocoding_utils as geo_utils
import S1_processing.S1_feature_extraction as S1_feat
import S1_processing.utils as S1_utils

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
    
    p.add_argument("S1_base", help = "S1 basename")
    p.add_argument("-ML", default="5x5", help = "multilook window size (default=5x5)")
    p.add_argument("-epsg", default="3031", help = "target epsg code (default=3031)")
    p.add_argument("-pixel_spacing", default="80", help = "target pixel spacing (default=80)")
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
    S1_base       = args.S1_base
    ML            = args.ML
    epsg          = args.epsg
    pixel_spacing = args.pixel_spacing
    overwrite     = args.overwrite
    loglevel      = args.loglevel

    logger.info(f"Processing image: {S1_base}")

    # Set GPT variable from local .env file
    GPT = S1_utils.get_GPT_path('local')
    logger.info(f"GPT: {GPT}")

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

    logger.debug(f"DATA_DIR:    {DATA_DIR}")
    logger.debug(f"S1_DIR:      {S1_DIR}")
    logger.debug(f"S1_L1_DIR:   {S1_L1_DIR}")
    logger.debug(f"S1_FEAT_DIR: {S1_FEAT_DIR}")
    logger.debug(f"S1_GEO_DIR:  {S1_GEO_DIR}")

    # Build path to feature folder and check that it exists
    feat_folder = S1_FEAT_DIR / f"ML_{ML}" / f"{S1_base}"
    if not feat_folder.is_dir():
        logger.error(f"Could not find feat_folder: {feat_folder}")
        return

    logger.info(f"feat_folder: {feat_folder}")


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

    features_2_geocode = [
        "Sigma0_HH_dB",
        "Sigma0_HV_dB",
        "IA",
    ]

# --------------------------------------------------------------------------- #

    for feature in features_2_geocode:

        logger.info(f"Geocoding: {feature}")

        # Build input and output paths
        feat_path   = feat_folder / f"{feature}.img"
        lat_path    = feat_folder / f"lat.img"
        lon_path    = feat_folder / f"lon.img"
        output_path = S1_GEO_DIR / f"{S1_base}_{feature}_epsg{epsg}_pixelspacing{pixel_spacing}.tif"

        logger.debug(f"feat_path:   {feat_path}")
        logger.debug(f"lat_path:    {lat_path}")
        logger.debug(f"lon_path:    {lon_path}")
        logger.debug(f"output_path: {output_path}")

        # Check if output already exists
        if output_path.is_file() and not overwrite:
            logger.info("Output file already exists.")
            continue

        # Check that lat/lon files exist, try to extract if needed (needs S1_processing installation anf GPT)
        if not lat_path.is_file() or not lon_path.is_file():
            logger.info(f"Could not find lat/lon files. Trying to extract (S1_processing)")

            # Build path to S1 zip or safe and check that it exists
            S1_zip  = S1_L1_DIR / f"{S1_base}.zip"
            S1_safe = S1_L1_DIR / f"{S1_base}.SAFE"
            if S1_zip.is_file():
                S1_safe_zip = S1_zip
            elif S1_safe.is_dir():
                S1_safe_zip = S1_safe
            else:
                logger.error(f"Could not find S1_zip:  {S1_zip}")
                logger.error(f"Could not find S1_safe: {S1_safe}")
                return
            logger.info(f"S1_safe_zip: {S1_safe_zip}")


            S1_feat.get_S1_lat_lon(
                S1_safe_zip, 
                feat_folder,
                GPT,
                loglevel = loglevel,
                overwrite = overwrite
            )

        # Geocode feature
        gen_geo.geocode_image_from_lat_lon(
            feat_path,
            lat_path,
            lon_path,
            output_path,
            epsg,
            pixel_spacing,
            tie_points = 21,
            srcnodata = 0,
            dstnodata = 0,
            order = 3,
            resampling = 'near',
            keep_gcp_file = False,
            overwrite = overwrite,
            loglevel = loglevel,
        )


# --------------------------------------------------------------------------- #

    # Stack intensities for QGIS use

    HH_path = S1_GEO_DIR / f"{S1_base}_Sigma0_HH_dB_epsg{epsg}_pixelspacing{pixel_spacing}.tif"
    HV_path = S1_GEO_DIR / f"{S1_base}_Sigma0_HV_dB_epsg{epsg}_pixelspacing{pixel_spacing}.tif"
    output_path = S1_GEO_DIR / f"{S1_base}_intensities_epsg{epsg}_pixelspacing{pixel_spacing}.tif"

    geo_utils.stack_geocoded_images(
        HH_path,
        HV_path,
        output_path,
        overwrite = overwrite,
        loglevel = loglevel,
    )

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    main()
    
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# ---- End of <S1_geocode_image.py> ----
