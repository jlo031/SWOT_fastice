# ---- This is <S1_preprocess_image.py> ----

"""
Pre-process S1 input image.
Folder structure is read from config folder.

    Default feature extraction: Sigma0_HH_dB, Sigma0_HV_dB, IA, swath_mask
    Optional feature extraction: lat/lon, landmask, RGB
"""

import pathlib
import sys
import argparse

from loguru import logger

import S1_processing.utils as S1_utils
import S1_processing.S1_feature_extraction as S1_feat

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
    p.add_argument("-get_lat_lon", action = "store_true", help = "extract lat/lon")
    p.add_argument("-get_landmask", action = "store_true", help = "extract landmask")
    p.add_argument("-make_RGB", action = "store_true", help = "make RGB image")
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
    S1_base      = args.S1_base
    ML           = args.ML
    get_lat_lon  = args.get_lat_lon
    get_landmask = args.get_landmask
    make_RGB     = args.make_RGB
    overwrite    = args.overwrite
    loglevel     = args.loglevel

    logger.info(f"Processing image:     {S1_base}")

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

    # Build path to feature folder and RGB folder (for ML setting)
    feat_folder = S1_FEAT_DIR / f"ML_{ML}" / f"{S1_base}"
    rgb_folder  =  S1_RGB_DIR / f"ML_{ML}"

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

    # Get intensities

    for intensity in ['HH', 'HV']:
        S1_feat.get_S1_intensity(
            S1_safe_zip,
            feat_folder,
            intensity,
            GPT,
            ML = ML,
            dB = True,
            overwrite = overwrite,
            loglevel = loglevel
        )

# --------------------------------------------------------------------------- #
    
    # Get meta data
    
    # IA
    S1_feat.get_S1_IA(
        S1_safe_zip,
        feat_folder,
        GPT,
        loglevel = loglevel,
        overwrite = overwrite
    )

    # swath_mask
    S1_feat.get_S1_swath_mask(
        S1_safe_zip,
        feat_folder,
        loglevel = loglevel,
        overwrite = overwrite
    )

    # lat/lon
    if get_lat_lon:
        S1_feat.get_S1_lat_lon(
            S1_safe_zip, 
            feat_folder,
            GPT,
            loglevel = loglevel,
            overwrite = overwrite
        )

    # lat/lon
    if get_landmask:
        logger.warning("Landmask not implemented yet")

# --------------------------------------------------------------------------- #

    # Make RGB for labelme

    if make_RGB:

        # Full path to RGB output file
        RGB_img_path = rgb_folder / f"{S1_base}_rgb.tif"

        if RGB_img_path.is_file() and not overwrite:
            logger.info("RGB tif file already exists.")
        else:

            for intensity in ['HH', 'HV']:
                S1_feat.get_S1_intensity(
                    S1_safe_zip,
                    feat_folder,
                    intensity,
                    GPT,
                    ML = ML,
                    dB = False,
                    overwrite = overwrite,
                    loglevel = loglevel
                )

            S1_feat.make_S1_rgb(
                feat_folder,
                rgb_folder,
                hhMin = -30,
                hhMax = 0,
                hvMin = -35,
                hvMax = -5,
                newMin = 0,
                newMax = 255,
                red = 'HV',
                green = 'HH',
                blue = 'HH',
                loglevel = loglevel,
                overwrite = overwrite
            )

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    main()
    
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# ---- End of <S2_preprocess_image.py> ----
