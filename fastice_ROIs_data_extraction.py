# ---- This is <fastice_ROIs_data_extraction.py> ----

"""
Extract S1 and SWOT data from within the fast ice ROIs.
Save as pickle files.
"""

import json
import pickle
from loguru import logger

from config.load_config import *

import numpy as np
import rasterio
from rasterio.mask import mask

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# Define labels
LI_label = 1
DI_label = 2

# Initialize S1 stats
HH_dict = dict()
HV_dict = dict()
IA_dict = dict()

# Initialize SWOT stats
SSHA_dict = dict()
SIG0_dict = dict()
COH_dict = dict()

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# S1: Some ROIs are close to the scene boundary, hence individual lists for LI and DI are needed
# SWOT: Swaths L and R are narrow and sometimes only cover one group of ROIs 

S1_image_list_DI_ROIs = [
    "S1A_EW_GRDM_1SDH_20241009T153800_20241009T153900_056027_06DA59_ACFF",
    "S1A_EW_GRDM_1SDH_20241012T160127_20241012T160231_056071_06DC11_C850",
    "S1A_EW_GRDM_1SDH_20241016T152949_20241016T153049_056129_06DE63_8202",
    "S1A_EW_GRDM_1SDH_20241021T153800_20241021T153900_056202_06E14B_F39C",
    "S1A_EW_GRDM_1SDH_20241024T160127_20241024T160231_056246_06E300_4816",
    "S1A_EW_GRDM_1SDH_20241028T152949_20241028T153049_056304_06E555_11F3",
]

S1_image_list_LI_ROIs = [
    "S1A_EW_GRDM_1SDH_20241009T153800_20241009T153900_056027_06DA59_ACFF",
    "S1A_EW_GRDM_1SDH_20241012T160231_20241012T160331_056071_06DC11_E5B8",
    "S1A_EW_GRDM_1SDH_20241016T152949_20241016T153049_056129_06DE63_8202",
    "S1A_EW_GRDM_1SDH_20241021T153800_20241021T153900_056202_06E14B_F39C",
    "S1A_EW_GRDM_1SDH_20241024T160231_20241024T160331_056246_06E300_C62B",
    "S1A_EW_GRDM_1SDH_20241028T152949_20241028T153049_056304_06E555_11F3",
]

SWOT_image_list_DI_ROIs = [
    "SWOT_L2_LR_SSH_Unsmoothed_022_012_20241001T184713_20241001T193752_PIC0_01_sig0_ssha_coh_ssha_1",
    "SWOT_L2_LR_SSH_Unsmoothed_022_290_20241011T170841_20241011T180008_PIC0_01_sig0_ssha_coh_ssha_2",
    "SWOT_L2_LR_SSH_Unsmoothed_022_318_20241012T170912_20241012T180039_PIC0_01_sig0_ssha_coh_ssha_1",
    "SWOT_L2_LR_SSH_Unsmoothed_023_133_20241026T231634_20241027T000801_PIC2_01_sig0_ssha_coh_ssha_1",
]

SWOT_image_list_LI_ROIs = [
    "SWOT_L2_LR_SSH_Unsmoothed_022_161_20241007T023202_20241007T032328_PIC0_01_sig0_ssha_coh_ssha_2",
    "SWOT_L2_LR_SSH_Unsmoothed_022_290_20241011T170841_20241011T180008_PIC0_01_sig0_ssha_coh_ssha_2",
    "SWOT_L2_LR_SSH_Unsmoothed_022_318_20241012T170912_20241012T180039_PIC0_01_sig0_ssha_coh_ssha_1",
    "SWOT_L2_LR_SSH_Unsmoothed_023_161_20241027T231705_20241028T000832_PIC2_01_sig0_ssha_coh_ssha_2",
    "SWOT_L2_LR_SSH_Unsmoothed_023_133_20241026T231634_20241027T000801_PIC2_01_sig0_ssha_coh_ssha_1",
]

SWOT_orbits = dict()
SWOT_orbits["SWOT_L2_LR_SSH_Unsmoothed_022_012_20241001T184713_20241001T193752_PIC0_01_sig0_ssha_coh_ssha_1"] = "descending"
SWOT_orbits["SWOT_L2_LR_SSH_Unsmoothed_022_161_20241007T023202_20241007T032328_PIC0_01_sig0_ssha_coh_ssha_2"] = "ascending"
SWOT_orbits["SWOT_L2_LR_SSH_Unsmoothed_022_290_20241011T170841_20241011T180008_PIC0_01_sig0_ssha_coh_ssha_2"] = "descending"
SWOT_orbits["SWOT_L2_LR_SSH_Unsmoothed_022_318_20241012T170912_20241012T180039_PIC0_01_sig0_ssha_coh_ssha_1"] = "descending"
SWOT_orbits["SWOT_L2_LR_SSH_Unsmoothed_023_133_20241026T231634_20241027T000801_PIC2_01_sig0_ssha_coh_ssha_1"] = "ascending"
SWOT_orbits["SWOT_L2_LR_SSH_Unsmoothed_023_161_20241027T231705_20241028T000832_PIC2_01_sig0_ssha_coh_ssha_2"] = "ascending"

LI_ROI_file = "LI_fastice_ROIs.geojson"
DI_ROI_file = "DI_fastice_ROIs.geojson"

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# Loop over all ice types
for ice_type in ["LI", "DI"]:

    logger.info(f"Processing ice type: {ice_type}")

    # Select S1_image_list and ROI_geojson for current ice type
    S1_image_list   = eval(f"S1_image_list_{ice_type}_ROIs")
    SWOT_image_list = eval(f"SWOT_image_list_{ice_type}_ROIs")
    geojson_path    = WORK_DIR / "ROIs" / eval(f"{ice_type}_ROI_file")
    label           = eval(f"{ice_type}_label")

    logger.debug(f"S1_image_list:   {S1_image_list}")
    logger.debug(f"SWOT_image_list: {S1_image_list}")
    logger.debug(f"geojson_path:    {geojson_path}")
    logger.debug(f"label:           {label}")

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

    # Read all ROIs from geojson file

    # Initialize empty ROI list
    roi_list = []

    # Read geojson file
    with open (geojson_path, "r") as f:
        roi_data = json.load(f)

    # Loop through features and add ROIs to list
    if "features"in roi_data:
        for feature in roi_data["features"]:
            roi_list.append(feature["geometry"])

    n_rois = len(roi_list)

    logger.info(f"Found {n_rois} ROIs for current ice type")

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

    # Loop over all images for current ice type
    for S1_base in S1_image_list:

        logger.debug(f"Processing S1 scene: {S1_base}")

        HH_geotiff_path = S1_GEO_DIR / f"{S1_base}_Sigma0_HH_dB_epsg3031_pixelspacing80.tif"
        HV_geotiff_path = S1_GEO_DIR / f"{S1_base}_Sigma0_HV_dB_epsg3031_pixelspacing80.tif"
        IA_geotiff_path = S1_GEO_DIR / f"{S1_base}_IA_epsg3031_pixelspacing80.tif"

        # Loop over all ROIs
        for i,roi in enumerate(roi_list):

            logger.debug(f"Processing ROI {i+1}/{n_rois}")

            with rasterio.open(HH_geotiff_path) as src:
                HH, out_transform = mask(src, [roi], crop=True)

            with rasterio.open(HV_geotiff_path) as src:
                HV, out_transform = mask(src, [roi], crop=True)

            with rasterio.open(IA_geotiff_path) as src:
                IA, out_transform = mask(src, [roi], crop=True)

            HH[HH==0] = np.nan
            HV[HV==0] = np.nan
            IA[IA==0] = np.nan


            HH_dict[f"{geojson_path.stem}__roi_{i+1}__{S1_base}"] = dict()
            HV_dict[f"{geojson_path.stem}__roi_{i+1}__{S1_base}"] = dict()
            IA_dict[f"{geojson_path.stem}__roi_{i+1}__{S1_base}"] = dict()

            HH_dict[f"{geojson_path.stem}__roi_{i+1}__{S1_base}"]["data"] = HH
            HV_dict[f"{geojson_path.stem}__roi_{i+1}__{S1_base}"]["data"] = HV
            IA_dict[f"{geojson_path.stem}__roi_{i+1}__{S1_base}"]["data"] = IA

            HH_dict[f"{geojson_path.stem}__roi_{i+1}__{S1_base}"]["label"] = label
            HV_dict[f"{geojson_path.stem}__roi_{i+1}__{S1_base}"]["label"] = label
            IA_dict[f"{geojson_path.stem}__roi_{i+1}__{S1_base}"]["label"] = label


    for SWOT_ssha_base in SWOT_image_list:

        logger.debug(f"Processing SWOT scene: {S1_base}")

        # Find SWOT_base name and LR swath (#1,2)
        LR        = SWOT_ssha_base[-1]
        SWOT_base = SWOT_ssha_base[0:-7]

        # Build paths to SWOT data and ROI geojson
        SWOT_ssha_geotiff_path = SWOT_GEOTIFF_DIR / f"{SWOT_ssha_base}_epsg3031_pixelspacing250.tiff"
        SWOT_sig0_geotiff_path = SWOT_GEOTIFF_DIR / f"{SWOT_base}_sig0_{LR}_epsg3031_pixelspacing250.tiff"
        SWOT_coh_geotiff_path  = SWOT_GEOTIFF_DIR / f"{SWOT_base}_coh_{LR}_epsg3031_pixelspacing250.tiff"

        # Loop over all ROIs
        for i,roi in enumerate(roi_list):

            logger.debug(f"Processing ROI {i+1}/{n_rois}")

            with rasterio.open(SWOT_ssha_geotiff_path) as src:
                SSHA, out_transform = mask(src, [roi], crop=True)

            with rasterio.open(SWOT_sig0_geotiff_path) as src:
                SIG0, out_transform = mask(src, [roi], crop=True)

            with rasterio.open(SWOT_coh_geotiff_path) as src:
                COH, out_transform = mask(src, [roi], crop=True)

            SSHA[SSHA==0] = np.nan
            SIG0[SIG0==0] = np.nan
            COH[COH==0] = np.nan

            SSHA_dict[f"{geojson_path.stem}__roi_{i+1}__{SWOT_base}"] = dict()
            SIG0_dict[f"{geojson_path.stem}__roi_{i+1}__{SWOT_base}"] = dict()
            COH_dict[f"{geojson_path.stem}__roi_{i+1}__{SWOT_base}"]  = dict()

            SSHA_dict[f"{geojson_path.stem}__roi_{i+1}__{SWOT_base}"]["data"] = SSHA
            SIG0_dict[f"{geojson_path.stem}__roi_{i+1}__{SWOT_base}"]["data"] = SIG0
            COH_dict[f"{geojson_path.stem}__roi_{i+1}__{SWOT_base}"]["data"]  = COH

            SSHA_dict[f"{geojson_path.stem}__roi_{i+1}__{SWOT_base}"]["label"]  = label
            SIG0_dict[f"{geojson_path.stem}__roi_{i+1}__{SWOT_base}"]["label"]  = label
            COH_dict[f"{geojson_path.stem}__roi_{i+1}__{SWOT_base}"]["label"]   = label

            SSHA_dict[f"{geojson_path.stem}__roi_{i+1}__{SWOT_base}"]["orbit"] = SWOT_orbits[SWOT_ssha_base]
            SIG0_dict[f"{geojson_path.stem}__roi_{i+1}__{SWOT_base}"]["orbit"] = SWOT_orbits[SWOT_ssha_base] 
            COH_dict[f"{geojson_path.stem}__roi_{i+1}__{SWOT_base}"]["orbit"]  =  SWOT_orbits[SWOT_ssha_base]

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

HH_fastice_stats_path   = ROI_ANALYSIS_DIR / f"fastice_ROIs_HH.pickle"
HV_fastice_stats_path   = ROI_ANALYSIS_DIR / f"fastice_ROIs_HV.pickle"
IA_fastice_stats_path   = ROI_ANALYSIS_DIR / f"fastice_ROIs_IA.pickle"
SSHA_fastice_stats_path = ROI_ANALYSIS_DIR / f"fastice_ROIs_SSHA.pickle"
SIG0_fastice_stats_path = ROI_ANALYSIS_DIR / f"fastice_ROIs_SIG0.pickle"
COH_fastice_stats_path  = ROI_ANALYSIS_DIR / f"fastice_ROIs_COH.pickle"

logger.info("Saving stats dictionaries as pickle files")

with open(HH_fastice_stats_path, 'wb') as f:
    pickle.dump(HH_dict, f)

with open(HV_fastice_stats_path, 'wb') as f:
    pickle.dump(HV_dict, f)

with open(IA_fastice_stats_path, 'wb') as f:
    pickle.dump(IA_dict, f)

with open(SSHA_fastice_stats_path, 'wb') as f:
    pickle.dump(SSHA_dict, f)

with open(SIG0_fastice_stats_path, 'wb') as f:
    pickle.dump(SIG0_dict, f)

with open(COH_fastice_stats_path, 'wb') as f:
    pickle.dump(COH_dict, f)

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# ---- End of <fastice_ROIs_data_extraction.py> ----
