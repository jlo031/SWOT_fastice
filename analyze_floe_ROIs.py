# ---- This is <analyze_floe_ROIs.py> ----

"""
TODO
"""

from loguru import logger

from config.load_config import *


import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import json
from scipy import stats

import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

HH_min = -25
HH_max = 0
HV_min = -35
HV_max = -10

class_colors = [
    np.array([0, 153, 153])/255,
    np.array([153, 0, 0])/255,
]
class_colors = [
    np.array([60, 100, 180])/255,
    np.array([180, 140, 60])/255,
]

LI_label = 1
DI_label = 2

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# Initialize S1 stats

HH_dict = dict()
HV_dict = dict()
IA_dict = dict()

HH_means = []
HV_means = []
IA_means = []

HH_vars = []
HV_vars = []
IA_vars = []

HH_stds = []
HV_stds = []
IA_stds = []

S1_labels = []


# Initialize SWOT stats

SSHA_dict = dict()
SIG0_dict = dict()
COH_dict = dict()

SSHA_means = []
SIG0_means = []
COH_means = []

SSHA_vars = []
SIG0_vars = []
COH_vars = []

SSHA_stds = []
SIG0_stds = []
COH_stds = []

SWOT_labels = []

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

floe_list = [
    "floe_01",
    "floe_02"
]

floe_labels = dict()
floe_labels["floe_01"] = DI_label
floe_labels["floe_02"] = LI_label

S1_image_list        = dict()
SWOT_ssha_image_list = dict()
S1_ROI_list          = dict()
SWOT_ROI_list        = dict()

S1_image_list["floe_01"] = [
    "S1A_EW_GRDM_1SDH_20241030T151338_20241030T151438_056333_06E671_6637",
    "S1A_EW_GRDM_1SDH_20241028T152949_20241028T153049_056304_06E555_11F3",
    "S1A_EW_GRDM_1SDH_20241025T150544_20241025T150644_056260_06E393_E973",
    "S1A_EW_GRDM_1SDH_20241023T152142_20241023T152242_056231_06E267_734D",
    "S1A_EW_GRDM_1SDH_20241020T145733_20241020T145833_056187_06E0AF_C060",
]

SWOT_ssha_image_list["floe_01"] = [
    "SWOT_L2_LR_SSH_Unsmoothed_023_234_20241030T135243_20241030T144410_PIC2_01_sig0_ssha_coh_ssha_1",
    "SWOT_L2_LR_SSH_Unsmoothed_023_206_20241029T135212_20241029T144339_PIC2_01_sig0_ssha_coh_ssha_2",
    "SWOT_L2_LR_SSH_Unsmoothed_023_161_20241027T231705_20241028T000832_PIC2_01_sig0_ssha_coh_ssha_2",
    "SWOT_L2_LR_SSH_Unsmoothed_023_133_20241026T231634_20241027T000801_PIC2_01_sig0_ssha_coh_ssha_1",
]

S1_ROI_list["floe_01"] =[
    "floe_01_S1_20241020",
    "floe_01_S1_20241023",
    "floe_01_S1_20241025",
    "floe_01_S1_20241028",
    "floe_01_S1_20241030",
]

SWOT_ROI_list["floe_01"] = [
    "floe_01_SWOT_20241026",
    "floe_01_SWOT_20241027",
    "floe_01_SWOT_20241029",
    "floe_01_SWOT_20241030",
]

S1_image_list["floe_02"] = [
    "S1A_EW_GRDM_1SDH_20241030T151338_20241030T151438_056333_06E671_6637",
    "S1A_EW_GRDM_1SDH_20241028T152949_20241028T153049_056304_06E555_11F3",
]

SWOT_ssha_image_list["floe_02"] = [
    "SWOT_L2_LR_SSH_Unsmoothed_023_161_20241027T231705_20241028T000832_PIC2_01_sig0_ssha_coh_ssha_2",
]

S1_ROI_list["floe_02"] =[

    "floe_02_S1_20241028",
    "floe_02_S1_20241030",
]

SWOT_ROI_list["floe_02"] = [
    "floe_02_SWOT_20241027",
]



# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# Loop over all ice types
for floe in floe_list:

    logger.info(f"Processing floe: {floe}")
    label = floe_labels[floe]

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

    logger.info("Processing S1 images")

    # Loop over all ROI/image pairs for current floe
    for S1_ROI in S1_ROI_list[floe]:

        logger.debug(f"Processing ROI: {S1_ROI}")

        # Get date from current ROI name
        date = S1_ROI.split("_")[3]

        # Find corresponding S1_base name
        S1_base = [ f for f in S1_image_list[floe] if date in f ][0]

        # Build paths to S1 data and ROI geojson
        HH_geotiff_path = S1_GEO_DIR / f"{S1_base}_Sigma0_HH_dB_epsg3031_pixelspacing80.tif"
        HV_geotiff_path = S1_GEO_DIR / f"{S1_base}_Sigma0_HV_dB_epsg3031_pixelspacing80.tif"
        IA_geotiff_path = S1_GEO_DIR / f"{S1_base}_IA_epsg3031_pixelspacing80.tif"
        geojson_path    = WORK_DIR / "ROIs" / f"{S1_ROI}.geojson"

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

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

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

            HH_dict[f"{floe}__roi_{i+1}__{S1_base}"] = HH
            HV_dict[f"{floe}__roi_{i+1}__{S1_base}"] = HV
            IA_dict[f"{floe}__roi_{i+1}__{S1_base}"] = IA

            HH_mean = np.nanmean(HH).mean()
            HV_mean = np.nanmean(HV)
            IA_mean = np.nanmean(IA)

            HH_var = np.nanvar(HH)
            HV_var = np.nanvar(HV)
            IA_var = np.nanvar(IA)

            HH_std = np.nanstd(HH)
            HV_std = np.nanstd(HV)
            IA_std = np.nanstd(IA)

            HH_means.append(HH_mean)
            HH_vars.append(HH_var)
            HH_stds.append(HH_std)
            HV_means.append(HV_mean)
            HV_vars.append(HV_var)
            HV_stds.append(HV_std)
            IA_means.append(IA_mean)
            IA_vars.append(IA_var)
            IA_stds.append(IA_std)

            S1_labels.append(label)

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

    logger.info("Processing SWOT images")

    # Loop over all ROI/image pairs for current floe
    for SWOT_ROI in SWOT_ROI_list[floe]:

        logger.debug(f"Processing ROI: {SWOT_ROI}")

        # Get date from current ROI name
        date = SWOT_ROI.split("_")[3]

        # Find corresponding SWOT_ssh_base name
        SWOT_ssha_base = [ f for f in SWOT_ssha_image_list[floe] if date in f ][0]

        # Find SWOT_base name and LR (1,2)
        LR        = SWOT_ssha_base[-1]
        SWOT_base = SWOT_ssha_base[0:-7]

        # Build paths to SWOT data and ROI geojson
        SWOT_ssha_geotiff_path = SWOT_GEOTIFF_DIR / f"{SWOT_ssha_base}_epsg3031_pixelspacing250.tiff"
        SWOT_sig0_geotiff_path = SWOT_GEOTIFF_DIR / f"{SWOT_base}_sig0_{LR}_epsg3031_pixelspacing250.tiff"
        SWOT_coh_geotiff_path  = SWOT_GEOTIFF_DIR / f"{SWOT_base}_coh_{LR}_epsg3031_pixelspacing250.tiff"
        geojson_path           = WORK_DIR / "ROIs" / f"{SWOT_ROI}.geojson"

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

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

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

            SSHA_dict[f"{floe}__roi_{i+1}__{S1_base}"] = SSHA
            SIG0_dict[f"{floe}__roi_{i+1}__{S1_base}"] = SIG0
            COH_dict[f"{floe}__roi_{i+1}__{S1_base}"]  = COH

            SSHA_mean = np.nanmean(SSHA).mean()
            SIG0_mean = np.nanmean(SIG0).mean()
            COH_mean = np.nanmean(COH).mean()

            SSHA_var = np.nanvar(SSHA)
            SIG0_var = np.nanvar(SIG0)
            COH_var = np.nanvar(COH)

            SSHA_std = np.nanstd(SSHA)
            SIG0_std = np.nanstd(SIG0)
            COH_std = np.nanstd(COH)

            SSHA_means.append(SSHA_mean)
            SSHA_vars.append(SSHA_var)
            SSHA_stds.append(SSHA_std)
            SIG0_means.append(SIG0_mean)
            SIG0_vars.append(SIG0_var)
            SIG0_stds.append(SIG0_std)
            COH_means.append(COH_mean)
            COH_vars.append(COH_var)
            COH_stds.append(COH_std)
                        
            SWOT_labels.append(label)













# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #


S1_labels = np.array(S1_labels)
HH_means  = np.array(HH_means)
HH_vars   = np.array(HH_vars)
HH_stds   = np.array(HH_stds)
HV_means  = np.array(HV_means)
HV_vars   = np.array(HV_vars)
HV_stds   = np.array(HV_stds)
IA_means  = np.array(IA_means)
IA_vars   = np.array(IA_vars)
IA_stds   = np.array(IA_stds)

SWOT_labels = np.array(S1_labels)
SSHA_means  = np.array(SSHA_means)
SSHA_vars   = np.array(SSHA_vars)
SSHA_stds   = np.array(SSHA_stds)

DI_idx = S1_labels==DI_label
DI_HH_b, DI_HH_a, DI_HH_r_value, DI_HH_p_value, DI_HH_std_err = stats.linregress(IA_means[DI_idx], HH_means[DI_idx])
DI_HH_R2 = DI_HH_r_value**2
DI_HV_b, DI_HV_a, DI_HV_r_value, DI_HV_p_value, DI_HV_std_err = stats.linregress(IA_means[DI_idx], HV_means[DI_idx])
DI_HV_R2 = DI_HV_r_value**2

LI_idx = S1_labels==LI_label
LI_HH_b, LI_HH_a, LI_HH_r_value, LI_HH_p_value, LI_HH_std_err = stats.linregress(IA_means[LI_idx], HH_means[LI_idx])
LI_HH_R2 = LI_HH_r_value**2
LI_HV_b, LI_HV_a, LI_HV_r_value, LI_HV_p_value, LI_HV_std_err = stats.linregress(IA_means[LI_idx], HV_means[LI_idx])
LI_HV_R2 = LI_HV_r_value**2


IA_min = 18
IA_max = 45
IA_lin = np.linspace(IA_min, IA_max, IA_max-IA_min+1)
DI_HH_lin = DI_HH_a + DI_HH_b * IA_lin
DI_HV_lin = DI_HV_a + DI_HV_b * IA_lin
LI_HH_lin = LI_HH_a + LI_HH_b * IA_lin
LI_HV_lin = LI_HV_a + LI_HV_b * IA_lin

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

fig, axes = plt.subplots(2,1,sharex=True,figsize=((12,8)))
axes = axes.ravel()

# LI
axes[0].errorbar(IA_means[LI_idx], HH_means[LI_idx], yerr=HH_stds[LI_idx], fmt="o", capsize=3, elinewidth=1, color=class_colors[0])
axes[1].errorbar(IA_means[LI_idx], HV_means[LI_idx], yerr=HV_stds[LI_idx], fmt="o", capsize=3, elinewidth=1, color=class_colors[0])
axes[0].plot(IA_lin, LI_HH_lin, "--", color=class_colors[0])
axes[1].plot(IA_lin, LI_HV_lin, "--", color=class_colors[0])

# DI
axes[0].errorbar(IA_means[DI_idx], HH_means[DI_idx], yerr=HH_stds[DI_idx], fmt="o", capsize=3, elinewidth=1, color=class_colors[1])
axes[1].errorbar(IA_means[DI_idx], HV_means[DI_idx], yerr=HV_stds[DI_idx], fmt="o", capsize=3, elinewidth=1, color=class_colors[1])
axes[0].plot(IA_lin, DI_HH_lin, "--", color=class_colors[1])
axes[1].plot(IA_lin, DI_HV_lin, "--", color=class_colors[1])

##axes[0].set_ylim(-15.5,-9.5)
##axes[1].set_ylim(-28.0,-22.0)

axes[1].set_xlabel("IA (deg)")
axes[0].set_ylabel("HH (dB)")
axes[1].set_ylabel("HV (dB)")

axes[0].set_xlim((IA_min,IA_max))

axes[0].legend([f"Level Ice\nb={LI_HH_b:.2f}", f"Deformed Ice\nb={DI_HH_b:.2f}"])
axes[1].legend([f"Level Ice\nb={LI_HV_b:.2f}", f"Deformed Ice\nb={DI_HV_b:.2f}"])

plt.show()

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #





"""
def extract_pixel_values_from_geometry(geotiff_path, geojson_path):

    geojson = gpd.read_file(geojson_path)
    geometries = [ json.loads(geojson.geometry.to_json())["features"][0]["geometry"] ]
    
    with rasterio.open(geotiff_path) as src:
        out_image, out_transform = mask(src, geometries, crop=True)
        out_meta = src.meta

    pixel_values = out_image.flatten()

    nodata_value = out_meta.get("nodata", None)

    
    return pixel_values
"""    
