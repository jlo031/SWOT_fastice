# ---- This is <gamma_2_sigma_conversion_check.py> ----

"""
Load GA processed data for input image.
Convert gamma0 to sigma0 with IA layer and with conversion layer.
Compute differences and show histograms.
"""

import argparse
import os
import sys
from loguru import logger

from scipy import stats
import numpy as np
import rasterio
import matplotlib.pyplot as plt

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
    loglevel      = args.loglevel

    logger.info(f"Processing image: {S1_base}")

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

    # Get list of all GA tif files in data folder
    GA_file_list = [ f for f in os.listdir(GA_DATA_DIR/S1_base) if f.endswith(".tif") ]

    # Get S1_GA_base
    S1_GA_base = GA_file_list[0][0:27]

    # Define full paths to GA tif files
    HH_gamma0_dB_path      = GA_DATA_DIR / S1_base / f"{S1_GA_base}_HH_grd_mli_gamma0-rtc_geo_db_3031.tif"
    HV_gamma0_dB_path      = GA_DATA_DIR / S1_base / f"{S1_GA_base}_HV_grd_mli_gamma0-rtc_geo_db_3031.tif"
    HH_gamma0_path         = GA_DATA_DIR / S1_base / f"{S1_GA_base}_HH_grd_mli_gamma0-rtc_geo_3031.tif"
    HV_gamma0_path         = GA_DATA_DIR / S1_base / f"{S1_GA_base}_HV_grd_mli_gamma0-rtc_geo_3031.tif"
    gamma_sigma_ratio_path = GA_DATA_DIR / S1_base / f"{S1_GA_base}_gs_ratio_geo_3031.tif"
    dem_path               = GA_DATA_DIR / S1_base / f"{S1_GA_base}_dem_seg_geo_3031.tif"
    IA_path                = GA_DATA_DIR / S1_base / f"{S1_GA_base}_inc_geo_3031.tif"

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

    # Read data

    logger.info("Reading image data")

    with rasterio.open(HH_gamma0_path) as src:
        HH_gamma0 = src.read(1)

    with rasterio.open(HV_gamma0_path) as src:
        HV_gamma0 = src.read(1)

    with rasterio.open(gamma_sigma_ratio_path) as src:
        gamma_sigma_ratio = src.read(1)

    with rasterio.open(dem_path) as src:
        DEM = src.read(1)

    with rasterio.open(IA_path) as src:
        IA = src.read(1)


    # Convert DEM and IA values outside of image data to nans
    DEM[~np.isfinite(HH_gamma0)] = np.nan
    IA[~np.isfinite(HH_gamma0)]  = np.nan

    # Convert IA to deg
    IA_deg = np.rad2deg(IA)

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

    # Convert gamma0 to sigma0

    logger.info("Converting gamma0 to sigma0")

    # With IA
    my_HH_sigma0 = HH_gamma0 * np.cos(IA)
    my_HV_sigma0 = HV_gamma0 * np.cos(IA)

    # With conversion layer
    GA_HH_sigma0 = HH_gamma0 * gamma_sigma_ratio
    GA_HV_sigma0 = HV_gamma0 * gamma_sigma_ratio

    # Check individual pixels
    # cos(IA) and conversion layer should be identical
    logger.debug(f"GA HH_sigma0_dB:   {10*np.log10(GA_HH_sigma0[9000:9005,9000])}")
    logger.debug(f"my HH_sigma0_dB:   {10*np.log10(my_HH_sigma0[9000:9005,9000])}")
    logger.debug(f"GA HV_sigma0_dB:   {10*np.log10(GA_HV_sigma0[9000:9005,9000])}")
    logger.debug(f"my HV_sigma0_dB:   {10*np.log10(my_HV_sigma0[9000:9005,9000])}")
    logger.debug(f"cos(IA):           {np.cos(IA)[9000:9005,9000]}")
    logger.debug(f"gamma_sigma_ratio: {gamma_sigma_ratio[9000:9005,9000]}")

    HH_gamma0 = HV_gamma0 = 0 

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

    # Compute sigma0 diff

    logger.info("Selecting valid indices and calculating sigma0 difference")

    # Select all finite (non nan) values
    fin_IA           = IA_deg[np.isfinite(IA_deg)]
    fin_DEM          = DEM[np.isfinite(DEM)]
    fin_my_HH_sigma0 =  my_HH_sigma0[np.isfinite(my_HH_sigma0)]
    fin_GA_HH_sigma0 =  GA_HH_sigma0[np.isfinite(GA_HH_sigma0)]
    fin_my_HV_sigma0 =  my_HV_sigma0[np.isfinite(my_HV_sigma0)]
    fin_GA_HV_sigma0 =  GA_HV_sigma0[np.isfinite(GA_HV_sigma0)]

    # Convert to dB
    fin_my_HH_sigma0_dB = 10*np.log10(fin_my_HH_sigma0)
    fin_GA_HH_sigma0_dB = 10*np.log10(fin_GA_HH_sigma0)
    fin_my_HV_sigma0_dB = 10*np.log10(fin_my_HV_sigma0)
    fin_GA_HV_sigma0_dB = 10*np.log10(fin_GA_HV_sigma0)

    # Compute differences of dB values for all pixels
    HH_dB_diff = fin_my_HH_sigma0_dB - fin_GA_HH_sigma0_dB
    HV_dB_diff = fin_my_HV_sigma0_dB - fin_GA_HV_sigma0_dB

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

    # FIGURES

    plot_quicklook         = True
    plot_histograms        = True
    plot_IA_histograms     = True
    plot_IA_vs_sigma_diff  = True
    plot_DEM_histograms    = True
    plot_DEM_vs_sigma_diff = True

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

    if plot_quicklook: 

        logger.info("Drawing Figure 1: Quicklook")

        # Read data
        with rasterio.open(HH_gamma0_dB_path) as src:
            HH_gamma0_dB = src.read(1)

        with rasterio.open(HV_gamma0_dB_path) as src:
            HV_gamma0_dB = src.read(1)

        # Subsample quicklook
        sub_ql = 10

        fig, axes = plt.subplots(1,2,sharex=True,sharey=True,figsize=((12,5)))
        axes = axes.ravel()
        axes[0].imshow(HH_gamma0_dB[::sub_ql,::sub_ql], vmin=-22,vmax=-8, cmap="gray", interpolation="nearest")
        axes[1].imshow(HV_gamma0_dB[::sub_ql,::sub_ql], vmin=-32,vmax=-18, cmap="gray", interpolation="nearest")

        axes[0].set_title("HH")
        axes[1].set_title("HV")

        plt.suptitle(f"{S1_base}")

        ###plt.savefig(f"figures/gamma_sigma_conversion/{S1_base}_01_quicklook.pdf", dpi=300)
        plt.savefig(f"figures/gamma_sigma_conversion/{S1_base}_01_quicklook.png", dpi=300)

        HH_gamma0_dB = HV_gamma0_dB = 0

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

    if plot_histograms:

        logger.info("Drawing Figure 2: Full histograms")

        fig, axes = plt.subplots(1,2,sharex=True,sharey=True,figsize=((12,5)))
        axes = axes.ravel()
        axes[0].hist(HH_dB_diff, bins=50, range=(-3,1))
        axes[1].hist(HV_dB_diff, bins=50, range=(-3,1))

        axes[0].set_yscale('log')
        axes[1].set_yscale('log')

        axes[0].set_xlabel(r'$_{my}\sigma_{HH}^{0}$ (dB) - $_{GA}\sigma_{HH}^{0}$ (dB)')
        axes[1].set_xlabel(r'$_{my}\sigma_{HV}^{0}$ (dB) - $_{GA}\sigma_{HV}^{0}$ (dB)')
        axes[0].set_ylabel("Counts")

        plt.suptitle(f"{S1_base}")

        ###plt.savefig(f"figures/gamma_sigma_conversion/{S1_base}_02_sigma_diff_histograms.pdf", dpi=300)
        plt.savefig(f"figures/gamma_sigma_conversion/{S1_base}_02_sigma_diff_histograms.png", dpi=300)

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

    if plot_IA_histograms:

        logger.info("Drawing Figure 3: IA-dependent histograms")

        IA_threshold_1 = 30
        IA_threshold_2 = 40

        IA_low  = (fin_IA <= IA_threshold_1)
        IA_mid  = (fin_IA > IA_threshold_1) & (fin_IA < IA_threshold_2)
        IA_high = (fin_IA >= IA_threshold_2)

        HH_dB_diff_IA_low  = HH_dB_diff[IA_low]
        HH_dB_diff_IA_mid  = HH_dB_diff[IA_mid]
        HH_dB_diff_IA_high = HH_dB_diff[IA_high]
        HV_dB_diff_IA_low  = HV_dB_diff[IA_low]
        HV_dB_diff_IA_mid  = HV_dB_diff[IA_mid]
        HV_dB_diff_IA_high = HV_dB_diff[IA_high]

        fig, axes = plt.subplots(3,2,sharex=True,sharey=True,figsize=((12,15)))
        axes = axes.ravel()
        axes[0].hist(HH_dB_diff_IA_low, bins=50, range=(-3,1))
        axes[2].hist(HH_dB_diff_IA_mid, bins=50, range=(-3,1))
        axes[4].hist(HH_dB_diff_IA_high, bins=50, range=(-3,1))
        axes[1].hist(HV_dB_diff_IA_low, bins=50, range=(-3,1))
        axes[3].hist(HV_dB_diff_IA_mid, bins=50, range=(-3,1))
        axes[5].hist(HV_dB_diff_IA_high, bins=50, range=(-3,1))

        axes[0].set_yscale('log')
        axes[1].set_yscale('log')
        axes[2].set_yscale('log')
        axes[3].set_yscale('log')
        axes[4].set_yscale('log')
        axes[5].set_yscale('log')

        axes[4].set_xlabel(r'$_{my}\sigma_{HH}^{0}$ (dB) - $_{GA}\sigma_{HH}^{0}$ (dB)')
        axes[5].set_xlabel(r'$_{my}\sigma_{HV}^{0}$ (dB) - $_{GA}\sigma_{HV}^{0}$ (dB)')

        axes[0].set_ylabel("Counts")
        axes[2].set_ylabel("Counts")
        axes[4].set_ylabel("Counts")

        axes[0].set_title(f"For IA<{IA_threshold_1}")
        axes[2].set_title(f"For {IA_threshold_1}<IA<{IA_threshold_2}")
        axes[4].set_title(f"For IA>{IA_threshold_2}")

        axes[1].set_title(f"For IA<{IA_threshold_1}")
        axes[3].set_title(f"For {IA_threshold_1}<IA<{IA_threshold_2}")
        axes[5].set_title(f"For IA>{IA_threshold_2}")

        plt.suptitle(f"{S1_base}")

        ###plt.savefig(f"figures/gamma_sigma_conversion/{S1_base}_03_sigma_diff_histograms_per_IA_range.pdf", dpi=300)
        plt.savefig(f"figures/gamma_sigma_conversion/{S1_base}_03_sigma_diff_histograms_per_IA_range.png", dpi=300)

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

    if plot_IA_vs_sigma_diff:

        logger.info("Drawing Figure 4: IA versus sigma0 difference")

        # No need to plot all points
        # Might be worth plotting a density / 2D histogram with log color scale
        sub = 100

        IA_min = int(np.floor(fin_IA.min()))
        IA_max = int(np.ceil(fin_IA.max()))
        IA_lin = np.linspace(IA_min, IA_max, IA_max-IA_min+1)

        fig, axes = plt.subplots(2,1,sharex=True,sharey=True,figsize=((12,5)))
        axes = axes.ravel()
        axes[0].plot(fin_IA[::sub],HH_dB_diff[::sub],'o')
        axes[1].plot(fin_IA[::sub],HV_dB_diff[::sub],'o')

        axes[0].plot(IA_lin, len(IA_lin)*[0], '--', color=[1,0.6,0])
        axes[1].plot(IA_lin, len(IA_lin)*[0], '--', color=[1,0.6,0])

        axes[0].set_ylabel(r'$_{my}\sigma_{HH}^{0}$ (dB) - $_{GA}\sigma_{HH}^{0}$ (dB)')
        axes[1].set_ylabel(r'$_{my}\sigma_{HV}^{0}$ (dB) - $_{GA}\sigma_{HV}^{0}$ (dB)')
        axes[1].set_xlabel("IA (deg)")

        axes[0].set_xlim([IA_min,IA_max])
        axes[1].set_xlim([IA_min,IA_max])

        plt.suptitle(f"{S1_base}")

        ###plt.savefig(f"figures/gamma_sigma_conversion/{S1_base}_04_sigma_diff_vs_IA.pdf", dpi=300)
        plt.savefig(f"figures/gamma_sigma_conversion/{S1_base}_04_sigma_diff_vs_IA.png", dpi=300)
   
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

    if plot_DEM_histograms:

        logger.info("Drawing Figure 5: DEM-dependent histograms")

        # Conservative DEM threshold to ensure low DEM values are only open water/sea ice
        DEM_threshold = 20

        DEM_low  = (fin_DEM <= DEM_threshold)
        DEM_high = (fin_DEM >= DEM_threshold)

        HH_dB_diff_DEM_low  = HH_dB_diff[DEM_low]
        HH_dB_diff_DEM_high = HH_dB_diff[DEM_high]
        HV_dB_diff_DEM_low  = HV_dB_diff[DEM_low]
        HV_dB_diff_DEM_high = HV_dB_diff[DEM_high]

        IA_DEM_low          = fin_IA[DEM_low]
        IA_DEM_high         = fin_IA[DEM_high]

        fig, axes = plt.subplots(2,2,sharex=True,sharey=True,figsize=((12,10)))
        axes = axes.ravel()
        axes[0].hist(HH_dB_diff_DEM_low, bins=50, range=(-5,1))
        axes[2].hist(HH_dB_diff_DEM_high, bins=50, range=(-5,1))
        axes[1].hist(HV_dB_diff_DEM_low, bins=50, range=(-5,1))
        axes[3].hist(HV_dB_diff_DEM_high, bins=50, range=(-5,1))

        axes[0].set_yscale('log')
        axes[1].set_yscale('log')
        axes[2].set_yscale('log')
        axes[3].set_yscale('log')

        axes[2].set_xlabel(r'$_{my}\sigma_{HH}^{0}$ (dB) - $_{GA}\sigma_{HH}^{0}$ (dB)')
        axes[3].set_xlabel(r'$_{my}\sigma_{HV}^{0}$ (dB) - $_{GA}\sigma_{HV}^{0}$ (dB)')

        axes[0].set_ylabel("Counts")
        axes[2].set_ylabel("Counts")

        axes[0].set_title(f"For DEM<{DEM_threshold}")
        axes[1].set_title(f"For DEM<{DEM_threshold}")
        axes[2].set_title(f"For DEM>{DEM_threshold}")
        axes[3].set_title(f"For DEM>{DEM_threshold}")

        plt.suptitle(f"{S1_base}")

        ###plt.savefig(f"figures/gamma_sigma_conversion/{S1_base}_05_sigma_diff_histograms_per_IA_range.pdf", dpi=300)
        plt.savefig(f"figures/gamma_sigma_conversion/{S1_base}_05_sigma_diff_histograms_per_DEM_range.png", dpi=300)

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

    if plot_DEM_vs_sigma_diff:

        logger.info("Drawing Figure 6: DEM versus sigma0 difference")

        # No need to plot all points
        # Might be worth plotting a density / 2D histogram with log color scale
        sub = 100

        DEM_min = int(np.floor(fin_DEM.min()))
        DEM_max = int(np.ceil(fin_DEM.max()))
        DEM_lin = np.linspace(DEM_min, DEM_max, DEM_max-DEM_min+1)

        fig, axes = plt.subplots(2,1,sharex=True,sharey=True,figsize=((12,5)))
        axes = axes.ravel()
        axes[0].plot(fin_DEM[::sub],HH_dB_diff[::sub],'o')
        axes[1].plot(fin_DEM[::sub],HV_dB_diff[::sub],'o')

        axes[0].plot(DEM_lin, len(DEM_lin)*[0], '--', color=[1,0.6,0])
        axes[1].plot(DEM_lin, len(DEM_lin)*[0], '--', color=[1,0.6,0])

        axes[0].set_ylabel(r'$_{my}\sigma_{HH}^{0}$ (dB) - $_{GA}\sigma_{HH}^{0}$ (dB)')
        axes[1].set_ylabel(r'$_{my}\sigma_{HV}^{0}$ (dB) - $_{GA}\sigma_{HV}^{0}$ (dB)')
        axes[1].set_xlabel("DEM (dm)")

        axes[0].set_xlim([DEM_min,DEM_max])
        axes[1].set_xlim([DEM_min,DEM_max])

        axes[0].set_xlim([0,100])
        axes[1].set_xlim([0,100])

        plt.suptitle(f"{S1_base}")

        ###plt.savefig(f"figures/gamma_sigma_conversion/{S1_base}_06_sigma_diff_vs_DEM_1.pdf", dpi=300)
        plt.savefig(f"figures/gamma_sigma_conversion/{S1_base}_06_sigma_diff_vs_DEM_1.png", dpi=300)




        logger.info("Drawing Figure 7: DEM versus sigma0 difference")

        # No need to plot all points
        # Might be worth plotting a density / 2D histogram with log color scale
        sub = 100

        DEM_min = int(np.floor(fin_DEM.min()))
        DEM_max = int(np.ceil(fin_DEM.max()))
        DEM_lin = np.linspace(DEM_min, DEM_max, DEM_max-DEM_min+1)

        fig, axes = plt.subplots(2,1,sharex=True,sharey=True,figsize=((12,5)))
        axes = axes.ravel()
        axes[0].plot(fin_DEM[::sub],HH_dB_diff[::sub],'o')
        axes[1].plot(fin_DEM[::sub],HV_dB_diff[::sub],'o')

        axes[0].plot(DEM_lin, len(DEM_lin)*[0], '--', color=[1,0.6,0])
        axes[1].plot(DEM_lin, len(DEM_lin)*[0], '--', color=[1,0.6,0])

        axes[0].set_ylabel(r'$_{my}\sigma_{HH}^{0}$ (dB) - $_{GA}\sigma_{HH}^{0}$ (dB)')
        axes[1].set_ylabel(r'$_{my}\sigma_{HV}^{0}$ (dB) - $_{GA}\sigma_{HV}^{0}$ (dB)')
        axes[1].set_xlabel("DEM (dm)")

        axes[0].set_xlim([DEM_min,DEM_max])
        axes[1].set_xlim([DEM_min,DEM_max])

        plt.suptitle(f"{S1_base}")

        ###plt.savefig(f"figures/gamma_sigma_conversion/{S1_base}_07_sigma_diff_vs_DEM_2.pdf", dpi=300)
        plt.savefig(f"figures/gamma_sigma_conversion/{S1_base}_07_sigma_diff_vs_DEM_2.png", dpi=300)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

    plt.close("all")

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    main()
    
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# ---- End of <gamma_2_sigma_conversion_check.py> ----

