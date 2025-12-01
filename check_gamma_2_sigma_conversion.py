# ---- This is <check_gamma_2_sigma_conversion.py> ----

"""
Load GA processed S1 data and test conversions between gamma0 and sigma0
"""

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

S1_base    = "S1A_EW_GRDM_1SDH_20241021T153656_20241021T153800_056202_06E14B_37D0"
S1_base    = "S1A_EW_GRDM_1SDH_20241028T153049_20241028T153140_056304_06E555_015D"
S1_base    = "S1A_EW_GRDM_1SDH_20241009T153800_20241009T153900_056027_06DA59_ACFF"


# Get list of all GA tif files in data folder
GA_file_list = [ f for f in os.listdir(GA_DATA_DIR/S1_base) if f.endswith(".tif") ]

# Get S1_GA_base
S1_GA_base = GA_file_list[0][0:27]


HH_gamma0_dB_path = GA_DATA_DIR / S1_base / f"{S1_GA_base}_HH_grd_mli_gamma0-rtc_geo_db_3031.tif"
HV_gamma0_dB_path = GA_DATA_DIR / S1_base / f"{S1_GA_base}_HV_grd_mli_gamma0-rtc_geo_db_3031.tif"
HH_gamma0_path    = GA_DATA_DIR / S1_base / f"{S1_GA_base}_HH_grd_mli_gamma0-rtc_geo_3031.tif"
HV_gamma0_path    = GA_DATA_DIR / S1_base / f"{S1_GA_base}_HV_grd_mli_gamma0-rtc_geo_3031.tif"

gamma_sigma_ratio_path = GA_DATA_DIR / S1_base / f"{S1_GA_base}_gs_ratio_geo_3031.tif"
inc_path = GA_DATA_DIR / S1_base / f"{S1_GA_base}_inc_geo_3031.tif"


with rasterio.open(HH_gamma0_dB_path) as src:
    HH_gamma0_dB = src.read(1)

with rasterio.open(HH_gamma0_path) as src:
    HH_gamma0 = src.read(1)

with rasterio.open(HV_gamma0_dB_path) as src:
    HV_gamma0_dB = src.read(1)

with rasterio.open(HV_gamma0_path) as src:
    HV_gamma0 = src.read(1)

with rasterio.open(gamma_sigma_ratio_path) as src:
    gamma_sigma_ratio = src.read(1)

with rasterio.open(inc_path) as src:
    inc = src.read(1)

# Convert IA to deg
inc_deg = np.rad2deg(inc)

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# Plot a quicklook with gamma0_dB

sub_ql = 10

fig, axes = plt.subplots(1,2,sharex=True,sharey=True,figsize=((12,5)))
axes = axes.ravel()
axes[0].imshow(HH_gamma0_dB[::sub_ql,::sub_ql], vmin=-22,vmax=-8, cmap="gray")
axes[1].imshow(HV_gamma0_dB[::sub_ql,::sub_ql], vmin=-32,vmax=-18, cmap="gray")

axes[0].set_title("HH")
axes[1].set_title("HV")

plt.suptitle(f"Quicklook: {S1_base}")

plt.savefig(f"figures/{S1_base}_quicklook.pdf", dpi=300)
plt.close('all')

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

"""
# Test dB conversion for gamma0
my_HH_gamma0_dB = 10*np.log10(HH_gamma0)
my_HV_gamma0_dB = 10*np.log10(HV_gamma0)

# Check individual pixels
logger.info(f"GA HH_gamma0_dB: {HH_gamma0_dB[9000:9005,9000]}")
logger.info(f"my HH_gamma0_dB: {my_HH_gamma0_dB[9000:9005,9000]}")
logger.info(f"GA HV_gamma0_dB: {HV_gamma0_dB[9000:9005,9000]}")
logger.info(f"my HV_gamma0_dB: {my_HV_gamma0_dB[9000:9005,9000]}")
"""

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# Clear space
my_HH_gamma0_dB = my_HV_gamma0_dB = HH_gamma0_dB = HV_gamma0_dB = 0

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# Convert gamma0 to sigma0

# With IA
my_HH_sigma0 = HH_gamma0 * np.cos(inc)
my_HV_sigma0 = HV_gamma0 * np.cos(inc)

# With conversion layer
GA_HH_sigma0 = HH_gamma0 * gamma_sigma_ratio
GA_HV_sigma0 = HV_gamma0 * gamma_sigma_ratio

# Check individual pixels
# cos(IA) and conversion layer should be identical
logger.info(f"GA HH_sigma0_dB:   {10*np.log10(GA_HH_sigma0[9000:9005,9000])}")
logger.info(f"my HH_sigma0_dB:   {10*np.log10(my_HH_sigma0[9000:9005,9000])}")
logger.info(f"GA HV_sigma0_dB:   {10*np.log10(GA_HV_sigma0[9000:9005,9000])}")
logger.info(f"my HV_sigma0_dB:   {10*np.log10(my_HV_sigma0[9000:9005,9000])}")
logger.info(f"cos(IA):           {np.cos(inc)[9000:9005,9000]}")
logger.info(f"gamma_sigma_ratio: {gamma_sigma_ratio[9000:9005,9000]}")

HH_gamma0 = my_gamma0 = 0 

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# Select all finite (non nan) values
IA           = inc_deg[np.isfinite(my_HH_sigma0)]
my_HH_sigma0 =  my_HH_sigma0[np.isfinite(my_HH_sigma0)]
GA_HH_sigma0 =  GA_HH_sigma0[np.isfinite(GA_HH_sigma0)]
my_HV_sigma0 =  my_HV_sigma0[np.isfinite(my_HV_sigma0)]
GA_HV_sigma0 =  GA_HV_sigma0[np.isfinite(GA_HV_sigma0)]


my_HH_sigma0_dB = 10*np.log10(my_HH_sigma0)
GA_HH_sigma0_dB = 10*np.log10(GA_HH_sigma0)
my_HV_sigma0_dB = 10*np.log10(my_HV_sigma0)
GA_HV_sigma0_dB = 10*np.log10(GA_HV_sigma0)

# Compute differences of dB values for all pixels
HH_dB_diff = my_HH_sigma0_dB - GA_HH_sigma0_dB
HV_dB_diff = my_HV_sigma0_dB - GA_HV_sigma0_dB

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# Plot histograms of differences

fig, axes = plt.subplots(1,2,sharex=True,sharey=True,figsize=((12,5)))
axes = axes.ravel()
axes[0].hist(HH_dB_diff, bins=50, range=(-1,1))
axes[1].hist(HV_dB_diff, bins=50, range=(-1,1))
axes[0].set_yscale('log')
axes[1].set_yscale('log')

axes[0].set_xlabel(r'$_{my}\sigma_{HH}^{0}$ (dB) - $_{GA}\sigma_{HH}^{0}$ (dB)')
axes[1].set_xlabel(r'$_{my}\sigma_{HV}^{0}$ (dB) - $_{GA}\sigma_{HV}^{0}$ (dB)')
axes[0].set_ylabel("Counts")

plt.suptitle(f"{S1_base}")

plt.savefig(f"figures/{S1_base}_sigma_diff_histograms.pdf", dpi=300)
plt.close('all')

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# Plot histograms of differences, grouped by IA

IA_threshold_1 = 30
IA_threshold_2 = 40

IA_low  = (IA <= IA_threshold_1)
IA_mid  = (IA > IA_threshold_1) & (IA < IA_threshold_2)
IA_high = (IA >= IA_threshold_2)

HH_dB_diff_IA_low  = HH_dB_diff[IA_low]
HH_dB_diff_IA_mid  = HH_dB_diff[IA_mid]
HH_dB_diff_IA_high = HH_dB_diff[IA_high]
HV_dB_diff_IA_low  = HV_dB_diff[IA_low]
HV_dB_diff_IA_mid  = HV_dB_diff[IA_mid]
HV_dB_diff_IA_high = HV_dB_diff[IA_high]


fig, axes = plt.subplots(3,2,sharex=True,sharey=True,figsize=((10,12)))
axes = axes.ravel()
axes[0].hist(HH_dB_diff_IA_low, bins=50, range=(-1,1))
axes[2].hist(HH_dB_diff_IA_mid, bins=50, range=(-1,1))
axes[4].hist(HH_dB_diff_IA_high, bins=50, range=(-1,1))
axes[1].hist(HV_dB_diff_IA_low, bins=50, range=(-1,1))
axes[3].hist(HV_dB_diff_IA_mid, bins=50, range=(-1,1))
axes[5].hist(HV_dB_diff_IA_high, bins=50, range=(-1,1))

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

plt.savefig(f"figures/{S1_base}_sigma_diff_histograms_per_IA_range.pdf", dpi=300)
plt.close('all')

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# Compute linear regression of differences with IA
HH_b, HH_a, HH_r_value, HH_p_value, HH_std_err = stats.linregress(IA, HH_dB_diff)
HV_b, HV_a, HV_r_value, HV_p_value, HV_std_err = stats.linregress(IA, HV_dB_diff)
HH_R2 = HH_r_value**2
HV_R2 = HV_r_value**2

# Prepare to plot
IA_min = int(np.floor(IA.min()))
IA_max = int(np.ceil(IA.max()))
IA_lin = np.linspace(IA_min, IA_max, IA_max-IA_min+1)
HH_lin = HH_a + HH_b * IA_lin
HV_lin = HV_a + HV_b * IA_lin

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# Plot differences against IA

sub = 100

fig, axes = plt.subplots(2,1,sharex=True,sharey=True,figsize=((12,5)))
axes = axes.ravel()
axes[0].plot(IA[::sub],HH_dB_diff[::sub],'o')
axes[1].plot(IA[::sub],HV_dB_diff[::sub],'o')

axes[0].plot(IA_lin, len(IA_lin)*[0], '--', color=[1,0.6,0])
axes[1].plot(IA_lin, len(IA_lin)*[0], '--', color=[1,0.6,0])
#axes[0].plot([18,18], [-2,2], '--', color=[1,0.6,0])
#axes[0].plot([45,45], [-2,2], '--', color=[1,0.6,0])


axes[0].set_ylabel(r'$_{my}\sigma_{HH}^{0}$ (dB) - $_{GA}\sigma_{HH}^{0}$ (dB)')
axes[1].set_ylabel(r'$_{my}\sigma_{HV}^{0}$ (dB) - $_{GA}\sigma_{HV}^{0}$ (dB)')
axes[1].set_xlabel("IA (deg)")

axes[0].set_xlim([IA_min,IA_max])
axes[1].set_xlim([IA_min,IA_max])

plt.suptitle(f"{S1_base}")

plt.savefig(f"figures/{S1_base}_sigma_diff_vs_IA.pdf", dpi=300)
plt.close('all')


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# ---- End of <check_gamma_2_sigma_conversion.py> ----


















