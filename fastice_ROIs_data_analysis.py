# ---- This is <fastice_ROIs_data_analysis.py> ----

"""
Extract S1 and SWOT data from within the fast ice ROIs.
Save as pickle files.
"""

import pickle
from loguru import logger

from config.load_config import *

import pandas as pd
import numpy as np
import rasterio
from rasterio.mask import mask
from scipy import stats

import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

HH_min = -25
HH_max = 0
HV_min = -35
HV_max = -10

alpha = 0.2

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

# Define paths to pickle files
HH_fastice_stats_path   = ROI_ANALYSIS_DIR / f"fastice_ROIs_HH.pickle"
HV_fastice_stats_path   = ROI_ANALYSIS_DIR / f"fastice_ROIs_HV.pickle"
IA_fastice_stats_path   = ROI_ANALYSIS_DIR / f"fastice_ROIs_IA.pickle"
SSHA_fastice_stats_path = ROI_ANALYSIS_DIR / f"fastice_ROIs_SSHA.pickle"
SIG0_fastice_stats_path = ROI_ANALYSIS_DIR / f"fastice_ROIs_SIG0.pickle"
COH_fastice_stats_path  = ROI_ANALYSIS_DIR / f"fastice_ROIs_COH.pickle"

logger.info("Loading stats dictionaries from pickle files")

with open(HH_fastice_stats_path, 'rb') as f:
    HH_dict = pickle.load(f)

with open(HV_fastice_stats_path, 'rb') as f:
    HV_dict = pickle.load(f)

with open(IA_fastice_stats_path, 'rb') as f:
    IA_dict = pickle.load(f)

with open(SSHA_fastice_stats_path, 'rb') as f:
    SSHA_dict = pickle.load(f)

with open(SIG0_fastice_stats_path, 'rb') as f:
    SIG0_dict = pickle.load(f)

with open(COH_fastice_stats_path, 'rb') as f:
    COH_dict = pickle.load(f)

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# Extract all data from the dictionaries
# Compute basic statistics,
# Format for easy use

HH_means  = []
HH_vars   = []
HH_stds   = []
HH_labels = []

HV_means  = []
HV_vars   = []
HV_stds   = []
HV_labels = []

IA_means  = []
IA_vars   = []
IA_stds   = []
IA_labels = []

HH_all = []
HV_all = []
IA_all = []
HH_labels_all = [] 
HV_labels_all = [] 
IA_labels_all = [] 

SSHA_means      = []
SSHA_vars       = []
SSHA_stds       = []
SSHA_labels     = []
SSHA_timestamps = []
SSHA_orbit      = []

SIG0_means      = []
SIG0_vars       = []
SIG0_stds       = []
SIG0_labels     = []
SIG0_timestamps = []
SIG0_orbit      = []

COH_means      = []
COH_vars       = []
COH_stds       = []
COH_labels     = []
COH_timestamps = []
COH_orbit      = []

SSHA_all = []
SIG0_all = []
COH_all  = []
SSHA_labels_all = [] 
SIG0_labels_all = [] 
COH_labels_all  = [] 



for k in HH_dict.keys():
    logger.debug(f"Processing HH_dict key: {k}")

    HH    = HH_dict[k]["data"]
    label = HH_dict[k]["label"]

    HH_all.extend(list(HH[np.isfinite(HH)].flatten()))
    HH_labels_all.extend([label]*len(HH[np.isfinite(HH)].flatten()))

    HH_mean = np.nanmean(HH).mean()
    HH_var = np.nanvar(HH)
    HH_std = np.nanstd(HH)

    logger.debug(f"    label:    {label}")
    logger.debug(f"    mean(HH): {HH_mean:.1f}")
    logger.debug(f"    var(HH):  {HH_var:.1f}")
    logger.debug(f"    std(HH):  {HH_std:.1f}")

    HH_means.append(HH_mean)
    HH_vars.append(HH_var)
    HH_stds.append(HH_std)
    HH_labels.append(label)


for k in HV_dict.keys():
    logger.debug(f"Processing HV_dict key: {k}")

    HV    = HV_dict[k]["data"]
    label = HV_dict[k]["label"]

    HV_all.extend(list(HV[np.isfinite(HV)].flatten()))
    HV_labels_all.extend([label]*len(HV[np.isfinite(HV)].flatten()))

    HV_mean = np.nanmean(HV).mean()
    HV_var  = np.nanvar(HV)
    HV_std  = np.nanstd(HV)

    logger.debug(f"    label:    {label}")
    logger.debug(f"    mean(HV): {HV_mean:.1f}")
    logger.debug(f"    var(HV):  {HV_var:.1f}")
    logger.debug(f"    std(HV):  {HV_std:.1f}")

    HV_means.append(HV_mean)
    HV_vars.append(HV_var)
    HV_stds.append(HV_std)
    HV_labels.append(label)


for k in IA_dict.keys():
    logger.debug(f"Processing IA_dict key: {k}")

    IA    = IA_dict[k]["data"]
    label = IA_dict[k]["label"]

    IA_all.extend(list(IA[np.isfinite(IA)].flatten()))
    IA_labels_all.extend([label]*len(IA[np.isfinite(IA)].flatten()))

    IA_mean = np.nanmean(IA).mean()
    IA_var  = np.nanvar(IA)
    IA_std  = np.nanstd(IA)

    logger.debug(f"    label:    {label}")
    logger.debug(f"    mean(IA): {IA_mean:.1f}")
    logger.debug(f"    var(IA):  {IA_var:.1f}")
    logger.debug(f"    std(IA):  {IA_std:.1f}")

    IA_means.append(IA_mean)
    IA_vars.append(IA_var)
    IA_stds.append(IA_std)
    IA_labels.append(label)


for k in SSHA_dict.keys():
    logger.debug(f"Processing SSHA_dict key: {k}")

    SSHA_time = k.split("_")[14]

    SSHA    = SSHA_dict[k]["data"]
    label   = SSHA_dict[k]["label"]
    orbit   = SSHA_dict[k]["orbit"]

    SSHA_all.extend(list(SSHA[np.isfinite(SSHA)].flatten()))
    SSHA_labels_all.extend([label]*len(SSHA[np.isfinite(SSHA)].flatten()))

    SSHA_mean = np.nanmean(SSHA).mean()
    SSHA_var  = np.nanvar(SSHA)
    SSHA_std  = np.nanstd(SSHA)

    logger.debug(f"    label:    {label}")
    logger.debug(f"    mean(SSHA): {SSHA_mean:.1f}")
    logger.debug(f"    var(SSHA):  {SSHA_var:.1f}")
    logger.debug(f"    std(SSHA):  {SSHA_std:.1f}")

    SSHA_means.append(SSHA_mean)
    SSHA_vars.append(SSHA_var)
    SSHA_stds.append(SSHA_std)
    SSHA_labels.append(label)

    SSHA_timestamps.append(SSHA_time)
    SSHA_orbit.append(orbit)

for k in SIG0_dict.keys():
    logger.debug(f"Processing SIG0_dict key: {k}")

    SIG0    = SIG0_dict[k]["data"]
    label   = SIG0_dict[k]["label"]

    SIG0_all.extend(list(SIG0[np.isfinite(SIG0)].flatten()))
    SIG0_labels_all.extend([label]*len(SIG0[np.isfinite(SIG0)].flatten()))

    SIG0_mean = np.nanmean(SIG0).mean()
    SIG0_var  = np.nanvar(SIG0)
    SIG0_std  = np.nanstd(SIG0)

    logger.debug(f"    label:    {label}")
    logger.debug(f"    mean(SIG0): {SIG0_mean:.1f}")
    logger.debug(f"    var(SIG0):  {SIG0_var:.1f}")
    logger.debug(f"    std(SIG0):  {SIG0_std:.1f}")

    SIG0_means.append(SIG0_mean)
    SIG0_vars.append(SIG0_var)
    SIG0_stds.append(SIG0_std)
    SIG0_labels.append(label)


for k in COH_dict.keys():
    logger.debug(f"Processing COH_dict key: {k}")

    COH    = COH_dict[k]["data"]
    label   = COH_dict[k]["label"]

    COH_all.extend(list(COH[np.isfinite(COH)].flatten()))
    COH_labels_all.extend([label]*len(COH[np.isfinite(COH)].flatten()))

    COH_mean = np.nanmean(COH).mean()
    COH_var  = np.nanvar(COH)
    COH_std  = np.nanstd(COH)

    logger.debug(f"    label:    {label}")
    logger.debug(f"    mean(COH): {COH_mean:.1f}")
    logger.debug(f"    var(COH):  {COH_var:.1f}")
    logger.debug(f"    std(COH):  {COH_std:.1f}")

    COH_means.append(COH_mean)
    COH_vars.append(COH_var)
    COH_stds.append(COH_std)
    COH_labels.append(label)

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# Convert lists to arrays

HH_means    = np.array(HH_means)
HH_vars     = np.array(HH_vars)
HH_stds     = np.array(HH_stds)
HH_labels   = np.array(HH_labels)
HV_means    = np.array(HV_means)
HV_vars     = np.array(HV_vars)
HV_stds     = np.array(HV_stds)
HV_labels   = np.array(HV_labels)
IA_means    = np.array(IA_means)
IA_vars     = np.array(IA_vars)
IA_stds     = np.array(IA_stds)
IA_labels   = np.array(HH_labels)

HH_all        = np.array(HH_all)
HH_labels_all = np.array(HH_labels_all)
HV_all        = np.array(HV_all)
HV_labels_all = np.array(HV_labels_all)
IA_all        = np.array(IA_all)
IA_labels_all = np.array(IA_labels_all)

SSHA_means  = np.array(SSHA_means)
SSHA_vars   = np.array(SSHA_vars)
SSHA_stds   = np.array(SSHA_stds)
SSHA_labels = np.array(SSHA_labels)
SIG0_means  = np.array(SIG0_means)
SIG0_vars   = np.array(SIG0_vars)
SIG0_stds   = np.array(SIG0_stds)
SIG0_labels = np.array(SIG0_labels)
COH_means   = np.array(COH_means)
COH_vars    = np.array(COH_vars)
COH_stds    = np.array(COH_stds)
COH_labels  = np.array(COH_labels)

SSHA_orbit = np.array(SSHA_orbit)

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# Find idx for LI and DI
DI_idx = SSHA_labels==DI_label
LI_idx = SSHA_labels==LI_label

# Get SWOT stats
SSHA_LI_mean = np.mean(SSHA_means[LI_idx])
SSHA_DI_mean = np.mean(SSHA_means[DI_idx])
SSHA_LI_std = np.std(SSHA_means[LI_idx])
SSHA_DI_std = np.std(SSHA_means[DI_idx])
SSHA_LI_var = np.var(SSHA_means[LI_idx])
SSHA_DI_var = np.var(SSHA_means[DI_idx])

logger.info(f"LI SSHA in October (cm): {(100*SSHA_LI_mean):.1f}+-{(100*SSHA_LI_std):.1f}")
logger.info(f"DI SSHA in October (cm): {(100*SSHA_DI_mean):.1f}+-{(100*SSHA_DI_std):.1f}")

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# S1 Slope analysis

# Find idx for LI and DI
DI_idx = HH_labels==DI_label
LI_idx = HH_labels==LI_label

# Get LI HH and HV slopes
LI_HH_b, LI_HH_a, LI_HH_r_value, LI_HH_p_value, LI_HH_std_err = stats.linregress(IA_means[LI_idx], HH_means[LI_idx])
LI_HH_R2 = LI_HH_r_value**2
LI_HV_b, LI_HV_a, LI_HV_r_value, LI_HV_p_value, LI_HV_std_err = stats.linregress(IA_means[LI_idx], HV_means[LI_idx])
LI_HV_R2 = LI_HV_r_value**2

# Get DI HH and HV slopes
DI_HH_b, DI_HH_a, DI_HH_r_value, DI_HH_p_value, DI_HH_std_err = stats.linregress(IA_means[DI_idx], HH_means[DI_idx])
DI_HH_R2 = DI_HH_r_value**2
DI_HV_b, DI_HV_a, DI_HV_r_value, DI_HV_p_value, DI_HV_std_err = stats.linregress(IA_means[DI_idx], HV_means[DI_idx])
DI_HV_R2 = DI_HV_r_value**2

logger.info(f"LI HH slope: {LI_HH_b:.2f}")
logger.info(f"LI HV slope: {LI_HV_b:.2f}")
logger.info(f"DI HH slope: {DI_HH_b:.2f}")
logger.info(f"DI HV slope: {DI_HV_b:.2f}")

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# Project all S1 data points to reference angle

IA_ref = 30

# Get all LI data
HH_LI = HH_all[HH_labels_all==LI_label]
HV_LI = HV_all[HV_labels_all==LI_label]
IA_LI = IA_all[IA_labels_all==LI_label]

# Get all DI data
HH_DI = HH_all[HH_labels_all==DI_label]
HV_DI = HV_all[HV_labels_all==DI_label]
IA_DI = IA_all[IA_labels_all==DI_label]

# Project along slopes to IA_ref
HH_LI_proj = HH_LI - LI_HH_b * (IA_LI-IA_ref)
HV_LI_proj = HV_LI - LI_HV_b * (IA_LI-IA_ref)
HH_DI_proj = HH_DI - DI_HH_b * (IA_DI-IA_ref)
HV_DI_proj = HV_DI - DI_HV_b * (IA_DI-IA_ref)

# Get means
HH_LI_mean_IA_ref = np.mean(HH_LI_proj)
HV_LI_mean_IA_ref = np.mean(HV_LI_proj)
HH_DI_mean_IA_ref = np.mean(HH_DI_proj)
HV_DI_mean_IA_ref = np.mean(HV_DI_proj)

# Get std
HH_LI_std_IA_ref = np.std(HH_LI_proj)
HV_LI_std_IA_ref = np.std(HV_LI_proj)
HH_DI_std_IA_ref = np.std(HH_DI_proj)
HV_DI_std_IA_ref = np.std(HV_DI_proj)

# Get var
HH_LI_var_IA_ref = np.var(HH_LI_proj)
HV_LI_var_IA_ref = np.var(HV_LI_proj)
HH_DI_var_IA_ref = np.var(HH_DI_proj)
HV_DI_var_IA_ref = np.var(HV_DI_proj)

logger.info(f"LI HH at IA_ref={IA_ref}: {HH_LI_mean_IA_ref:.1f}+-{HH_LI_std_IA_ref:.1f}")
logger.info(f"LI HV at IA_ref={IA_ref}: {HV_LI_mean_IA_ref:.1f}+-{HV_LI_std_IA_ref:.1f}")
logger.info(f"DI HH at IA_ref={IA_ref}: {HH_DI_mean_IA_ref:.1f}+-{HH_DI_std_IA_ref:.1f}")
logger.info(f"DI HV at IA_ref={IA_ref}: {HV_DI_mean_IA_ref:.1f}+-{HV_DI_std_IA_ref:.1f}")


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# Plot fast ice slopes

# Prepare to plot linear fit
IA_min = 18
IA_max = 45
IA_lin = np.linspace(IA_min, IA_max, IA_max-IA_min+1)
DI_HH_lin = DI_HH_a + DI_HH_b * IA_lin
DI_HV_lin = DI_HV_a + DI_HV_b * IA_lin
LI_HH_lin = LI_HH_a + LI_HH_b * IA_lin
LI_HV_lin = LI_HV_a + LI_HV_b * IA_lin

LI_HH_lin_lower = LI_HH_lin - HH_LI_std_IA_ref
LI_HH_lin_upper = LI_HH_lin + HH_LI_std_IA_ref
DI_HH_lin_lower = DI_HH_lin - HH_DI_std_IA_ref
DI_HH_lin_upper = DI_HH_lin + HH_DI_std_IA_ref
LI_HV_lin_lower = LI_HV_lin - HV_LI_std_IA_ref
LI_HV_lin_upper = LI_HV_lin + HV_LI_std_IA_ref
DI_HV_lin_lower = DI_HV_lin - HV_DI_std_IA_ref
DI_HV_lin_upper = DI_HV_lin + HV_DI_std_IA_ref

fig, axes = plt.subplots(2,1,sharex=True,figsize=((10,8)))
axes = axes.ravel()

# LI polygon means
axes[0].errorbar(IA_means[LI_idx], HH_means[LI_idx], yerr=HH_stds[LI_idx], fmt="o", capsize=3, elinewidth=1, color=class_colors[0])
axes[1].errorbar(IA_means[LI_idx], HV_means[LI_idx], yerr=HV_stds[LI_idx], fmt="o", capsize=3, elinewidth=1, color=class_colors[0])

# DI polygon means
axes[0].errorbar(IA_means[DI_idx], HH_means[DI_idx], yerr=HH_stds[DI_idx], fmt="o", capsize=3, elinewidth=1, color=class_colors[1])
axes[1].errorbar(IA_means[DI_idx], HV_means[DI_idx], yerr=HV_stds[DI_idx], fmt="o", capsize=3, elinewidth=1, color=class_colors[1])

# Linear fit
axes[0].plot(IA_lin, LI_HH_lin, "--", color=class_colors[0])
axes[1].plot(IA_lin, LI_HV_lin, "--", color=class_colors[0])
axes[0].plot(IA_lin, DI_HH_lin, "--", color=class_colors[1])
axes[1].plot(IA_lin, DI_HV_lin, "--", color=class_colors[1])

# Linear fit std
axes[0].fill_between(IA_lin, LI_HH_lin_lower, LI_HH_lin_upper, color=class_colors[0], alpha=alpha)
axes[0].fill_between(IA_lin, DI_HH_lin_lower, DI_HH_lin_upper, color=class_colors[1], alpha=alpha)
axes[1].fill_between(IA_lin, LI_HV_lin_lower, LI_HV_lin_upper, color=class_colors[0], alpha=alpha)
axes[1].fill_between(IA_lin, DI_HV_lin_lower, DI_HV_lin_upper, color=class_colors[1], alpha=alpha)

# Labels
axes[1].set_xlabel("IA (deg)")
axes[0].set_ylabel("HH (dB)")
axes[1].set_ylabel("HV (dB)")

# IA min/max
axes[0].set_xlim((IA_min,IA_max))

axes[0].legend([f"Level Ice\nb={LI_HH_b:.2f}", f"Deformed Ice\nb={DI_HH_b:.2f}"])
axes[1].legend([f"Level Ice\nb={LI_HV_b:.2f}", f"Deformed Ice\nb={DI_HV_b:.2f}"])

# Add mean value at IA_ref
axes[0].errorbar(IA_ref, HH_LI_mean_IA_ref, yerr=HH_LI_std_IA_ref, fmt="o", capsize=5, elinewidth=3, color=class_colors[0]*0.7)
axes[0].errorbar(IA_ref, HH_DI_mean_IA_ref, yerr=HH_DI_std_IA_ref, fmt="o", capsize=5, elinewidth=3, color=class_colors[1]*0.7)
axes[1].errorbar(IA_ref, HV_LI_mean_IA_ref, yerr=HV_LI_std_IA_ref, fmt="o", capsize=5, elinewidth=3, color=class_colors[0]*0.7)
axes[1].errorbar(IA_ref, HV_DI_mean_IA_ref, yerr=HV_DI_std_IA_ref, fmt="o", capsize=5, elinewidth=3, color=class_colors[1]*0.7)

# Figure title
fig.suptitle("Sentinel-1 fastice backscatter ($\sigma_0$) in Oct 2024")

plt.tight_layout()

plt.savefig(WORK_DIR/"figures"/"fastice_S1_slopes.png", dpi=300, transparent=True)
plt.savefig(WORK_DIR/"figures"/"fastice_S1_slopes.pdf", dpi=300, transparent=True)

##plt.close('all')
plt.show()

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# Plot SSHA vs time

# Create time index
time_index = pd.to_datetime(SSHA_timestamps, format="%Y%m%dT%H%M%S")

# Select time index for each ice type
time_index_LI = time_index[SSHA_labels==LI_label]
time_index_DI = time_index[SSHA_labels==DI_label]

# Select time series of mean and std for each ice type
SSHA_means_LI = SSHA_means[SSHA_labels==LI_label]
SSHA_means_DI = SSHA_means[SSHA_labels==DI_label]
SSHA_stds_LI  = SSHA_stds[SSHA_labels==LI_label]
SSHA_stds_DI  = SSHA_stds[SSHA_labels==DI_label]

# Select orbits for each ice type
SSHA_orbit_LI = SSHA_orbit[SSHA_labels==LI_label]
SSHA_orbit_DI = SSHA_orbit[SSHA_labels==DI_label]

# Prepare plot of mean SSHA
time_min_max = pd.to_datetime(['20241001T000000','20241031T235959'], format="%Y%m%dT%H%M%S")
SSHD_LI_line = 2*[SSHA_LI_mean]
SSHD_DI_line = 2*[SSHA_DI_mean]
SSHD_LI_upper = 2*[SSHA_LI_mean+SSHA_LI_std]
SSHD_LI_lower = 2*[SSHA_LI_mean-SSHA_LI_std]
SSHD_DI_upper = 2*[SSHA_DI_mean+SSHA_DI_std]
SSHD_DI_lower = 2*[SSHA_DI_mean-SSHA_DI_std]

fig, ax = plt.subplots(1,1, figsize=((14,6)))
ax.errorbar(time_index_LI[SSHA_orbit_LI=="ascending"], SSHA_means_LI[SSHA_orbit_LI=="ascending"], yerr=SSHA_stds_LI[SSHA_orbit_LI=="ascending"], fmt="o", capsize=3, elinewidth=1, color=class_colors[0]*1.2)
ax.errorbar(time_index_LI[SSHA_orbit_LI=="descending"], SSHA_means_LI[SSHA_orbit_LI=="descending"], yerr=SSHA_stds_LI[SSHA_orbit_LI=="descending"], fmt="o", capsize=3, elinewidth=1, color=class_colors[0]*0.8)
ax.errorbar(time_index_DI[SSHA_orbit_DI=="ascending"], SSHA_means_DI[SSHA_orbit_DI=="ascending"], yerr=SSHA_stds_DI[SSHA_orbit_DI=="ascending"], fmt="o", capsize=3, elinewidth=1, color=class_colors[1]*1.2)
ax.errorbar(time_index_DI[SSHA_orbit_DI=="descending"], SSHA_means_DI[SSHA_orbit_DI=="descending"], yerr=SSHA_stds_DI[SSHA_orbit_DI=="descending"], fmt="o", capsize=3, elinewidth=1, color=class_colors[1]*0.8)

# Add legend for the errorbar plots
ax.legend([f"LI, ascending", f"LI, descending", f"DI, ascending", f"DI, descending"])

# Mean and std
ax.plot(time_min_max, SSHD_LI_line, "--", color=class_colors[0])
ax.plot(time_min_max, SSHD_DI_line, "--", color=class_colors[1])
ax.fill_between(time_min_max, SSHD_LI_lower, SSHD_LI_upper, color=class_colors[0], alpha=alpha)
ax.fill_between(time_min_max, SSHD_DI_lower, SSHD_DI_upper, color=class_colors[1], alpha=alpha)

ax.set_xlabel("Time")
ax.set_ylabel("SSHA (m)")

# IA min/max
ax.set_xlim(time_min_max)

# Figure title
fig.suptitle("Fastice SSHA in Oct 2024")

plt.tight_layout()

plt.savefig(WORK_DIR/"figures"/"fastice_SSHA_october.png", dpi=300, transparent=True)
plt.savefig(WORK_DIR/"figures"/"fastice_SSHA_october.pdf", dpi=300, transparent=True)

plt.close('all')
plt.show()

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# ---- End of <fastice_ROIs_data_analysis.py> ----


















