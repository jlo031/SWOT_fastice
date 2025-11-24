 # ---- This is <load_config.py> ----

"""
Load project config directory structure from json file
"""

import json
from pathlib import Path

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

with open("config/config.json", "r") as f:
    config = json.load(f)

# Project work dir
WORK_DIR = Path(config["WORK_DIR"])

# Main project dir
DATA_DIR = Path(config["DATA_DIR"])

# Sub-dirs
ROI_ANALYSIS_DIR                = DATA_DIR / config["sub_dirs"]["ROI_ANALYSIS"]
S1_DIR                          = DATA_DIR / config["sub_dirs"]["S1"]
SWOT_L2_D                       = DATA_DIR / config["sub_dirs"]["SWOT_L2_D"],
SWOT_L2_D_Expert                = DATA_DIR / config["sub_dirs"]["SWOT_L2_D_Expert"],
SWOT_L2_D_Expert_cropped        = DATA_DIR / config["sub_dirs"]["SWOT_L2_D_Expert_cropped"]
SWOT_L2_D_Unsmoothed            = DATA_DIR / config["sub_dirs"]["SWOT_L2_D_Unsmoothed"]
SWOT_L2_D_Unsmoothed_cropped    = DATA_DIR / config["sub_dirs"]["SWOT_L2_D_Unsmoothed_cropped"]
SWOT_SWATHS_ssha_sig0_coherence = DATA_DIR / config["sub_dirs"]["SWOT_SWATHS_ssha_sig0_coherence"]
SWOT_GEOTIFF_DIR                = DATA_DIR / config["sub_dirs"]["SWOT_GEOTIFF_DIR"]

# S1 sub-dirs
S1_L1_DIR    = S1_DIR / config["sub_dirs"]["S1_subdirs"]["L1"]
S1_FEAT_DIR  = S1_DIR / config["sub_dirs"]["S1_subdirs"]["FEAT"]
S1_GEO_DIR   = S1_DIR / config["sub_dirs"]["S1_subdirs"]["GEO"]
S1_RGB_DIR   = S1_DIR / config["sub_dirs"]["S1_subdirs"]["RGB"]
S1_TRAIN_DIR = S1_DIR / config["sub_dirs"]["S1_subdirs"]["TRAIN"]

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# Create directories if needed

DATA_DIR.mkdir(parents=True, exist_ok=True)

ROI_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

S1_DIR.mkdir(parents=True, exist_ok=True)

SWOT_GEOTIFF_DIR.mkdir(parents=True, exist_ok=True)

S1_L1_DIR.mkdir(parents=True, exist_ok=True)
S1_FEAT_DIR.mkdir(parents=True, exist_ok=True)
S1_GEO_DIR.mkdir(parents=True, exist_ok=True)
S1_RGB_DIR.mkdir(parents=True, exist_ok=True)
S1_TRAIN_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <load_config.py> ----
