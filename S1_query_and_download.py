# ---- This is <S1_query_and_download.py> ----

"""
Query CDSE for Sentinel-1 products.
Download products according to defined specifications.
"""

import pathlib
import sys

from loguru import logger

from shapely import wkt
from shapely.geometry import shape

import numpy as np

import CDSE.utils as CDSE_utils
import CDSE.json_utils as CDSE_json
import CDSE.search_and_download as CDSE_sd

from config.load_config import *

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# Set loglevel
loglevel = "DEBUG"

logger.remove()
logger.add(sink=sys.stdout, level=loglevel)

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

logger.debug(f"DATA_DIR:    {DATA_DIR}")
logger.debug(f"S1_DIR:      {S1_DIR}")
logger.debug(f"S1_L1_DIR:   {S1_L1_DIR}")
logger.debug(f"S1_FEAT_DIR: {S1_FEAT_DIR}")
logger.debug(f"S1_GEO_DIR:  {S1_GEO_DIR}")

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# Set search parameters
sensor       = 'Sentinel-1'
start_date   = '2024-10-01'
end_date     = '2024-10-31'
product_type = 'GRD'
start_time   = '00:00:01'
end_time     = '23:59:59'

# Read CDSE user credentials from '.env'
username, password = CDSE_utils.get_user_and_passwd()

# Define ROI choices
ROIs = [
    "SWOT_S1_test_site"
]

# Specify download
download_all             = True
download_minimum_overlap = True
minimum_overlap          = 0.2

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# Build path to ROI dir
ROI_DIR = WORK_DIR / "ROIs"

for ROI in ROIs:

    logger.info(f'Processing ROI: {ROI}')

    # build path to ROI json file
    json_path = ROI_DIR / f'{ROI}_epsg4326.geojson'
    logger.debug(f'json_path: {json_path}')

    response_json = CDSE_sd.search_CDSE_catalogue(
        sensor,
        json_path,
        start_date,
        end_date,
        start_time = start_time,
        end_time = end_time,
        sensor_mode = None,
        product_type = product_type,
        processing_level=None,
        max_results=1000,
        expand_attributes=True,
        loglevel=loglevel,
    )

    product_list = response_json['value']

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# Download products

if download_all:

    logger.info("Downloading all products (non-COG, dual-pol)")

    download_minimum_overlap = False

    # loop over products and get overlaps
    for p in product_list:
    
        logger.debug(f"{p['Name']}")

        if not 'COG' in p['Name'] and '1SDH' in p['Name']:
            logger.info(f"{p['Name']}")
            logger.info("    Downloading this product")
        
            CDSE_sd.download_product_from_cdse(p, S1_L1_DIR, username, password)

# --------------------------------------------------------------------------- #

if download_minimum_overlap:

    logger.info("Downloading products with minimum ROI overlap (non-COG, dual-pol")

    # Get ROI area as polygon
    search_polygon =  shape(CDSE_json.read_geojson(json_path)['features'][0]['geometry'])

    # loop over products and get overlaps
    for p in product_list:
    
        # get footprint polygon
        footprint_polygon = wkt.loads(p['Footprint'].split(";")[1])

        perc_overlap = CDSE_json.get_polygon_overlap(footprint_polygon, search_polygon)

        logger.debug(f"{p['Name']}")
        logger.debug(f"    Percentage overlap: {np.round(perc_overlap,2)}")


        if not 'COG' in p['Name'] and '1SDH' in p['Name'] and perc_overlap > minimum_overlap:
            logger.info(f"{p['Name']}")
            logger.info("    Downloading this product")
        
            CDSE_sd.download_product_from_cdse(p, S1_L1_DIR, username, password)

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# ---- End of <S1_query_and_download.py> ----
