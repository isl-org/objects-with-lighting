# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path

DEFAULT_GAMMA = 2.2

ROOT_PATH = Path(__file__).resolve().parent.parent
CALIBRATION_PATH = ROOT_PATH/'calibration'
SOURCE_DATA_PATH = ROOT_PATH/'source_data'
INTERMEDIATE_DATA_PATH = ROOT_PATH/'intermediate_data'
DATASET_PATH = ROOT_PATH/'dataset'

COLMAP_BIN = os.environ.get('COLMAP_BIN')
if not COLMAP_BIN:
    COLMAP_BIN = str(Path.home()/'projects/colmap/build/src/exe/colmap')