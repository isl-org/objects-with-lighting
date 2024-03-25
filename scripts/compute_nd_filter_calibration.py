# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
import argparse
import numpy as np
from utils import hdri
from utils.constants import *


def main():

    parser = argparse.ArgumentParser(
        description="Script that computes the f-stop reduction of the nd filter sheet used in the dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    args = parser.parse_args()

    print('Loading images')
    raws_with_filter =  [hdri.read_raw_and_meta(x) for x in (CALIBRATION_PATH/'ndfilter'/'with_1_filter_sheet').glob('*.CR3')]
    raws_without_filter =  [hdri.read_raw_and_meta(x) for x in (CALIBRATION_PATH/'ndfilter'/'without').glob('*.CR3')]

    print('Computing HDRI 1')
    hdr_with_filter = hdri.compute_hdri_from_raws(raws_with_filter)
    print('Computing HDRI 2')
    hdr_without_filter = hdri.compute_hdri_from_raws(raws_without_filter)

    scale = hdr_with_filter.mean()/hdr_without_filter.mean()
    fstop_reduction = -np.log2(scale)
    print('mean without filter', hdr_without_filter.mean())
    print('   mean with filter', hdr_with_filter.mean())
    print('scale value', scale)
    print('fstop reduction', fstop_reduction)

    np.savetxt(CALIBRATION_PATH/'ndfilter'/'fstop_reduction.txt', [fstop_reduction])

    
if __name__ == '__main__':
    main()

