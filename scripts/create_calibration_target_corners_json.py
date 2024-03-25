# SPDX-License-Identifier: Apache-2.0
import numpy as np
import json
from collections import OrderedDict
from pathlib import Path
import argparse

def main():

    parser = argparse.ArgumentParser(
        description="Writes the coordinates of the corners of the apriltags to a json file for the calibration target. The unit of the output coordinates is meter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("output", type=Path, help="Output json file.")


    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    print(args)

    rows = 9
    cols = 6
    tag_id_row_offset = 45
    tag_family = 'tagStandard41h12'
    tag_size = 0.015
    tag_distance = 2*tag_size

    result = OrderedDict()

    corners = np.array([[0,0], [tag_size, 0], [tag_size,tag_size], [0, tag_size]])

    for r in range(rows):
        for c in range(cols):
            tag_id = r*tag_id_row_offset + c
            offset = np.array([c*tag_distance, -r*tag_distance])
            for corner_i, corner in enumerate(corners):
                result[f'{tag_family}.{tag_id}.{corner_i}'] = tuple(corner + offset) + (0,)

    with open(args.output,'w') as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    main()