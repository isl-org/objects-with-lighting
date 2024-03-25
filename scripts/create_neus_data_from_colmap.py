# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
import argparse
from utils.reconstruction import colmap_to_neus

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script that creates inputs for neus from the intermediate files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("env_dir", type=Path, help="Path to the directory with the intermediate files that has the COLMAP reconstructions.")
    parser.add_argument("--overwrite", action='store_true', help="If True overwrite existing directories")

    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    print(args)

    output_dir = args.env_dir/'neus_data'
    if not output_dir.exists() or args.overwrite:
        colmap_to_neus(args.env_dir, output_dir)
    else:
        print(f'{str(output_dir)} already exists!')
