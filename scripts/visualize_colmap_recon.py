# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
import pycolmap
import argparse
from utils import reconstruction as sfm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script that visualizes COLMAP sparse reconstructions with the RPC interface.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("recon_dir", type=Path, help="Paths to the recon directory.")

    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    print(args)

    recon_dir = args.recon_dir
    db_path = recon_dir/'database.db'

    recon = pycolmap.Reconstruction(recon_dir/'0')
    custom_keypoints = sfm.get_custom_keypoints_from_db(db_path)
    sfm.visualize_reconstruction(recon, custom_keypoints, sg_prefix=f'{recon_dir.parent.parent.name}.{recon_dir.parent.name}')
