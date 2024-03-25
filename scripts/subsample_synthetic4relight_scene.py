# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
import argparse
import numpy as np
import cv2
import shutil
import json
from tqdm import tqdm
from utils.reconstruction import visualize_camera
from utils.hdri import read_hdri, convert_to_8bit, simple_tonemap, write_hdri
from utils.equirectangular import rotate_envmap_simple

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Downsamples scenes of the converted Synthetic4Relight dataset and predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("input_dir", type=Path, help="Path to a scene of the converted synth4relight dataset.")
    parser.add_argument("output_dir", type=Path, help="Path to the output directory.")
    parser.add_argument("--overwrite", action='store_true', help="If True overwrite existing directories")

    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    print(args)

    output_dir = args.output_dir
    if output_dir.exists() and not args.overwrite:
        print(f'{str(output_dir)} already exists!')
        sys.exit(1)

    assert args.input_dir.exists()
    
    output_dir.mkdir(exist_ok=True, parents=True)

    globs = ('pr_image_????.png', 'pr_image_????.exr',  'pr_image_????.npy', 'gt_image_????.png', 'gt_mask_????.png', 'gt_camera_????.txt', 'gt_exposure_????.txt', 'gt_world_to_env_????.txt', )
    for g in globs:
        files = sorted(list(args.input_dir.glob(g)))
        if files:
            assert len(files) == 600, f'glob {g}, len {len(files)}'
            envs_all_files = np.array_split(files,3)
            envs_subsampled = [x[::16] for x in envs_all_files]

            idx = 0
            for env in envs_subsampled:
                for src in env:
                    dst = args.output_dir/g.replace('????', f'{idx:04d}')
                    print(str(src),'->', str(dst))
                    shutil.copy2(src, dst)
                    idx += 1


    globs = ('gt_env_????.hdr', 'gt_env_512_rotated_????.hdr')
    for g in globs:
        files = sorted(list(args.input_dir.glob(g)))
        if files:
            assert len(files) == 600, f'glob {g}, len {len(files)}'
            envs_all_files = np.array_split(files,3)
            envs_subsampled = [x[::16] for x in envs_all_files]

            idx = 0
            for env in envs_subsampled:
                copy = True
                for src in env:
                    dst = args.output_dir/g.replace('????', f'{idx:04d}')
                    print(str(src),'->', str(dst))
                    if copy:
                        shutil.copy2(src, dst)
                        copy = False
                        copy_idx = idx
                    else:
                        dst.symlink_to(g.replace('????', f'{copy_idx:04d}'))
                    idx += 1

    if (args.input_dir/'inputs').exists():
        shutil.copytree(args.input_dir/'inputs', args.output_dir/'inputs')

    if (args.input_dir/'neus_mesh.ply').exists():
        shutil.copy(args.input_dir/'neus_mesh.ply', args.output_dir)
