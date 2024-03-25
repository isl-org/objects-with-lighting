# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
import argparse
import numpy as np
import cv2
import shutil
from utils.reconstruction_neus import write_neus_config_file

    
def create_neus_data(input_dir: Path, output_dir: Path, use_masks: bool=False):
    print(input_dir/'inputs')
    assert (input_dir/'inputs').exists()
    input_dir = input_dir/'inputs'
    
    image_output_dir = output_dir/'image'
    image_output_dir.mkdir(exist_ok=True, parents=True)
    mask_output_dir = output_dir/'mask'
    mask_output_dir.mkdir(exist_ok=True)

    camera_paths = sorted(list(input_dir.glob('camera_*.txt')))
    image_paths = sorted(list(input_dir.glob('image_*.png')))
    assert len(camera_paths) == len(image_paths) and len(image_paths)

    bounds = np.loadtxt(input_dir/'object_bounding_box.txt')
    bounds = bounds.reshape(-1,2)
    center = np.mean(bounds, axis=-1)
    radius = 0.5*np.linalg.norm(bounds.max(axis=-1) - bounds.min(axis=-1))

    scale_mat = np.eye(4)
    scale_mat[:3,:3] = radius*np.eye(3)
    scale_mat[:3,3] = center

    scale_mat_inv = np.eye(4)
    scale_mat_inv[:3,:3] = 1/radius*np.eye(3)
    scale_mat_inv[:3,3] = -center/radius

    cameras_sphere = {}
    for i, (cam_path, im_path) in enumerate(zip(camera_paths, image_paths)):
        params = np.loadtxt(cam_path)
        K, R, t, (width, height, _) = params[:3], params[3:6], params[6], params[7].astype(int)
        Rt = np.concatenate((R,t[:,None]), axis=-1)
        P = K @ Rt
        world_mat = np.eye(4)
        world_mat[:3,:] = P

        cameras_sphere[f'world_mat_{i}'] = world_mat.astype(np.float64)
        cameras_sphere[f'world_mat_inv_{i}'] = np.linalg.inv(world_mat).astype(np.float64)
        cameras_sphere[f'scale_mat_{i}'] = scale_mat
        cameras_sphere[f'scale_mat_inv_{i}'] = scale_mat_inv

        shutil.copy2(im_path, image_output_dir/f'{i:06d}.png')

        if use_masks:
            mask_path = im_path.parent/im_path.name.replace('image_', 'mask_binary_')
            shutil.copy2(mask_path, mask_output_dir/f'{i:06d}.png')
        else:
            dummy_mask = np.full(shape=(height, width, 3), dtype=np.uint8, fill_value=255)
            cv2.imwrite(str(mask_output_dir/f'{i:06d}.png'), dummy_mask)
    
    np.savez(output_dir/'cameras_sphere.npz', **cameras_sphere)
    write_neus_config_file(output_dir, use_masks)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script that creates inputs for neus from the inputs for a scene from the dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("input_dir", type=Path, help="Path to the train/valid/test directory of an object in the dataset. The directory should have the 'inputs' subdir.")
    parser.add_argument("output_dir", type=Path, help="Path to the output directory.")
    parser.add_argument("--masks", action='store_true', help="If True enables masks for the reconstruction")
    parser.add_argument("--overwrite", action='store_true', help="If True overwrite existing directories")

    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    print(args)

    input_dir = args.input_dir
    output_dir = args.output_dir
    if not output_dir.exists() or args.overwrite:
        create_neus_data(input_dir, output_dir, args.masks)
    else:
        print(f'{str(output_dir)} already exists!')

