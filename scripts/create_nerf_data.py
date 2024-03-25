# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
import argparse
import numpy as np
import cv2
import shutil
from utils.hdri import read_hdri

TH = 240

def create_nerf_data(input_dir: Path, output_dir: Path):
    print(input_dir/'inputs')
    assert (input_dir/'inputs').exists()
    input_dir = input_dir/'inputs'

    image_output_dir = output_dir/'images'
    image_output_dir.mkdir(exist_ok=True, parents=True)
    mask_output_dir = output_dir/'masks'
    mask_output_dir.mkdir(exist_ok=True)

    camera_paths = sorted(list(input_dir.glob('camera_*.txt')))
    image_paths = sorted(list(input_dir.glob('image_*.png')))
    mask_paths = sorted(list(input_dir.glob('mask_*.png')))
    assert len(camera_paths) == len(image_paths) and len(camera_paths) == len(mask_paths) and len(image_paths)

    val_camera_paths = sorted(list(input_dir.glob('../gt_camera_*.txt')))
    val_image_paths = sorted(list(input_dir.glob('../gt_image_*.png')))
    val_mask_paths = sorted(list(input_dir.glob('../gt_mask_*.png')))
    assert len(val_camera_paths) == len(val_image_paths) and len(val_camera_paths) == len(val_mask_paths) and len(val_image_paths)

    print(f"Found {len(camera_paths)} training images and {len(val_camera_paths)} validation images")

    bounds = np.loadtxt(input_dir/'object_bounding_box.txt')
    bounds = bounds.reshape(-1,2)
    center = np.mean(bounds, axis=-1)
    radius = 0.5*np.linalg.norm(bounds.max(axis=-1) - bounds.min(axis=-1))

    exposure_train = [np.loadtxt(input_dir/'exposure.txt').item()] * len(camera_paths)

    poses = []
    cameras_sphere = {}
    for i, (cam_path, im_path, mask_path) in enumerate(zip(camera_paths, image_paths, mask_paths)):
        params = np.loadtxt(cam_path)
        K, R, t, (width, height, _) = params[:3], params[3:6], params[6], params[7].astype(int)
        Rt = np.concatenate((R,t[:,None]), axis=-1)
        w2c = np.eye(4)
        w2c[:3] = Rt
        c2w = np.linalg.inv(w2c)
        hwf = np.array([K[0][2], K[1][2], K[0,0]])[:,None]  # cx, cy, f, instead of h,w,f as in llff format
        pose = np.concatenate((c2w[:3], hwf), axis=-1)
        pose = np.concatenate([pose[:, 1:2], pose[:, 0:1], -pose[:, 2:3], pose[:, 3:4], pose[:, 4:5]], axis=1)  # transform colmap to llff format

        center_in_cam = (w2c[:3] @ np.concatenate([center, [1.0]])[:,None])[:, 0]
        near = center_in_cam[2] - radius
        far = center_in_cam[2] + radius

        pose = pose.flatten().tolist() + [near, far]
        poses.append(pose)
        assert near < far

        shutil.copy2(im_path, image_output_dir/f'{i:06d}.png')
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask[mask >= TH] = 255
        mask[mask < TH] = 0
        cv2.imwrite(str(mask_output_dir/f'{i:06d}.png'), mask)

    poses = np.array(poses)
    print("pose shape = ", poses.shape)

    exposure_val = [np.loadtxt(f).item() for f in sorted(list(input_dir.glob('../gt_exposure_*.txt')))]

    val_poses = []
    for i, (cam_path, im_path, mask_path) in enumerate(zip(val_camera_paths, val_image_paths, val_mask_paths)):
        params = np.loadtxt(cam_path)
        K, R, t, (width, height, _) = params[:3], params[3:6], params[6], params[7].astype(int)
        Rt = np.concatenate((R,t[:,None]), axis=-1)
        w2c = np.eye(4)
        w2c[:3] = Rt
        c2w = np.linalg.inv(w2c)
        hwf = np.array([K[0][2], K[1][2], K[0,0]])[:,None]  # cx, cy, f, instead of h,w,f as in llff format
        pose = np.concatenate((c2w[:3], hwf), axis=-1)
        pose = np.concatenate([pose[:, 1:2], pose[:, 0:1], -pose[:, 2:3], pose[:, 3:4], pose[:, 4:5]], axis=1)

        center_in_cam = (w2c[:3] @ np.concatenate([center, [1.0]])[:,None])[:, 0]
        near = center_in_cam[2] - radius
        far = center_in_cam[2] + radius

        pose = pose.flatten().tolist() + [near, far]
        val_poses.append(pose)
        assert near < far

        shutil.copy2(im_path, image_output_dir/f'val_{i:06d}.png')
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask[mask >= TH] = 255
        mask[mask < TH] = 0
        cv2.imwrite(str(mask_output_dir/f'val_{i:06d}.png'), mask)

    val_poses = np.array(val_poses)
    print("val pose shape = ", val_poses.shape)

    np.save(output_dir / 'poses_bounds.npy', poses)
    np.save(output_dir / 'val_poses_bounds.npy', val_poses)
    np.save(output_dir / 'exposure.npy', np.concatenate([exposure_train, exposure_val]))
    print("exposure shape = ", np.concatenate([exposure_train, exposure_val]).shape)

    # val envmap for relighting
    val_envmaps = np.stack([read_hdri(f) for f in sorted(list(input_dir.glob('../gt_env_512_rotated_*.hdr')))])
    np.save(output_dir / 'val_envmaps.npy', val_envmaps)
    print("val envmap shape = ", val_envmaps.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script that creates inputs for nerf based baselines from the inputs for a scene from the dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("input_dir", type=Path, help="Path to the train/valid/test directory of an object in the dataset. The directory should have the 'inputs' subdir.")
    parser.add_argument("output_dir", type=Path, help="Path to the output directory.")
    parser.add_argument("--overwrite", action='store_true', help="If True overwrite existing directories")

    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    print(args)

    input_dir = args.input_dir
    output_dir = args.output_dir
    if not output_dir.exists() or args.overwrite:
        create_nerf_data(input_dir, output_dir)
    else:
        print(f'{str(output_dir)} already exists!')

