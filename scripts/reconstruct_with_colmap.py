# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
import pycolmap
import argparse
import numpy as np
from addict import Dict
from utils import reconstruction as sfm
from utils.features import CustomKeypoint
import json
from collections import defaultdict
import shutil
from utils.constants import *


def load_object_keypoints(path: Path):
    """Helper to read the 'object_keypoint.json' file and convert to a list of CustomKeypoints with images filenames as key."""
    with open(path, 'r') as f:
        data = json.load(f)
    img_kps_dict = defaultdict(list)
    for kp_id, img_pos_dict in data.items():
        for img, pos in img_pos_dict.items():
            img_kps_dict[Path(img).with_suffix('.jpg').name].append(CustomKeypoint(f'obj_{kp_id}', (pos['x'], pos['y'])))
    return img_kps_dict
    


def reconstruct_scene(args):
    """Reconstruct the scene with COLMAP."""
    recon_dir = args.scene_dir.absolute()/'recon'
    object_name = args.scene_dir.absolute().parent.name
    env_name = args.scene_dir.absolute().name
    print(object_name, env_name)
    
    if recon_dir.exists() and args.overwrite:
        shutil.rmtree(recon_dir)
    
    db_path = recon_dir/'database.db'
    img_dir = args.scene_dir/'images'

    pose_priors = args.scene_dir/'pose_priors.npz'
    if not pose_priors.exists():
        pose_priors = None
        
    # this will create the recon_dir
    sfm.init_colmap_reconstruction(recon_dir, img_dir, args.overwrite, pose_priors=pose_priors)

    # set intrinsics
    intrinsics = Dict()
    for k in ('dist_coeffs', 'K', 'width_height'):
        intrinsics[k] = np.loadtxt(args.scene_dir/f'{k}.txt')
    intrinsics['width'], intrinsics['height'] = intrinsics['width_height']
    sfm.update_cam_with_opencv_params(db_path, intrinsics.K, intrinsics.dist_coeffs)

    sfm.add_apriltag_keypoints_to_colmap_reconstruction(recon_dir, img_dir, debug=True)

    # add object keypoints
    object_keypoints_path = args.scene_dir.parent/'object_keypoints.json'
    print(object_keypoints_path)
    assert object_keypoints_path.exists()
    object_keypoints = load_object_keypoints(object_keypoints_path)
    sfm.add_custom_keypoints_to_colmap_reconstruction(recon_dir, object_keypoints)

    for i in range(1):
        sfm.run_mapper(recon_dir, img_dir)
        # Running the triangulator may cause (more) duplicate 3d points!
        sfm.point_triangulator(recon_dir, img_dir) # try to triangulate some more points


    # transform reconstruction to apriltag board
    atag_board_keypoints = sfm.read_apriltag_keypoints_json(CALIBRATION_PATH/'targets'/'atag_board_keypoints.json')
    recon = pycolmap.Reconstruction(recon_dir/'0')
    scale, Rt = sfm.compute_transform_to_custom_keypoints(recon_dir, recon, atag_board_keypoints)
    Rt = Rt[:3,:]
    Rt[:3,:3] *= scale
    recon.transform(Rt)
    board_point_ids = sfm.set_keypoint_positions_in_reconstruction(recon_dir, recon, atag_board_keypoints)
    recon.write(str(recon_dir/'0'))

    sfm.run_bundle_adjustment(recon_dir/'0', recon_dir/'0', False, False, False, constant_point_ids=board_point_ids)

    recon = pycolmap.Reconstruction(recon_dir/'0')
    sfm.merge_duplicate_keypoint_point3Ds(db_path, recon)
    recon.write(str(recon_dir/'0'))

    board_point_ids = sfm.set_keypoint_positions_in_reconstruction(recon_dir, recon, atag_board_keypoints)
    sfm.run_bundle_adjustment(recon_dir/'0', recon_dir/'0', False, False, False, constant_point_ids=board_point_ids)

    if args.visualize:
        custom_keypoints = sfm.get_custom_keypoints_from_db(db_path)
        sfm.visualize_reconstruction(recon, custom_keypoints, sg_prefix=f'{recon_dir.parent.parent.name}.{recon_dir.parent.name}')


    # check which of the images are missing
    registered_images = set([x.name for _, x in recon.images.items()])
    print('registered images', sorted(list(registered_images)))
    source_data_dir = SOURCE_DATA_PATH/object_name/env_name

    for p in sorted(list((source_data_dir/'images').glob('*.CR3'))):
        im_name = p.with_suffix('.jpg').name
        if im_name not in registered_images:
            print(f'!!! Did not find image "{im_name}" in registered_images !!!')

    # check if the reconstruction contains the test images
    for im_dir in source_data_dir.iterdir():
        if im_dir.is_dir() and 'test' in im_dir.name and not 'env' in im_dir.name:
            print(im_dir)
            paths = sorted(list(im_dir.glob('*.CR3')))
            test_im_path = paths[len(paths)//2:][0]
            test_im_name = test_im_path.with_suffix('.jpg').name
            if test_im_name in registered_images:
                print(f'found {test_im_name} in registered_images')
            else:
                print(f'!!! Did not find test image "{test_im_name}" in registered_images !!!')

    # check if the reconstruction contains all images
    print(f'registered {len(recon.images)} / {len(list(img_dir.glob("*.jpg")))} images')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script for reconstructing scenes with COLMAP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("scene_dir", type=Path, help="Paths to the intermediate_data directory of the env that contains the 'images', 'env' and 'test*' subdirs.")
    parser.add_argument("--overwrite", action='store_true', help="If True overwrite existing directories")
    parser.add_argument("--visualize", action='store_true', help="If True visualize with o3d rpc interface")

    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    print(args)

    reconstruct_scene(args)
