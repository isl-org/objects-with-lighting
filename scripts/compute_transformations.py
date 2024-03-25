# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path, PureWindowsPath
sys.path.append(str(Path(__file__).absolute().parent.parent))
import argparse
import numpy as np
import open3d as o3d
from utils.constants import *
from itertools import combinations
import pycolmap
import cv2
import utils
from utils import hdri
from utils import reconstruction as sfm
from utils import equirectangular
from utils import features
from utils.color_calibration import apply_color_transform
from utils.constants import *


def compute_envmap_transforms(scene_dir: Path, visualize: bool=False):
    """Computes the transformation for the environment map.
    Args:
        scene_dir: This is the dir that contains the 'env', 'test1_env', ... directories.
    """
    theta_ct = np.loadtxt(CALIBRATION_PATH/'color'/'theta_to_theta_scaled.txt')

    for d in scene_dir.iterdir():
        if 'env' in d.name and d.is_dir():
            env_hdr_path = scene_dir/f'{d.name}.hdr'
            if env_hdr_path.exists():
                pass # nothing to do if the file already exists
            else:
                print(d)
                env_hdr = hdri.compute_hdri_from_tiffs(list(d.glob('*.tiff')))
                env_hdr = apply_color_transform(env_hdr, theta_ct)
                hdri.write_hdri(hdri.simple_downsample(env_hdr), env_hdr_path)

        elif 'env' in d.name and d.name.endswith('.hdr'):
            env_hdr_path = d
            env_hdr = hdri.read_hdri(env_hdr_path)
            env_ldr_path = scene_dir/f'{d.stem}.jpg'
            if env_ldr_path.exists():
                env_ldr = cv2.imread(str(env_ldr_path))[...,[2,1,0]]
            else:
                env_ldr = hdri.compute_ldr_from_hdri_opencv(env_hdr)
                cv2.imwrite(str(env_ldr_path), env_ldr[...,[2,1,0]])

            w, h = 1200, 900
            images = equirectangular.create_pinhole_images_for_alignment(env_ldr.astype(np.float32), w, h) 
            cameras = []
            rig_qvecs = []
            rig_tvecs = []
            points2d = []
            points3d = []

            atag_board_keypoints = sfm.read_apriltag_keypoints_json(CALIBRATION_PATH/'targets'/'atag_board_keypoints.json')

            detector = features.ApriltagDetector(valid_ids=features.DEFAULT_VALID_TAG_IDS)

            for i, im in enumerate(images):
                if visualize:
                    sfm.visualize_camera(im.T[:3,:3], im.T[:3,3], path='transforms/env_pinhole_cam', time=i)
                dbg_dir = scene_dir/'dbg'
                dbg_dir.mkdir(exist_ok=True)
                im.image = np.clip(im.image, 0, 255).astype(np.uint8)
                ans, dbg_im = detector.detect(features.get_grayscale_array_for_detection(im.image), min_size=50, min_margin=8, debug=True)
                cv2.imwrite(str(dbg_dir/f'{d.name}_{i}.jpg'), np.asarray(dbg_im)[...,[2,1,0]])
                if not ans:
                    continue
                keypoints = []
                for x in ans:
                    keypoints.extend(features.convert_tag_to_customkeypoints(x))

                p2d = []
                p3d = []
                for kp in keypoints:
                    p2d.append(kp.point)
                    p3d.append(atag_board_keypoints[kp.id])
                points2d.append(np.array(p2d))
                points3d.append(np.array(p3d))
                
                cam = pycolmap.Camera(model='PINHOLE', width=w, height=h, params=[im.K[0,0], im.K[1,1], im.K[0,2], im.K[1,2]])
                cameras.append(cam)
                rig_qvecs.append(pycolmap.rotmat_to_qvec(im.T[:3,:3]))
                rig_tvecs.append(im.T[:3,3])

            ans = pycolmap.rig_absolute_pose_estimation(points2d, points3d, cameras, rig_qvecs, rig_tvecs)
            assert ans['success'], "Failed to estimate envmap pose"
            R = pycolmap.qvec_to_rotmat(ans['qvec'])
            T = np.eye(4)
            T[:3,:3] = R
            T[:3,3] = ans['tvec']
            np.savetxt(scene_dir/f'{d.stem}_world_to_env.txt', T)

            if visualize:
                equirectangular.visualize_envmap_as_sphere(env_ldr, T)
                sfm.visualize_camera(T[:3,:3], T[:3,3], 'transforms/envmap0')
                for i, im in enumerate(images):
                    w2c = im.T @ T
                    sfm.visualize_camera(w2c[:3,:3], w2c[:3,3], 'transforms/envmap', time=i)


def compute_transformations(args):
    object_dir = args.object_dir
    env_dirs = sorted([x for x in object_dir.iterdir() if len(x.name) == 1 or x.name in ('train', 'valid', 'test')])

    for env_dir in env_dirs:
        print(env_dir)
        compute_envmap_transforms(env_dir, visualize=args.visualize)

    for env_a_path, env_b_path in combinations(env_dirs, 2):
        env_a_name = env_a_path.name
        env_b_name = env_b_path.name
        print(env_a_path, env_b_path)
        
        recon_a = pycolmap.Reconstruction(env_a_path/'recon'/'0')
        recon_b = pycolmap.Reconstruction(env_b_path/'recon'/'0')

        kpname_xyz_b = sfm.get_3d_keypoints(env_b_path/'recon'/'database.db', recon_b)
        kpname_xyz_b = {k: v for k,v in kpname_xyz_b.items() if k.startswith('obj_')}
        c, Rt = sfm.compute_transform_to_custom_keypoints(env_a_path/'recon', recon_a, kpname_xyz_b)
        
        out_path = object_dir/f'transform_object_{env_a_name}_to_{env_b_name}.txt'
        T = Rt
        T[:,3] /= c # include the scale here to not lose information and be able to check if the scale is close to 1
        np.savetxt(out_path, T)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute transformations for the environment maps and objects between different envirionments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("object_dir", type=Path, help="Paths to the intermediate_data directory that contains images of an object in multiple environments. Environment dirs are named as 'A', 'B', ...")
    parser.add_argument("--overwrite", action='store_true', help="If True overwrite existing directories")
    parser.add_argument("--visualize", action='store_true', help="If True visualize with o3d rpc interface")

    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    print(args)

    compute_transformations(args)
