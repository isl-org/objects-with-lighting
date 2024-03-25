# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
import re
import numpy as np
import pycolmap
import argparse
from utils import reconstruction as sfm
from utils import equirectangular
from utils import hdri
from utils.constants import INTERMEDIATE_DATA_PATH
import open3d as o3d


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script that visualizes an environment from the dataset with the RPC interface.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("env_dir", type=Path, help="Paths to the environment directory of the final dataset.")

    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    print(args)

    env_dir = args.env_dir.resolve()
    env_name = env_dir.name
    obj_name = env_dir.parent.name

    meshes = env_dir.glob('*.ply')
    for mesh in meshes:
        m = o3d.io.read_triangle_mesh(str(mesh))
        o3d.io.rpc.set_triangle_mesh(m, path=f'{obj_name}.{env_name}/mesh')

    recon_dir = INTERMEDIATE_DATA_PATH/obj_name/env_name/'recon'
    db_path = recon_dir/'database.db'
    if db_path.exists():
        recon = pycolmap.Reconstruction(recon_dir/'0')
        custom_keypoints = sfm.get_custom_keypoints_from_db(db_path)
        sfm.visualize_reconstruction(recon, custom_keypoints, sg_prefix=f'{recon_dir.parent.parent.name}.{recon_dir.parent.name}/colmap')


    # input cameras
    input_cams = sorted(list((env_dir/'inputs').glob('camera_*txt')))
    for cam_i, cam_path in enumerate(input_cams):
        params = np.loadtxt(cam_path)
        R, t = params[3:6], params[6]
        sfm.visualize_camera(R,t, path=f'{recon_dir.parent.parent.name}.{recon_dir.parent.name}/input_cams', time=cam_i)

    # test cameras
    test_cams = sorted(list(env_dir.glob('gt_camera_*txt')))
    for cam_i, cam_path in enumerate(test_cams):
        params = np.loadtxt(cam_path)
        R, t = params[3:6], params[6]
        sfm.visualize_camera(R,t, path=f'{recon_dir.parent.parent.name}.{recon_dir.parent.name}/gt_cams', time=cam_i)



    envmap_paths = sorted(list(env_dir.glob('*.hdr')))

    for envmap_p in envmap_paths:
        transform_path = envmap_p.parent/envmap_p.name.replace('env', 'world_to_env').replace('.hdr', '.txt')
        if transform_path.exists():
            print(str(envmap_p), str(transform_path))
            w2env = np.loadtxt(transform_path)
        else:
            print(str(envmap_p), 'no transform found')
            w2env = np.eye(4)
        hdr = hdri.read_hdri(envmap_p)
        ldr = hdri.compute_ldr_from_hdri_opencv(hdr)
        equirectangular.visualize_envmap_as_sphere(ldr, w2env, prefix=f'{obj_name}.{env_name}/envs/{envmap_p.stem}')

