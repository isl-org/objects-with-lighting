# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path, PureWindowsPath
sys.path.append(str(Path(__file__).absolute().parent.parent))
import argparse
import numpy as np
import json
import cv2
import rawpy
from utils import hdri
from utils.constants import *
from utils.camera import read_eos_intrinsics
from utils.equirectangular import create_equirectangular_from_theta_dfe
from utils.color_calibration import apply_color_transform


def create_intermediate_files_for_env_directory(input_dir: Path, output_dir: Path, args):
    """Creates intermediate files for a directory that contains images for one environment.
    Args:
        input_dir: Path to an environment dir with the source files, e.g., 'source_data/obj/A'
        output_dir: Output directory
        args: Additional arguments from the commandline
    """

    intrinsics = read_eos_intrinsics()
    output_dir.mkdir(exist_ok=True, parents=True)


    # Write intrinsic camera parameters to the output directory.
    np.savetxt(output_dir/'K.txt', intrinsics.K)
    np.savetxt(output_dir/'width_height.txt', intrinsics.width_height)
    if args.undistort_images:
        np.savetxt(output_dir/'dist_coeffs.txt', np.zeros_like(intrinsics.dist_coeffs))
    else:
        np.savetxt(output_dir/'dist_coeffs.txt', intrinsics.dist_coeffs)


    image_subdirs = []
    env_subdirs = []
    for x in input_dir.iterdir():
        if x.is_dir():
            if x.name.startswith('_'):
                continue
            if 'env' in x.name:
                env_subdirs.append(x)
            else:
                image_subdirs.append(x)
    

    tonemap = cv2.createTonemapReinhard()

    # Create intermediate files for the images taken with the eos90d.
    # Combine all images including the test images into one folder for COLMAP.
    images_out_dir = output_dir/'images'
    images_out_dir.mkdir(exist_ok=True)
    if args.write_images:
        for im_dir in image_subdirs:
            paths = sorted(list(im_dir.glob('*.CR3')))
            # select the center of the exposure sequence for the test dirs.
            if 'test' in im_dir.name:
                paths = paths[len(paths)//2:][:1]
            for p in paths:
                with rawpy.imread(str(p)) as raw:
                    im = raw.postprocess(user_flip=0, output_bps=16)
                    if args.tonemap:
                        im = np.nan_to_num(tonemap.process(im.astype(np.float32)/im.max()))
                        im = np.clip(255*im,0,255).astype(np.uint8)
                if args.undistort_images:
                    assert im.shape[0] == intrinsics.height, f'{im.shape[0]} == {intrinsics.height}'
                    assert im.shape[1] == intrinsics.width, f'{im.shape[1]} == {intrinsics.width}'
                    im = cv2.undistort(im, intrinsics['K'], intrinsics['dist_coeffs'])
                out_path = images_out_dir/p.with_suffix('.jpg').name
                print('writing', str(out_path))
                cv2.imwrite(str(out_path), im[...,[2,1,0]])
            

    # Prepare Theta images for stitching.
    if args.stitcher_images:
        win_tiff_paths = []
        for env_dir in env_subdirs:
            out_dir = output_dir/env_dir.name
            out_dir.mkdir(exist_ok=True)
            dng_paths = sorted(list(env_dir.glob('*.DNG')))
            tiff_paths = hdri.create_data_for_theta_stitcher(dng_paths, out_dir)
            # create commandline for the stitcher
            cmd = f'wine "{args.stitcher_bin}" '
            for p in tiff_paths:
                # assume wine maps the root to Z: 
                win_p = PureWindowsPath('Z:\\')/PureWindowsPath(p.resolve())
                win_tiff_paths.append(win_p)
                cmd += f' "{str(win_p)}"'
            with open(out_dir/'stitch_cmd.sh', 'w') as f:
                f.write(cmd)   
        cmd = f'wine "{args.stitcher_bin}" '
        for p in win_tiff_paths:
            cmd += f' "{str(p)}"'
        with open(output_dir/'stitch_cmd.sh', 'w') as f:
            f.write(cmd)   
    else:
        # use our stitching
        theta_ct = np.loadtxt(CALIBRATION_PATH/'color'/'theta_to_theta_scaled.txt')
        for env_dir in env_subdirs:
            out_path = output_dir/f'{env_dir.name}.hdr'
            dng_paths = sorted(list(env_dir.glob('*.DNG')))
            raws = [hdri.read_raw_and_meta(x) for x in dng_paths]
            dfe_hdr = hdri.compute_hdri_from_raws(raws)
            dfe_hdr = apply_color_transform(dfe_hdr, theta_ct)
            env_hdr = create_equirectangular_from_theta_dfe(dfe_hdr)
            hdri.write_hdri(env_hdr, out_path)
    return


def create_intermediate_files_for_object_directory(args):
    
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir/input_dir.name
    output_dir.mkdir(exist_ok=True, parents=True)

    # undistort annotated keypoints
    intrinsics = read_eos_intrinsics()

    # Write intrinsic camera parameters to the output directory.
    np.savetxt(output_dir/'K.txt', intrinsics.K)
    np.savetxt(output_dir/'width_height.txt', intrinsics.width_height)
    if args.undistort_images:
        np.savetxt(output_dir/'dist_coeffs.txt', np.zeros_like(intrinsics.dist_coeffs))
    else:
        np.savetxt(output_dir/'dist_coeffs.txt', intrinsics.dist_coeffs)

    object_keypoints_path = input_dir/'object_keypoints.json'
    with open(object_keypoints_path, 'r') as f:
        obj_keypoints = json.load(f)

    if args.undistort_images:
        for kp_name, im_point in obj_keypoints.items():
            for im, point in im_point.items():
                p = np.array([[point['x'], point['y']]])
                undist_point = cv2.undistortImagePoints(p, intrinsics.K, intrinsics.dist_coeffs)[:,0,:]
                point['x'] = undist_point[0,0]
                point['y'] = undist_point[0,1]
        
    with open(output_dir/object_keypoints_path.name, 'w') as f:
        json.dump(obj_keypoints, f, indent=4)


    env_dirs = []
    for env_dir in sorted(list(args.input_dir.iterdir())):
        if len(env_dir.name) > 1 and env_dir.name not in ('train', 'valid', 'test'):
            if not env_dir.is_file():
                print(f'ignoring {str(env_dir)}')
            continue
        env_dirs.append(env_dir)

    for env_dir in env_dirs:
        env_output_dir = output_dir/env_dir.name
        if env_output_dir.exists() and not args.overwrite:
            print(f'skipping {str(env_dir)} because {str(env_output_dir)} already exists')
            continue

        print(f'processing {str(env_dir)} -> {str(env_output_dir)}')
        create_intermediate_files_for_env_directory(env_dir, env_output_dir, args)



    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script that creates the intermediate files for COLMAP and envmap stitching.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("input_dir", type=Path, help="Paths to an object dir in the source_data directory that contains images of an object in multiple environments. Environment dirs are named as 'train', 'test', 'valid")
    parser.add_argument("output_dir", type=Path, help="Output directory. This is the path to intermediate_data")
    parser.add_argument("--overwrite", action='store_true', help="If True overwrite existing directories")
    parser.add_argument("--no_undistort_images", dest="undistort_images", action='store_false', help="Do not create undistort images.")
    parser.add_argument("--stitcher_images", dest="stitcher_images", action='store_true', help="Create images for the stitcher software.")
    parser.add_argument("--no_tonemap", dest="tonemap", action='store_false', help="Do not create tonemapped images for COLMAP.")
    parser.add_argument("--no_images", dest="write_images", action='store_false', help="Do not write/update images for COLMAP.")
    parser.add_argument("--stitcher_bin", type=Path, default=Path.home()/'.wine'/'drive_c'/'Program Files'/'RICOH THETA Stitcher'/'RICOH THETA Stitcher.exe', help="The path to the stitcher software" )

    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    print(args)

    create_intermediate_files_for_object_directory(args)
