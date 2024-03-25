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
        description="Script that converts the Synthetic4Relight dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("invrender_dir", type=Path, help="Path to the InvRender repository.")
    parser.add_argument("synth4relight_dir", type=Path, help="Path to the Synthetic4Relight directory.")
    parser.add_argument("output_dir", type=Path, help="Path to the output directory.")
    parser.add_argument("--overwrite", action='store_true', help="If True overwrite existing directories")

    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    print(args)

    invrender_dir = args.invrender_dir
    synth4relight_dir = args.synth4relight_dir
    output_dir = args.output_dir
    if output_dir.exists() and not args.overwrite:
        print(f'{str(output_dir)} already exists!')
        sys.exit(1)


    output_dir.mkdir(exist_ok=True)
    envmaps = {x.stem: x for x in invrender_dir.glob('code/envmaps/envmap*exr')}

    for scene in synth4relight_dir.iterdir():
        if not scene.is_dir():
            continue

        print(str(scene))

        ###### input images 
        with open(scene/'transforms_train.json','r') as f:
            transforms = json.load(f)
        
        # camera parameters
        img = read_hdri(str(scene/transforms['frames'][0]['file_path'])+'_rgb.exr')
        width = img.shape[1]
        height = img.shape[0]
        camera_angle_x = transforms['camera_angle_x']
        fx = .5 * width / np.tan(.5 * camera_angle_x)
        K = np.eye(3)
        K[0,0] = K[1,1] = fx
        K[0,2] = width/2
        K[1,2] = height/2

        out = output_dir/scene.name/'inputs'
        out.mkdir(exist_ok=True, parents=True)
        exposure = 0.0
        with open(out/'exposure.txt', 'w') as f:
            f.write('0.0')
        np.savetxt(out/'object_bounding_box.txt', [-1,1,-1,1,-1,1])

        for i, frame in tqdm(enumerate(transforms['frames'])):
            c2w = np.array(frame['transform_matrix'])
            c2w[:,1:3] *= -1. # COLMAP => OpenGL
            w2c = np.linalg.inv(c2w)
            R = w2c[:3,:3]
            t = w2c[:3,3]
            rgb = read_hdri(scene/(frame['file_path']+'_rgb.exr'))
            mask = cv2.imread(str(scene/(frame['file_path']+'_mask.png')))
            im8bit = simple_tonemap(rgb,exposure=exposure)
            # visualize_camera(R,t, 's4rli/cameras', time=i)
            
            cv2.imwrite(str(out/f'image_{i:04d}.png'), im8bit[...,[2,1,0]])
            cv2.imwrite(str(out/f'mask_{i:04d}.png'), mask)
            cv2.imwrite(str(out/f'mask_binary_{i:04d}.png'), mask)
                
            # assemble camera parameters as a 8x3 matrix with [K,R,t,shape]
            camera_params = np.zeros((8,3))
            im_K = K.copy()
            # scale to new image size
            im_K[0,0] *= im8bit.shape[1]/width
            im_K[0,2] *= im8bit.shape[1]/width
            im_K[1,1] *= im8bit.shape[0]/height
            im_K[1,2] *= im8bit.shape[0]/height
            camera_params[:3,:] = im_K
            camera_params[3:6,:] = R
            camera_params[6,:] = t
            camera_params[7,:] = im8bit.shape[1], im8bit.shape[0], im8bit.shape[2]
            np.savetxt(out/f'camera_{i:04d}.txt', camera_params)

        ###### test images
        with open(scene/'transforms_test.json','r') as f:
            transforms = json.load(f)
        out = output_dir/scene.name

        i = 0
        for env in ('envmap3', 'envmap6', 'envmap12'):
            exr = read_hdri(envmaps[env])
            hdr_env_path = out/f'gt_env_{i:04d}.hdr'
            write_hdri(exr[...,:3], hdr_env_path)
            
            rotated_hdr = rotate_envmap_simple(cv2.resize(exr[...,:3], (1024,512)), np.eye(4))
            hdr_rotated_env_path = out/f'gt_env_512_rotated_{i:04d}.hdr'
            write_hdri(rotated_hdr, hdr_rotated_env_path)
            
            for frame in tqdm(transforms['frames']):
                
                if not (out/f'gt_env_{i:04d}.hdr').exists():
                    (out/f'gt_env_{i:04d}.hdr').unlink(missing_ok=True)
                    (out/f'gt_env_512_rotated_{i:04d}.hdr').unlink(missing_ok=True)
                    (out/f'gt_env_{i:04d}.hdr').symlink_to(hdr_env_path.name)
                    (out/f'gt_env_512_rotated_{i:04d}.hdr').symlink_to(hdr_rotated_env_path.name)
                
                c2w = np.array(frame['transform_matrix'])
                c2w[:,1:3] *= -1. # COLMAP => OpenGL
                w2c = np.linalg.inv(c2w)
                R = w2c[:3,:3]
                t = w2c[:3,3]
                if env == 'envmap3':
                    im_path = scene/'test'/f'{Path(frame["file_path"]).name}_rgba.png'
                else:
                    im_path = scene/'test_rli'/f'{env}_{Path(frame["file_path"]).name}.png'
                bgra = cv2.imread(str(im_path), cv2.IMREAD_UNCHANGED)
                
                # visualize_camera(R,t, 's4rli/cameras', time=i)
                cv2.imwrite(str(out/f'gt_image_{i:04d}.png'), bgra[...,:3])
                cv2.imwrite(str(out/f'gt_mask_{i:04d}.png'), bgra[...,[3,3,3]])
                with open(out/f'gt_exposure_{i:04d}.txt', 'w') as f:
                    f.write('0.0')

                # assemble camera parameters as a 8x3 matrix with [K,R,t,shape]
                camera_params = np.zeros((8,3))
                im_K = K.copy()
                # scale to new image size
                im_K[0,0] *= im8bit.shape[1]/width
                im_K[0,2] *= im8bit.shape[1]/width
                im_K[1,1] *= im8bit.shape[0]/height
                im_K[1,2] *= im8bit.shape[0]/height
                camera_params[:3,:] = im_K
                camera_params[3:6,:] = R
                camera_params[6,:] = t
                camera_params[7,:] = im8bit.shape[1], im8bit.shape[0], im8bit.shape[2]
                np.savetxt(out/f'gt_camera_{i:04d}.txt', camera_params)
                
                world2env = np.eye(4)
                np.savetxt(out/f'gt_world_to_env_{i:04d}.txt', world2env)
                
                i += 1
