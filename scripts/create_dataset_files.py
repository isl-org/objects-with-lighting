# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
import argparse
import re
from itertools import product
import numpy as np
from scipy.ndimage.morphology import binary_erosion
import pycolmap
import cv2
from utils import hdri
from utils import equirectangular
from utils.color_calibration import load_color_transforms, apply_color_transform
from utils.constants import *
from utils.filesystem import copyfile
from utils.camera import read_eos_intrinsics
from utils import reconstruction as sfm
import open3d as o3d


def downsample(arr, div: int=4):
    """Downsample function for creating the final images."""
    new_size = (arr.shape[1]//div, arr.shape[0]//div)
    return cv2.resize(arr, new_size, interpolation=cv2.INTER_AREA)

def scale_K(K, scale:float):
    ans = K * scale
    ans[-1,:] = (0,0,1)
    return ans

def compute_normalized_intrinsics(K, width: int, height: int, div: int=128):
    """Computes new normalized camera intrinsics and shape
    
    The new intrinsics will have the principal point at the image center and a 
    single focal length value.
    In addition the new image size will be divisible by div.
    
    Args:
        K: The 3x3 intrinsic matrix
        width: The integer image width
        height: The integer image height
        div: The image size will be at least divisable by this value
        
    Returns:
        Dict with the new camera intrinsic matrix and the new image dimensions.
    """
    focal = min(K[0,0], K[1,1])

    sx = focal/K[0,0]
    sy = focal/K[1,1]
    
    new_width = int(sx * width)
    new_height = int(sy * height)
    new_cx = sx * K[0,2]
    new_cy = sy * K[1,2]
    
    # compute the largest width and height after moving the principal point to the center
    new_width = int(2*min(new_cx, new_width-new_cx))
    new_height = int(2*min(new_cy, new_height-new_cy))
    
    # make sure we can divide the image size by divisor
    new_width = int(new_width/div)*div
    new_height = int(new_height/div)*div

    K2 = np.eye(3)
    K2[0,0] = K2[1,1] = focal
    K2[0,2] = new_width/2
    K2[1,2] = new_height/2
    return {'K': K2, 'width': new_width, 'height': new_height}


def warp_to_new_intrinsics(im, old_K, K, width, height):
    """Creates a new image by warping to the new intrinsics and image size
    
    Args:
        im: Image numpy array
        old_K: The old 3x3 intrinsic matrix that corresponds to 'im'
        K: The new 3x3 intrinsic matrix
        width: The integer image width of the new image
        height: The integer image height of the new image
        
    Returns:
        Array with the warped image.
    """
    M = K@np.linalg.inv(old_K)
    M = np.ascontiguousarray(M[:2,:])
    return cv2.warpAffine(im, M, (width, height))
    

def create_masks(mesh_path: Path, images_dir: Path, blur: bool=False):
    """Writes approximate masks as png files using the mesh.
    Args:
        mesh_path: Path to the ply mesh.
        images_dir: This is the path to the directory with the dataset images. e.g., 'obj/env/inputs'.
        blur: If True blur and dilate the masks to make clear that these masks are not GT.
    """
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)

    cameras = sorted(list(images_dir.glob('camera_*.txt')))
    images = sorted(list(images_dir.glob('image_*.png')))
    assert len(cameras) == len(images)

    for i, (camera_path, image_path) in enumerate(zip(cameras, images)):
        params = np.loadtxt(camera_path)
        K, R, t, (width, height, channels) = params[:3], params[3:6], params[6], params[7].astype(int)
        T = np.eye(4)
        T[:3,:3] = R
        T[:3,3] = t

        rays = scene.create_rays_pinhole(K, T, width, height)
        ans = scene.cast_rays(rays)

        mask = ans['t_hit'].numpy() 
        mask = np.isfinite(mask)

        if blur:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(63,63))
            mask = cv2.dilate(mask.astype(np.uint8), kernel,)
            mask[mask==1] = 255
            mask = cv2.GaussianBlur(mask, None, 16.0)
        else:
            mask = mask.astype(np.uint8)
            mask[mask==1] = 255

        cv2.imwrite(str(images_dir/f'mask_{i:04d}.png'), mask)


def create_debug_images(env_dir: Path):
    """Creates debug images for a dataset env dir.
    Args:
        env_dir: Path to an env dir of the dataset.
    """
    print('creating debug images for', str(env_dir))
    env_dir = Path(env_dir).resolve()
    object_name = env_dir.parent.name
    env_name = env_dir.name

    # corresponding source dir and intermediate dir
    source_dir = SOURCE_DATA_PATH/object_name/env_name
    interm_dir = INTERMEDIATE_DATA_PATH/object_name/env_name
    dbg_dir = interm_dir/'_dbg'
    dbg_dir.mkdir(exist_ok=True)
    
    mask_mesh_path = source_dir/'mask_mesh.ply'
    if mask_mesh_path.exists():

        mesh = o3d.io.read_triangle_mesh(str(mask_mesh_path))
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh)

        cameras = sorted(list(env_dir.glob('gt_camera_*.txt')))
        images = sorted(list(env_dir.glob('gt_image_*.png')))
        assert len(cameras) == len(images)
        for i, (camera_path, image_path) in enumerate(zip(cameras, images)):
            params = np.loadtxt(camera_path)
            K, R, t, (width, height, channels) = params[:3], params[3:6], params[6], params[7].astype(int)
            T = np.eye(4)
            T[:3,:3] = R
            T[:3,3] = t

            rays = scene.create_rays_pinhole(K, T, width, height)
            ans = scene.cast_rays(rays)

            mask = ans['t_hit'].numpy() 
            mask = np.isfinite(mask).astype(np.float32)
            mask[mask==1] = 1.2
            mask[mask==0] = 0.5
            im = cv2.imread(str(image_path)).astype(np.float32)
            im = np.clip(im * mask[...,None], 0, 255).astype(np.uint8)
            out_path = dbg_dir/f'gt_image_mesh_mask_{i:04d}.jpg'
            cv2.imwrite(str(out_path), im)
        


def create_files_for_env_dir(env_dir: Path, overwrite: bool):
    """Creates all files for the object for a specific environment.
    Args:
        env_dir: Path to the specific environment dir with the intermediate files.
    """
    env_dir = Path(env_dir).resolve()
    object_name = env_dir.parent.name
    env_name = env_dir.name

    output_dir = ROOT_PATH/'dataset'/object_name/env_name
    output_dir.mkdir(exist_ok=overwrite, parents=True)
    
    # corresponding source dir
    source_dir = SOURCE_DATA_PATH/object_name/env_name

    exposure = np.loadtxt(source_dir/'exposure.txt').item()

    color_transforms = load_color_transforms()

    K = np.loadtxt(env_dir/'K.txt')
    dist_coeffs = np.loadtxt(env_dir/'dist_coeffs.txt')
    assert np.all(dist_coeffs==0), f'Images in {env_dir} need to be undistorted.'
    width, height = np.loadtxt(env_dir/'width_height.txt')
    normalized_intrinsics = compute_normalized_intrinsics(K, width, height)
    
    recon = pycolmap.Reconstruction(env_dir/'recon'/'0')
    images = {Path(x.name).stem: x for x in recon.images.values()}
    
    # images dir
    image_paths = sorted([x for x in (source_dir/'images').glob('*.CR3') if x.stem in images])
    output_images_dir = output_dir/'inputs'
    output_images_dir.mkdir(exist_ok=True)
    for i, im_path in enumerate(image_paths):
        raw = hdri.read_raw_and_meta(im_path)
        # not an hdri, just use the same function to get the linear image.
        linear_im = hdri.compute_hdri_from_raws([raw])
        ct = color_transforms[f'eos90d_iso_{raw.iso}_to_theta']
        linear_im = apply_color_transform(linear_im, ct)
        linear_im_K = scale_K(K, 0.5) # compute_hdri_from_raws downsamples the image
        linear_im = warp_to_new_intrinsics(linear_im, linear_im_K, **normalized_intrinsics)
        linear_im = downsample(linear_im)
        im8bit = hdri.simple_tonemap(linear_im, exposure)
        cv2.imwrite(str(output_images_dir/f'image_{i:04d}.png'), im8bit[...,[2,1,0]])
        
        # assemble camera parameters as a 8x3 matrix with [K,R,t,shape]
        camera_params = np.zeros((8,3))
        im_K = normalized_intrinsics['K'].copy()
        # scale to new image size
        im_K[0,0] *= im8bit.shape[1]/normalized_intrinsics['width']
        im_K[0,2] *= im8bit.shape[1]/normalized_intrinsics['width']
        im_K[1,1] *= im8bit.shape[0]/normalized_intrinsics['height']
        im_K[1,2] *= im8bit.shape[0]/normalized_intrinsics['height']
        camera_params[:3,:] = im_K
        R = pycolmap.qvec_to_rotmat(images[im_path.stem].qvec)
        t = images[im_path.stem].tvec
        camera_params[3:6,:] = R
        camera_params[6,:] = t
        camera_params[7,:] = im8bit.shape[1], im8bit.shape[0], im8bit.shape[2]
        np.savetxt(output_images_dir/f'camera_{i:04d}.txt', camera_params)

    copyfile(source_dir/'exposure.txt', output_images_dir/'exposure.txt')
    

    # environment map
    copyfile(env_dir/'env.hdr', output_dir/'env.hdr')
    copyfile(env_dir/'env_world_to_env.txt', output_dir/'world_to_env.txt')


    # create downsampled and rotated version of the environment map
    hdr = hdri.read_hdri(env_dir/'env.hdr')
    rotated_hdr = equirectangular.rotate_envmap_simple(hdr, np.loadtxt(env_dir/'env_world_to_env.txt'))
    downsampled_rotated_hdr = cv2.resize(rotated_hdr, [1024,512], interpolation=cv2.INTER_AREA)
    hdri.write_hdri(downsampled_rotated_hdr, output_dir/f'env_{downsampled_rotated_hdr.shape[0]}_rotated.hdr')


    # copy object bounding box
    bounding_box_path = source_dir/'object_bounding_box.txt'
    if bounding_box_path.exists():
        copyfile(bounding_box_path, output_dir/'inputs'/bounding_box_path.name, overwrite=True)
    else:
        print(f'{str(bounding_box_path)} does not exist. Object does not have a bounding box!!!')

    mask_mesh_path = source_dir/'mask_mesh.ply'
    if mask_mesh_path.exists():
        create_masks(mask_mesh_path, output_images_dir)
    else:
        print(f'{str(mask_mesh_path)} does not exist. Object does not have a mesh for creating input mask images!!!')
        




def create_test_files_from_other_env(env_dir: Path, other_env_dir: Path):
    """Creates all files for the object for a specific environment.
    Args:
        env_dir: Path to the specific environment dir with the intermediate files.
    """
    env_dir = Path(env_dir).resolve()
    object_name = env_dir.parent.name
    env_name = env_dir.name

    other_env_dir = Path(other_env_dir).resolve()
    other_env_name = other_env_dir.name

    other_to_env_transform = None
    if env_name == other_env_name:
        other_to_env_transform = np.eye(4)
    else:
        object_transform_path = env_dir.parent/f'transform_object_{env_name}_to_{other_env_name}.txt'
        if object_transform_path.exists():
            other_to_env_transform = np.loadtxt(object_transform_path)
            other_to_env_transform[3,3] = 1 # ignore scale

        object_transform_path = env_dir.parent/f'transform_object_{other_env_name}_to_{env_name}.txt'
        if object_transform_path.exists():
            other_to_env_transform = np.loadtxt(object_transform_path)
            other_to_env_transform[3,3] = 1 # ignore scale
            other_to_env_transform = np.linalg.inv(other_to_env_transform)

    assert other_to_env_transform is not None

    recon_env = pycolmap.Reconstruction(env_dir/'recon'/'0')
    kpid_xyz = sfm.get_3d_keypoints(env_dir/'recon'/'database.db', recon_env)

    output_dir = ROOT_PATH/'dataset'/object_name/env_name

    source_dir = SOURCE_DATA_PATH/object_name/other_env_name
    exposure = np.loadtxt(source_dir/'exposure.txt').item()

    color_transforms = load_color_transforms()

    K = np.loadtxt(other_env_dir/'K.txt')
    dist_coeffs = np.loadtxt(other_env_dir/'dist_coeffs.txt')
    assert np.all(dist_coeffs==0), f'Images in {env_dir} need to be undistorted.'
    width, height = np.loadtxt(other_env_dir/'width_height.txt')
    normalized_intrinsics = compute_normalized_intrinsics(K, width, height)
    
    recon_other_env = pycolmap.Reconstruction(other_env_dir/'recon'/'0')
    imgname_kpid_xy = sfm.get_2d_keypoints(other_env_dir/'recon'/'database.db')
    images = {Path(x.name).stem: x for x in recon_other_env.images.values()}
    cam = recon_other_env.cameras[1]

    # test images
    first_test_i = 0
    while (output_dir/f'gt_image_{first_test_i:04d}.png').exists():
        first_test_i += 1

    test_i = first_test_i
    for test_dir in sorted(list(source_dir.iterdir())):
        if test_dir.name.startswith('test') and test_dir.is_dir() and not 'env' in test_dir.name:
            paths = sorted(list(test_dir.glob('*.CR3')))
            raws = [hdri.read_raw_and_meta(x) for x in paths]
            linear_im = hdri.compute_hdri_from_raws(raws)
            ct = color_transforms[f'eos90d_iso_{raws[0].iso}_to_theta']
            linear_im = apply_color_transform(linear_im, ct)
            linear_im_K = scale_K(K, 0.5) # compute_hdri_from_raws downsamples the image
            linear_im = warp_to_new_intrinsics(linear_im, linear_im_K, **normalized_intrinsics)
            linear_im = downsample(linear_im)
            im8bit = hdri.simple_tonemap(linear_im, exposure)
            cv2.imwrite(str(output_dir/f'gt_image_{test_i:04d}.png'), im8bit[...,[2,1,0]])

            # the center image was included in the reconstruction
            img_key = paths[len(paths)//2:][0].stem

            # assemble camera parameters as a 8x3 matrix with [K,R,t,shape]
            camera_params = np.zeros((8,3))
            im_K = normalized_intrinsics['K'].copy()
            # scale to new image size
            im_K[0,0] *= im8bit.shape[1]/normalized_intrinsics['width']
            im_K[0,2] *= im8bit.shape[1]/normalized_intrinsics['width']
            im_K[1,1] *= im8bit.shape[0]/normalized_intrinsics['height']
            im_K[1,2] *= im8bit.shape[0]/normalized_intrinsics['height']
            camera_params[:3,:] = im_K
            
            # estimate test image camera pose with pnp method
            if env_name != other_env_name and False:
                kpid_xy = imgname_kpid_xy[img_key]
                points2d = []
                points3d = []
                for id, xy in kpid_xy.items():
                    if id.startswith('obj_') and id in kpid_xyz:
                        points2d.append(xy)
                        points3d.append(kpid_xyz[id])

                pose = pycolmap.absolute_pose_estimation(points2d, points3d, cam)
                print(test_i, len(points2d), pose)
                assert pose['success']
                R = pycolmap.qvec_to_rotmat(pose['qvec'])
                t = pose['tvec']
                T = np.eye(4)
                T[:3,:3] = R
                T[:3,3] = t
                
            # estimate using the transformation between the two reconstructions
            else:
                R = pycolmap.qvec_to_rotmat(images[img_key].qvec)
                t = images[img_key].tvec
                T = np.eye(4)
                T[:3,:3] = R
                T[:3,3] = t
                T = T @ other_to_env_transform
                # refine the pose
                if env_name != other_env_name:
                    tvec = T[:3,3]
                    qvec = pycolmap.rotmat_to_qvec(T[:3,:3])

                    kpid_xy = imgname_kpid_xy[img_key]
                    points2d = []
                    points3d = []
                    for id, xy in kpid_xy.items():
                        if id.startswith('obj_') and id in kpid_xyz:
                            points2d.append(xy)
                            points3d.append(kpid_xyz[id])
                    pose = pycolmap.pose_refinement(tvec, qvec, points2d, points3d, len(points2d)*[True], cam)
                    assert pose['success']
                    R = pycolmap.qvec_to_rotmat(pose['qvec'])
                    t = pose['tvec']
                    T = np.eye(4)
                    T[:3,:3] = R
                    T[:3,3] = t

            camera_params[3:6,:] = T[:3,:3]
            camera_params[6,:] = T[:3,3]
            camera_params[7,:] = im8bit.shape[1], im8bit.shape[0], im8bit.shape[2]
            np.savetxt(output_dir/f'gt_camera_{test_i:04d}.txt', camera_params)
            copyfile(source_dir/'exposure.txt', output_dir/f'gt_exposure_{test_i:04d}.txt')
            
            # write undistorted mask for evaluation
            mask_path = source_dir/f'{test_dir.name}_mask.png'
            if mask_path.exists():
                intrinsics = read_eos_intrinsics()
                mask = cv2.imread(str(mask_path))
                mask_K = intrinsics.K.copy()
                mask_K[0,0] *= mask.shape[1]/intrinsics.width
                mask_K[0,2] *= mask.shape[1]/intrinsics.width
                mask_K[1,1] *= mask.shape[0]/intrinsics.height
                mask_K[1,2] *= mask.shape[0]/intrinsics.height
                mask = cv2.undistort(mask, mask_K, intrinsics.dist_coeffs)
                mask = warp_to_new_intrinsics(mask, scale_K(K,0.5), **normalized_intrinsics)
                mask = downsample(mask)
                mask[mask<200] = 0
                mask[mask>=200] = 255
                mask[...,1] = binary_erosion(mask[...,1]).astype(np.uint8)*255
                mask[...,0] = mask[...,1]
                mask[...,2] = mask[...,1]
                cv2.imwrite(str(output_dir/f'gt_mask_{test_i:04d}.png'), mask[...,:3])
            else:
                print(f'Mask for {str(test_dir)} does not exist!!!')

            test_i += 1

    # environment maps
    test_i = first_test_i
    for path in sorted(list(other_env_dir.glob('test*_env.hdr'))):
        copyfile(path, output_dir/f'gt_env_{test_i:04d}.hdr')
        world_to_env_transform = np.loadtxt(path.parent/f'{path.stem}_world_to_env.txt')
        world_to_env_transform = world_to_env_transform @ other_to_env_transform
        np.savetxt(output_dir/f'gt_world_to_env_{test_i:04d}.txt', world_to_env_transform)
        
        # create downsampled and rotated version of the environment map
        hdr = hdri.read_hdri(path)
        rotated_hdr = equirectangular.rotate_envmap_simple(hdr, world_to_env_transform)
        downsampled_rotated_hdr = cv2.resize(rotated_hdr, [1024,512], interpolation=cv2.INTER_AREA)
        hdri.write_hdri(downsampled_rotated_hdr, output_dir/f'gt_env_{downsampled_rotated_hdr.shape[0]}_rotated_{test_i:04d}.hdr')

        test_i += 1




def create_dataset_files(args):

    env_dirs = []
    for env_dir in sorted(list(args.intermediate_object_data.iterdir())):
        if len(env_dir.name) > 1 and env_dir.name not in ('train', 'valid', 'test'):
            if not env_dir.is_file():
                print(f'ignoring {str(env_dir)}')
            continue
        env_dirs.append(env_dir)

    for env_dir in env_dirs:
        print(env_dir)
        create_files_for_env_dir(env_dir, overwrite=args.overwrite)

    if args.overwrite:
        for env_dir in env_dirs:
            env_dir = Path(env_dir).resolve()
            object_name = env_dir.parent.name
            env_name = env_dir.name
            output_dir = ROOT_PATH/'dataset'/object_name/env_name
            for x in output_dir.glob('gt_*'):
                x.unlink()
            
    for env_a_path, env_b_path in product(env_dirs, repeat=2):
        print(env_a_path, env_b_path)
        create_test_files_from_other_env(env_a_path, env_b_path)

    for env_dir in env_dirs:
        env_dir = Path(env_dir).resolve()
        object_name = env_dir.parent.name
        env_name = env_dir.name
        output_dir = ROOT_PATH/'dataset'/object_name/env_name
        create_debug_images(output_dir)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script that creates the files for the dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("intermediate_object_data", type=Path, help="Path to the directory with the intermediate files that has the COLMAP reconstructions for the object in multiple environments")
    parser.add_argument("--overwrite", action='store_true', help="If True overwrite existing directories")

    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    print(args)

    create_dataset_files(args)
