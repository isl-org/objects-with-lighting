# SPDX-License-Identifier: Apache-2.0
import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation
import addict
import cv2
from .camera import create_pinhole_camera_parameters, fisheye_project
from .constants import CALIBRATION_PATH

def directions_to_uvs(dirs):
    """Converts a normalized direction vector to uv coordinates for an equirectngular image.

    Note that the uv coordinates are in the range [0,1] and use D3D convention.
    The first pixel in memory is at (0,0) and the v-axis is pointing down::

        (0,0) -- u 
        |
        v

    Args:
        dirs (np.ndarray): A Nx3 array of normalized direction vectors.

    Returns:
        Returns two arrays u, v with the same first dimensions as dirs.
    """
    assert dirs.ndim > 1
    assert dirs.shape[-1] == 3
    u = -np.arctan2(dirs[...,1], dirs[...,0]) / (2*np.pi) + 0.5
    v = np.arcsin(np.clip(-dirs[...,2],-1,1)) / np.pi + 0.5
    return u,v


def uvs_to_directions(u,v):
    """Converts uv coordinates to a normalized direction vector.
    This is the inverse of directions_to_uvs(). See directions_to_uvs() for more details.

    Args:
        u (np.ndarray): u coordinates.
        v (np.ndarray): v coordinates.

    Returns:
        Returns an array of normalized direction vectors with the same first dimensions as u and v.
    """
    assert u.ndim >= 1 and v.ndim >= 1
    assert u.shape == v.shape
    theta = -(u-0.5)*2*np.pi
    phi = (v-0.5)*np.pi
    c = np.cos(phi)
    dirs = np.stack([c*np.cos(theta), c*np.sin(theta), -np.sin(phi)], axis=-1)
    return dirs


def get_dirvec_image(width=128, height=64):
    """Returns an image with dir vectors of shape [h,w,3]"""
    uvs = np.stack(np.meshgrid(np.linspace(1/(width+1), 1-1/(width+1), num=width),
                                np.linspace(1/(height+1), 1-1/(height+1), num=height)),
                    axis=-1)
    dirs = uvs_to_directions(uvs[..., 0], uvs[..., 1]).astype(np.float32)
    return dirs


def create_pinhole_image_from_equirectangular(K, R, width, height, equirectangular_image):
    """Generates a pinhole camera image from a equirectangular image.
    
    Args:
        K (np.ndarray): 3x3 intrinsic matrix.
        R (np.ndarray): 3x3 rotation matrix.
        width (int): Width of the pinhole image.
        height (int): Height of the pinhole image.
        equirectangular_image (np.ndarray): Equirectangular image.

    Returns:
        Returns a pinhole camera image with shape (height, width, channels) where channels is the
        same number of channels as equirectangular_image.
    """
    assert K.shape == (3,3)
    assert R.shape == (3,3)
    assert width > 0
    assert height > 0
    assert equirectangular_image.ndim == 3
    assert equirectangular_image.dtype in (np.float32, np.float64)
    inv_K = np.linalg.inv(K)
    RT_invK = R.T @ inv_K
    dirs = np.meshgrid(np.arange(width)+0.5, np.arange(height)+0.5)
    dirs = np.stack(dirs+[np.ones_like(dirs[0])], axis=-1)
    dirs = (RT_invK @ dirs.reshape(-1,3).T).reshape(3,height,width).transpose(1,2,0)

    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)

    u, v = directions_to_uvs(dirs)
    
    h, w = equirectangular_image.shape[:2]
    yx = np.stack([v*h,u*w], axis=0).reshape(2,-1) - 0.5
    result = []
    for i in range(equirectangular_image.shape[-1]):
        result.append(ndimage.map_coordinates(equirectangular_image[...,i], yx, order=1, mode='nearest'))
    
    result = np.stack(result, axis=-1).reshape(height, width, -1)
    
    return result


def create_pinhole_images_for_alignment(equirectangular_image, width, height, target_z=(-0.6,), fov_deg=90):
    """Computes a set of images for aligning the equirectangular image in the scene.
    
    Args:
        equirectangular_image (np.ndarray): Equirectangular image.
        width (int): Width of the pinhole images.
        height (int): Height of the pinhole images.
        target_z (list): List of z-values the pinhole camera is looking at. For each z value 8 images will be generated. The default value makes the camera look slightly down to see markers on the ground.
        fov_deg (float): Field of view in degrees.
    
    Returns:
        A list of dictionaries with the generated images and the corresponding camera parameters.
    """
    assert equirectangular_image.ndim == 3
    assert equirectangular_image.dtype in (np.float32, np.float64)
    assert width > 0
    assert height > 0

    result = []
    for z in target_z:
        for deg in np.linspace(0, 360, endpoint=False, num=8):
            rad = np.deg2rad(deg)
            target_vector = [np.cos(rad), np.sin(rad), z]
            K, T = create_pinhole_camera_parameters(90, target_vector, [0,0,0], [0,0,-1], width, height)
            im = create_pinhole_image_from_equirectangular(K, T[:3,:3], width, height, equirectangular_image)
            result.append(addict.Dict(image=im, K=K, T=T))

    return result


def visualize_envmap_as_sphere(equirectangular_image: np.ndarray, T: np.ndarray=np.eye(4), prefix: str='envmap', time: int=0):
    """Visualize the environment map as a sphere with the o3d rpc interface.
    Args:
        equirectangular_image: The equirectangular image with dtype np.uint8
        T: The world to envmap transform. This transforms a point into the coordinate system of the environment map
        prefix: Prefix str for the scene graph
        time: Time value for the scene graph
    """
    import open3d as o3d
    sphere = o3d.geometry.TriangleMesh.create_sphere(resolution=200)
    sphere = o3d.t.geometry.TriangleMesh.from_legacy(sphere)
    dirs = sphere.vertex['positions'].numpy()
    u, v = directions_to_uvs(dirs)
    y = equirectangular_image.shape[0]*v
    x = equirectangular_image.shape[1]*u
    y = np.clip(y, 0, equirectangular_image.shape[0]-1).astype(np.int32)
    x = np.clip(x, 0, equirectangular_image.shape[1]-1).astype(np.int32)
    colors = equirectangular_image[y,x]

    X = np.concatenate((dirs, np.ones_like(dirs[:,:1])), axis=-1)
    X = X@np.linalg.inv(T).T
    o3d.io.rpc.set_mesh_data(path=prefix, time=time, vertices=X[:,:3], faces=sphere.triangle['indices'], vertex_attributes={'Colors': colors})


def create_equirectangular_from_theta_dfe(image: np.ndarray, width=3648, height=1824):
    """Creates an equirectangular image from the Theta Z1 double fisheye.
    Args:
        image: Image with dtype float64
        width: Width of the output equirectangular projection in pixels.
        height: Height of the output equirectangular projection in pixels.
    Returns:
        The equirectangular projection.
    """
    def easeInOutCubic(x, start_x=0, end_x=1): 
        x_ = np.clip((x-start_x)/(end_x-start_x), 0, 1)
        return np.where(x_ < 0.5, 4*x_**3, 1- (-2*x_+2)**3/2)

    calib = create_equirectangular_from_theta_dfe.calib
    if not create_equirectangular_from_theta_dfe.calib:
        # rotate the camera rig to match the convention that +z is the upper border of the equirectangular image
        M = np.eye(4)
        M[:3,:3] = Rotation.from_rotvec([0,np.pi/2,0]).as_matrix() @ Rotation.from_rotvec([np.pi/2,0,0]).as_matrix()
        for x in ('left', 'right'):
            for y in ('K', 'dist_coeffs', 'width_height', 'rig_extrinsics'):
                calib[x][y] = np.loadtxt(CALIBRATION_PATH/'intrinsics'/f'theta_{x}_{y}.txt')
            calib[x].width = calib[x]['width_height'][0]
            calib[x].height = calib[x]['width_height'][1]
            calib[x].world2cam = calib[x]['rig_extrinsics'] @ M
            
            calib[x].K[0,0] /= 2
            calib[x].K[1,1] /= 2
            calib[x].K[0,2] /= 2
            calib[x].K[1,2] /= 2
            calib[x].width_height /= 2
            calib[x].width /= 2
            calib[x].height /= 2

    assert image.dtype in (np.float32, np.float64)
    assert image.shape[:2] == (calib.left.height, calib.left.width*2)

    dirs = get_dirvec_image(width, height)
    
    d = addict.Dict()
    d.left.im = image[:, :image.shape[1]//2]
    d.right.im = image[:, image.shape[1]//2:]

    for x in ('left', 'right'):
        xy, theta = fisheye_project(dirs.reshape(-1,3), calib[x].K, calib[x].dist_coeffs, calib[x].world2cam[:3,:3], calib[x].world2cam[:3,3])
        xy = xy.reshape(dirs.shape[0], dirs.shape[1], xy.shape[-1])
        d[x].theta = theta.reshape(*xy.shape[:2])
        d[x].er = cv2.remap(d[x].im, xy.astype(np.float32), None, cv2.INTER_LINEAR )

    thr1 = np.deg2rad(90)
    thr2 = 92
    d.left.mask = np.zeros((1,))
    d.right.mask = np.zeros((1,))
    # while (d.left.mask+d.right.mask).min() == 0.0:
    d.left.mask = easeInOutCubic(d.left.theta, np.deg2rad(thr2), thr1)
    d.right.mask = easeInOutCubic(d.right.theta, np.deg2rad(thr2), thr1)
        # thr2 += 1
    print(thr2)
    thr2 = np.deg2rad(thr2)

    er = (d.left.er*d.left.mask[...,None] + d.right.er*d.right.mask[...,None])/(d.left.mask+d.right.mask)[...,None] 
    return er #, d.left.mask+d.right.mask
create_equirectangular_from_theta_dfe.calib = addict.Dict()


def rotate_envmap_simple(equirectangular_image: np.ndarray, T: np.ndarray):
    """Rotate the environment map and return a new equirectangular image.
    Args:
        equirectangular_image: The equirectangular image with dtype np.uint8
        T: The 4x4 transformation matrix. Only the rotational part will be used.
    Returns:
        The rotated equirectangular image.
    """
    assert equirectangular_image.dtype == np.float32
    w = equirectangular_image.shape[1]
    h = equirectangular_image.shape[0]
    dirs = get_dirvec_image(width=w, height=h)
    dirs = dirs.reshape(-1, 3)
    # append zero to each vector
    dirs = np.concatenate([dirs, np.zeros_like(dirs[...,:1])], axis=-1)
    dirs = dirs @ T.T
    
    u,v = directions_to_uvs(dirs[...,:3])
    u = u.reshape(h,w)
    v = v.reshape(h,w)
    x = np.clip((u*w - 0.5).astype(np.float32), 0, w-1)
    y = np.clip((v*h - 0.5).astype(np.float32), 0, h-1)
    ans = cv2.remap(equirectangular_image, x, y, cv2.INTER_LINEAR)
    return ans
