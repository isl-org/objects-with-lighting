# SPDX-License-Identifier: Apache-2.0
import numpy as np
import addict
from utils.constants import *

def create_pinhole_camera_parameters(fov_deg, center, eye, up, width, height):
    """Creates the intrinsics and extrinsic pinhole camera parameters.
    
    Args:
        fov_deg (float): Field of view in degrees.
        center (np.ndarray): The point the camera is looking at.
        eye (np.ndarray): The position of the camera.
        up (np.ndarray): The up vector of the camera.
        width (int): The width of the image in pixels.
        height (int): The height of the image in pixels.

    Returns:
        A tuple of (intrinsics, extrinsics) where intrinsics is a 3x3 matrix and
        extrinsics is the 4x4 world to camera transform.
    """
    center = np.asarray(center)
    eye = np.asarray(eye)
    up = np.asarray(up)
    assert center.shape == (3,)
    assert eye.shape == (3,)
    assert up.shape == (3,)
    assert width > 0
    assert height > 0
    assert fov_deg > 0
    focal_length = 0.5 * width / np.tan(0.5 * np.deg2rad(fov_deg))
    K = np.eye(3)
    K[0,0] = focal_length
    K[1,1] = focal_length
    K[0,2] = 0.5*width
    K[1,2] = 0.5*height

    R = np.eye(3)
    R[1,:] = up/np.linalg.norm(up)
    R[2,:] = center - eye
    R[2,:] /= np.linalg.norm(R[2,:])
    R[0,:] = np.cross(R[1,:], R[2,:])
    R[0,:] /= np.linalg.norm(R[0,:])
    R[1,:] = np.cross(R[2,:], R[0,:])
    t = -R @ eye[:,None]
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t[:,0]
    return K, T


def read_eos_intrinsics():
    """Read intrinsics for the eos camera and return them as a dict."""
    if read_eos_intrinsics.ans is not None:
        return read_eos_intrinsics.ans
    intrinsics = addict.Dict()
    intrinsics_dir = CALIBRATION_PATH/'intrinsics'
    for k in ('dist_coeffs', 'K', 'width_height'):
        intrinsics[k] = np.loadtxt(intrinsics_dir/f'eos90d_{k}.txt')
    intrinsics['width'], intrinsics['height'] = intrinsics['width_height']
    read_eos_intrinsics.ans = intrinsics
    return intrinsics
read_eos_intrinsics.ans = None


def fisheye_project(points: np.ndarray, K: np.ndarray, dist_coeffs: np.ndarray, R: np.ndarray, t: np.ndarray):
    """Modified OpenCV fisheye projection to support angles > 180.
    Args:
        points: 3D points with shape (N,3)
        K: Intrinsic camera matrix with shape (3,3)
        dist_coeffs: k1,k2,k3,k4 distortion coefficients
        R: World to cam rotation matrix with shape (3,3) as in x = RX+t
        t: World to cam translation vector with shape (3,3) as in x = RX+t
    Returns:
        This function returns two arrays:
         - The 2d positions in the image
         - The angle theta with respect to the optical axis
    """
    dist_coeffs = np.squeeze(np.asarray(dist_coeffs))
    assert points.shape[-1] == 3 and points.ndim == 2
    assert K.shape == (3,3)
    assert dist_coeffs.shape == (4,)
    assert R.shape == (3,3)
    assert t.shape == (3,)
    Xcam = points @ R.T + t[None,:]

    ## OpenCV
    # a = Xcam[:,0]/Xcam[:,2]
    # b = Xcam[:,1]/Xcam[:,2]
    # r = np.sqrt(a**2 + b**2)
    # theta = np.arctan(r)
    theta = np.arccos(((Xcam/np.linalg.norm(Xcam, keepdims=True, axis=-1))*np.array([[0,0,1.0]])).sum(axis=-1))
    theta_d = theta * (1 + dist_coeffs[0]*theta**2 + dist_coeffs[1]*theta**4 + dist_coeffs[2]*theta**6 + dist_coeffs[3]*theta**8 )

    ## OpenCV
    # x = a*theta_d/r
    # y = b*theta_d/r
    ab = Xcam[:,:2]/np.linalg.norm(Xcam[:,:2], keepdims=True, axis=-1)
    x = ab[:,0]*theta_d
    y = ab[:,1]*theta_d

    x = np.stack([x,y,np.ones_like(x)], axis=-1)
    x = x @ K.T
    return x[...,:2], theta
