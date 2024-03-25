# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
import argparse
import numpy as np
import cv2
import json
from utils import hdri
from utils.constants import *
from utils import features
from tqdm import tqdm
import addict
import pycolmap


def fisheye_undistort(im, K, dist_coeff):
    """Shortcut for fisheye undistortion"""
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, dist_coeff, np.eye(3), K, im.shape[::-1], cv2.CV_16SC2 )
    ans = cv2.remap(im, map1, map2, cv2.INTER_LINEAR)
    return ans

def imshow(*args, **kwargs):
    """cv2.imshow wrapped in try except block to skip display problems when running remotely"""
    try:
        cv2.imshow(*args, **kwargs)
    except:
        pass

def theta_intrinsics_calibration(args):

    paths = sorted(list((CALIBRATION_PATH/'intrinsics/theta/').glob('*.DNG')))

    with open(CALIBRATION_PATH/'targets/atag_board_keypoints.json','r') as f:
        keypoints = json.load(f)

    # visualize calibration target
    if args.visualize:
        import open3d as o3d
        from utils.reconstruction import visualize_camera
        points = np.array(list(keypoints.values()))
        o3d.io.rpc.set_mesh_data(path='cam_calib/target_points', vertices=points)

    left = addict.Dict()
    right = addict.Dict()

    print('Loading images..')
    left.images = []
    right.images = []
    for p in tqdm(paths):
        rawmeta = hdri.read_raw_and_meta(p)
        im = rawmeta.raw.postprocess()
        im = features.get_grayscale_array_for_detection(im)
        left.images.append(im[:,:im.shape[1]//2])
        right.images.append(im[:,im.shape[1]//2:])

    detector = features.ApriltagDetector()


    for name, data in (('left',left), ('right',right)):
        print(f'Calibrate {name} fisheye..')
        data.obj_points = []
        data.img_points = []
        data.detections = []
        for im in data.images:
            ans, dbg = detector.detect(im, min_size=50, debug=True)
            data.detections.append(ans)

            if args.visualize:
                imshow('detection', np.asarray(dbg)[...,[2,1,0]])

            # we need all tags for computing the intrinsics
            if len(ans) == len(keypoints)//4:
                points2d = []
                points3d = []
                for tag in ans:
                    for corner_i, corner in enumerate(tag.corners):
                        points2d.append(corner)
                        points3d.append(keypoints[f'{tag.family}.{tag.id:02d}.{corner_i}'])

                data.img_points.append(np.array(points2d, dtype=np.float32)[None,...])
                data.obj_points.append(np.array(points3d, dtype=np.float32)[None,...])

        flags = cv2.fisheye.CALIB_FIX_SKEW | cv2.fisheye.CALIB_FIX_K3 | cv2.fisheye.CALIB_FIX_K4 | cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        rms, K, dist_coeff, rvecs, tvecs = cv2.fisheye.calibrate(data.obj_points, data.img_points, im.shape[::-1], None, None, flags=flags)
        print(rms)
        data.best_rms, data.best_K, data.best_dist_coeff = rms, K.copy(), dist_coeff.copy()
        data.best_rvecs, data.best_tvecs = rvecs, tvecs

        rms, K, dist_coeff, rvecs, tvecs = cv2.fisheye.calibrate(data.obj_points, data.img_points, im.shape[::-1], None, None, flags=flags | cv2.fisheye.CALIB_FIX_K2)
        print(rms)
        if rms < data.best_rms:
            data.best_rms, data.best_K, data.best_dist_coeff = rms, K.copy(), dist_coeff.copy()
            data.best_rvecs, data.best_tvecs = rvecs, tvecs
        rms, K, dist_coeff, rvecs, tvecs = cv2.fisheye.calibrate(data.obj_points, data.img_points, im.shape[::-1], K, dist_coeff, flags=flags | cv2.fisheye.CALIB_USE_INTRINSIC_GUESS)
        print(rms)
        if rms < data.best_rms:
            data.best_rms, data.best_K, data.best_dist_coeff = rms, K.copy(), dist_coeff.copy()
            data.best_rvecs, data.best_tvecs = rvecs, tvecs

        if args.visualize:
            undistorted = fisheye_undistort(im, K, dist_coeff)
            imshow('original and undistorted last image', np.concatenate([im, undistorted], axis=-1))
            for i, (rvec, tvec) in enumerate(zip(data.best_rvecs, data.best_tvecs)):
                R,_ = cv2.Rodrigues(rvec)
                visualize_camera(R, tvec[:,0], f'cam_calib/intrinsics_{name}', time=i)

        np.savetxt(CALIBRATION_PATH/f'intrinsics/theta_{name}_K.txt', data.best_K)
        np.savetxt(CALIBRATION_PATH/f'intrinsics/theta_{name}_dist_coeffs.txt', data.best_dist_coeff)
        np.savetxt(CALIBRATION_PATH/f'intrinsics/theta_{name}_width_height.txt', [data.images[0].shape[1], data.images[0].shape[0]])

    # calibrate extrinsics
    
    # select image pair for which we have as many detections as possible in both images
    best_i = 0
    best_score = 0
    for i, (l,r) in enumerate(zip(left.detections, right.detections)):
        score = len(l) * len(r)
        if score > best_score:
            best_i = i
            best_score = score
    

    ## solvepnp from colmap with fisheye camera model
    for name, data in (('left',left), ('right',right)):
        points2d = []
        points3d = []
        for tag in data.detections[best_i]:
            for corner_i, corner in enumerate(tag.corners):
                points2d.append(corner)
                points3d.append(keypoints[f'{tag.family}.{tag.id:02d}.{corner_i}'])
        points2d = np.array(points2d, dtype=np.float32)
        points3d = np.array(points3d, dtype=np.float32)

        undist_points2d = cv2.fisheye.undistortPoints(points2d[None,...], data.best_K, data.best_dist_coeff)
        K = data.best_K
        dist_coeff = np.squeeze(data.best_dist_coeff)
        im = data.images[0]

        cam = pycolmap.Camera('OPENCV_FISHEYE', im.shape[1], im.shape[0], [K[0,0],K[1,1],K[0,2],K[1,2], dist_coeff[0], dist_coeff[1], dist_coeff[2], dist_coeff[3]])
        ans = pycolmap.absolute_pose_estimation(points2d, points3d, cam)
        print(ans)

        R = pycolmap.qvec_to_rotmat(ans['qvec'])
        data.T = np.eye(4)
        data.T[:3,:3] = R
        data.T[:3,3] = ans['tvec']
        if args.visualize:
            visualize_camera(R, ans['tvec'], f'cam_calib/extrinsics_{name}', time=0)
    

    inv_T = np.linalg.inv(left.T)
    left.T = np.eye(4)
    right.T = right.T @ inv_T

    # center the camera rig
    R = right.T[:3,:3]
    t = right.T[:3,3]
    inv_right_T = np.linalg.inv(right.T)
    C = inv_right_T[:3,3]
    left.T[:3,3] = 0.5*C
    inv_right_T[:3,3] -= 0.5*C
    right.T = np.linalg.inv(inv_right_T)

    for name, data in (('left',left), ('right',right)):
        if args.visualize:
            visualize_camera(data.T[:3,:3], data.T[:3,3], f'cam_calib/extrinsics_normalized_{name}', time=0)
        np.savetxt(CALIBRATION_PATH/f'intrinsics/theta_{name}_rig_extrinsics.txt', data.T)






def eos_intrinsics_calibration(args):

    paths = sorted(list((CALIBRATION_PATH/'intrinsics/eos90d/').glob('*.CR3')))

    with open(CALIBRATION_PATH/'targets/atag_calibration_target_keypoints.json','r') as f:
        keypoints = json.load(f)

    print('Loading images..')
    images = []
    for p in tqdm(paths):
        rawmeta = hdri.read_raw_and_meta(p)
        im = rawmeta.raw.postprocess(user_flip=0)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        images.append(im)

    detector = features.ApriltagDetector('tagStandard41h12')

    obj_points = []
    img_points = []
    for im in images:
        ans = detector.detect(im, min_size=10)
        if len(ans) == 54:
            points2d = []
            points3d = []
            for tag in ans:
                for corner_i, corner in enumerate(tag.corners):
                    points2d.append(corner)
                    points3d.append(keypoints[f'{tag.family}.{tag.id}.{corner_i}'])

            img_points.append(np.array(points2d, dtype=np.float32))
            obj_points.append(np.array(points3d, dtype=np.float32))

    flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
    rms, K, dist_coeff, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, im.shape[::-1], None, None, flags=flags | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2); print(rms)
    rms, K, dist_coeff, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, im.shape[::-1], K, dist_coeff, flags=flags | cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_K2); print(rms)

    rms, K, dist_coeff, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, im.shape[::-1], K, dist_coeff, flags=flags | cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_FOCAL_LENGTH); print(rms)
    rms, K, dist_coeff, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, im.shape[::-1], K, dist_coeff, flags=flags | cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_K2); print(rms)
    rms, K, dist_coeff, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, im.shape[::-1], K, dist_coeff, flags=flags | cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_FOCAL_LENGTH); print(rms)
    rms, K, dist_coeff, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, im.shape[::-1], K, dist_coeff, flags=flags | cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_K2); print(rms)
    rms, K, dist_coeff

    np.savetxt(CALIBRATION_PATH/'intrinsics/eos90d_K.txt', K)
    np.savetxt(CALIBRATION_PATH/'intrinsics/eos90d_dist_coeffs.txt', dist_coeff[0])
    np.savetxt(CALIBRATION_PATH/'intrinsics/eos90d_width_height.txt', [images[0].shape[1], images[0].shape[0]])



def main():
    parser = argparse.ArgumentParser(
        description="""Computes the camera intrinsics for the cameras""",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.set_defaults(
        func=lambda x: print("please specify a camera (theta, eos90d)"))
    parser.add_argument("--visualize", action='store_true', help="If True show intermediate images for debugging")

    subparsers = parser.add_subparsers()

    theta_parser = subparsers.add_parser(
        "theta",
        help="Command for computing the intrinsics for the theta z1 camera.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    theta_parser.set_defaults(func=theta_intrinsics_calibration)

    eos_parser = subparsers.add_parser(
        "eos",
        help="Command for computing the intrinsics for the eos90d camera.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    eos_parser.set_defaults(func=eos_intrinsics_calibration)
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    args.func(args)
    return 0



    
if __name__ == '__main__':
    main()
