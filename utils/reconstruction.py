# SPDX-License-Identifier: Apache-2.0
import numpy as np
from pathlib import Path
import pycolmap
import cv2
import subprocess
from tqdm import tqdm
from itertools import combinations
from collections import defaultdict
import tempfile
from typing import Dict, List, Union
import utils.colmap_database as coldb
from utils.features import *
from utils.geometry import *
from utils.constants import *
from utils.reconstruction_neus import *
import re


def init_colmap_reconstruction(output_dir: Path, image_dir: Path, overwrite: bool=False, pose_priors: Path=None):
    """Initializes a colmap reconstruction with SIFT feature extraction and matching.
    Args:
        output_dir: The root directory where the database and the reconstructions will be stored.
        image_dir: The image directory. Image names will be relative to this directory.
        pose_priors: Path to the npz file with prior pose information. If available this will be 
                     used to run the custom matcher.

    """
    output_dir = Path(output_dir)
    image_dir = Path(image_dir)
    output_dir.mkdir(exist_ok=overwrite)

    db_path = output_dir/'database.db'
    if overwrite and db_path.exists():
        db_path.unlink()

    cmd = [COLMAP_BIN, 'feature_extractor', '--image_path', str(image_dir), '--database_path', str(db_path), 
        '--ImageReader.single_camera', 1,
        '--SiftExtraction.num_octaves', 5,
        '--SiftExtraction.use_gpu', 0,
    ]
    print(' '.join(list(map(str,cmd))))
    status = subprocess.check_call(list(map(str,cmd)), shell=False)
    print(status)

    if pose_priors is not None:
        create_match_list(output_dir, pose_priors)
        custom_matcher(output_dir)
    else:
        exhaustive_matcher(output_dir)


def create_match_list(output_dir: Path, pose_priors: Path, angle_threshold: float=90):
    """This function creates the 'match_list.txt' for custom matching using an angle threshold.

    Args:
        output_dir: The root directory where the database and the reconstructions will be stored.
        pose_priors: Path to the npz file with prior pose information.
        angle_threshold: Angle threshold for neighboring images in degree
    """
    threshold = np.cos(np.deg2rad(angle_threshold))

    match_list_path = output_dir/'match_list.txt'
    priors = {k:v for k,v in np.load(pose_priors).items()}
    image_names = set()
    for k in priors.keys():
        image_names.add(re.match('(.*):.*', k).group(1))
    image_names = sorted(list(image_names))
    lines = []
    for i, im1 in enumerate(image_names):
        view_dir1 = priors[f'{im1}:R'][2,:]
        for im2 in image_names[(i+1):]:
            view_dir2 = priors[f'{im2}:R'][2,:]
            dot = view_dir1.dot(view_dir2)
            print(im1,im2, dot, threshold)
            if dot > threshold:
                lines.append(f'{im1} {im2}\n')
    with open(match_list_path, 'w') as f:
        f.writelines(lines)

def custom_matcher(output_dir: Path):
    """Runs the custom matcher using pose priors to identify image neighbours.
    Args:
        output_dir: The root directory where the database and the reconstructions will be stored.
        pose_priors: Path to the npz file with prior pose information.
        angle_threshold: Angle threshold for neighboring images in degree
    """
    output_dir = Path(output_dir)
    db_path = output_dir/'database.db'

    match_list_path = output_dir/'match_list.txt'

    cmd = [COLMAP_BIN, 'matches_importer', '--database_path', str(db_path), 
        '--match_list_path', str(match_list_path),
        '--SiftMatching.min_num_inliers', 15,
        '--SiftMatching.use_gpu', 0,
    ]
    status = subprocess.check_call(map(str,cmd), shell=False)
    print(status)


def exhaustive_matcher(output_dir: Path):
    """Runs the exhaustive matcher with modified default parameters.
    Args:
        output_dir: The root directory where the database and the reconstructions will be stored.
    """
    output_dir = Path(output_dir)
    db_path = output_dir/'database.db'
    cmd = [COLMAP_BIN, 'exhaustive_matcher', '--database_path', str(db_path), 
        '--SiftMatching.min_num_inliers', 15,
        '--SiftMatching.use_gpu', 0,
    ]
    status = subprocess.check_call(map(str,cmd), shell=False)
    print(status)


def force_single_camera(db_path: Path):
    """Workaround because we cannot configure pycolmap.extract_features to only create one camera."""
    db = coldb.COLMAPDatabase(db_path)
    cameras = db.get_cameras()
    cam = cameras[1]
    db.execute('DROP TABLE IF EXISTS cameras')
    db.create_cameras_table()
    db.add_camera(camera_id=1, **cam)

    images = db.get_images()
    for image_id, image in images.items():
        image['camera_id'] = 1
        db.update_image(image_id=image_id, **image)

    db.commit()
    db.close()


def update_cam_with_opencv_params(db_path: Path, K: np.ndarray, dist_coeffs: np.ndarray):
    """Updates the camera parameters in the database with the params for the OpenCV camera model.
    Args:
        db_path: Path to the dabase.
        K: Intrinsic 3x3 camera matrix with fx,fy,cx,cy.
        dist_coeffs: Distortion coefficients k1,k2,p1,2,k3. k3 will be ignored as it is not supported by colmap.
    """
    db = coldb.COLMAPDatabase(db_path)
    cameras = db.get_cameras()
    for cam_id, cam in cameras.items():
        cam['model'] = coldb.CameraModel.OPENCV
        cam['params'] = np.array([K[0,0], K[1,1], K[0,2], K[1,2], dist_coeffs[0], dist_coeffs[1], dist_coeffs[2], dist_coeffs[3]]) # fx fy cx cy k1 k2 p1 p2
        cam['prior_focal_length'] = True
        db.update_camera(camera_id=cam_id, **cam)
    db.commit()
    db.close()


def add_custom_keypoints_to_colmap_reconstruction(output_dir: Path, image_keypoints: Dict[str, List[CustomKeypoint]]):
    """Adds custom keypoints to the colmap database.
    Args:
        output_dir: The root directory where the database and the reconstructions will be stored.
        image_keypoints: Dictionary storing the keypoints for each image. The key is the name of the image.
    """
    output_dir = Path(output_dir)
    db_path = output_dir/'database.db'

    db = coldb.COLMAPDatabase(db_path)

    imageid_image = db.get_images()
    imgname_id = { v['name']: k for k,v in imageid_image.items()}

    imageid_kpts = {imgname_id[k]: v for k, v in image_keypoints.items() if k in imgname_id}

    # update descriptors and keypoints
    for img_id, tags in imageid_kpts.items():
        old_descriptors = db.get_descriptors(img_id)
        old_keypoints = db.get_keypoints(img_id)
        descriptors = []
        keypoints = np.empty((len(tags),6), dtype=np.float32)
        for tag_i, t in enumerate(tags):
            t.colmap_id = old_descriptors.shape[0] + tag_i
            descriptors.append(t.create_dummy_descriptor())
            keypoints[tag_i,:2] = t.point
            keypoints[tag_i,2:] = [1,0,0,1]

        descriptors = np.stack(descriptors, axis=0)
        descriptors = np.concatenate((old_descriptors, descriptors), axis=0)
        db.update_descriptors(img_id, descriptors)

        keypoints = np.concatenate((old_keypoints, keypoints), axis=0)
        db.update_keypoints(img_id, keypoints)


    # exhaustive matching
    for img_id1, img_id2 in tqdm(combinations(imageid_kpts.keys(), 2)):
        matches = []
        for tag1 in imageid_kpts[img_id1]:
            for tag2 in imageid_kpts[img_id2]:
                if tag1.id == tag2.id:
                    matches.append((tag1.colmap_id, tag2.colmap_id))
        if matches:
            old_matches = db.get_matches(img_id1, img_id2)
            if old_matches is not None:
                matches = np.concatenate((old_matches, np.array(matches)), axis=0)
                db.update_matches(img_id1, img_id2, matches)
            else:
                db.add_matches(img_id1, img_id2, np.array(matches))

    # remove the two_view_geometries to include the new keypoints in the estimation
    db.execute('DROP TABLE IF EXISTS two_view_geometries')
    db.create_two_view_geometries_table()
    db.commit()
    db.close()
    
    match_list_path = output_dir/'match_list.txt'
    if match_list_path.exists():
        custom_matcher(output_dir)
    else:
        exhaustive_matcher(output_dir)


def add_apriltag_keypoints_to_colmap_reconstruction(output_dir: Path, image_dir: Path, debug: bool=False):
    """Detects and adds april tag corner points to the colmap database.
    Args:
        output_dir: The root directory where the database and the reconstructions will be stored.
        image_dir: The image directory. Image names will be relative to this directory.
    """
    output_dir = Path(output_dir)
    image_dir = Path(image_dir)
    db_path = output_dir/'database.db'

    if debug:
        (output_dir/'dbg').mkdir(exist_ok=True)

    db = coldb.COLMAPDatabase(db_path)

    imageid_image = db.get_images()
    imgname_id = { v['name']: k for k,v in imageid_image.items()}

    db.close()

    # detect
    image_tagkps = {}
    detector = ApriltagDetector(valid_ids=DEFAULT_VALID_TAG_IDS)
    image_paths = [image_dir/x for x in imgname_id.keys()]
    print('Detecting April tags')
    for p in tqdm(image_paths):
        ans = detector.detect(get_grayscale_array_for_detection(p), debug=debug)
        if debug:
            tags, dbg = ans
            dbg.save(output_dir/'dbg'/f'at_{p.name}')
        else:
            tags = ans
        tagkps = []
        for tag in tags:
            tagkps.extend(convert_tag_to_customkeypoints(tag))
        if tagkps:
            image_tagkps[p.name] = tagkps

    add_custom_keypoints_to_colmap_reconstruction(output_dir, image_tagkps)
    


def run_mapper(output_dir: Path, image_dir: Path):
    """Runs the incremental mapper.

    This function is just for convenience to run with specific options.

    Args:
        output_dir: The root directory where the database and the reconstructions will be stored.
        image_dir: The image directory. Image names will be relative to this directory.

    """
    output_dir = Path(output_dir)
    image_dir = Path(image_dir)
    db_path = output_dir/'database.db'
    # return pycolmap.incremental_mapping(db_path, image_dir, output_dir, options)
    cmd = [COLMAP_BIN, 'mapper', '--database_path', db_path, '--output_path', output_dir, '--image_path', image_dir,
        '--Mapper.multiple_models', 0,
        '--Mapper.ba_refine_focal_length', int(False),
        '--Mapper.ba_refine_extra_params', int(False),
        '--Mapper.filter_max_reproj_error', 10,
        '--Mapper.abs_pose_min_num_inliers', 8,
        '--Mapper.max_reg_trials', 10,
    ]
    status = subprocess.check_call(map(str,cmd), shell=False)
    print(status)


def point_triangulator(output_dir: Path, image_dir: Path, filter_max_reproj_error: int=10, model_subdir: str='0' ):
    """Runs the point triangulator which triangulates new points followed by bundle adjustment.

    Args:
        output_dir: The root directory where the database and the reconstructions will be stored.
        image_dir: The image directory. Image names will be relative to this directory.

    """
    output_dir = Path(output_dir)
    image_dir = Path(image_dir)
    db_path = output_dir/'database.db'
    cmd = [COLMAP_BIN, 'point_triangulator', '--database_path', db_path, '--input_path', output_dir/model_subdir, '--output_path', output_dir/model_subdir, '--image_path', image_dir,
        '--Mapper.multiple_models', 0,
        '--Mapper.ba_refine_focal_length', int(False),
        '--Mapper.ba_refine_extra_params', int(False),
        '--Mapper.filter_max_reproj_error', filter_max_reproj_error,
        '--Mapper.abs_pose_min_num_inliers', 15,
        '--Mapper.tri_merge_max_reproj_error', 8,
    ]
    status = subprocess.check_call(map(str,cmd), shell=False)
    print(status)




def run_bundle_adjustment(input_dir: Path, output_dir: Path, refine_focal_length=True, refine_principal_point=False, refine_extra_params=True, refine_extrinsics=True, constant_point_ids=None):
    """Runs bundle adjustment.
    Args:
        input_dir: Path to the input model directory with the cameras.bin images.bin and points3D.bin files.
        output_dir: Path to the output model directory with the cameras.bin images.bin and points3D.bin files.
    """
    cmd = [COLMAP_BIN, 'bundle_adjuster', '--input_path', str(input_dir), '--output_path', str(output_dir), 
        '--BundleAdjustment.refine_focal_length', str(int(refine_focal_length)),
        '--BundleAdjustment.refine_principal_point', str(int(refine_principal_point)),
        '--BundleAdjustment.refine_extra_params', str(int(refine_extra_params)),
        '--BundleAdjustment.refine_extrinsics', str(int(refine_extrinsics)),
    ]
    
    if constant_point_ids is not None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(Path(tmpdir)/'constant_points','w') as f:
                f.write(','.join(map(str,constant_point_ids)))
            cmd += ['--constant_point_ids_path', str(Path(tmpdir)/'constant_points')]
            status = subprocess.check_call(cmd, shell=False)
    else:
        status = subprocess.check_call(cmd, shell=False)
    
    print(status)


def read_apriltag_keypoints_json(json_path: Path) -> Dict:
    """Reads the json file which describes the 3D position of the apriltag corners."""
    import json
    json_path = Path(json_path)

    with open(json_path, 'r') as f:
        data = json.load(f)
    result = {}
    for k, v in data.items():
        k = k.split('.')
        # convert list to tuple and remove possible zero padding in the key
        result[f'{k[0]}.{int(k[1])}.{int(k[2])}'] = tuple(v)
    return result


def get_custom_keypoints_from_db(db_or_path: Union[coldb.COLMAPDatabase, Path]) -> Dict[int, Dict[int, CustomKeypoint]]:
    """Returns a dictionary of the custom keypoints for all images.
    Args:
        db_or_path: Either a colmap database or a path to a colmap database
    Returns:
        A dict with the image id as key and a dictionary with colmap point id and custom keypoint as value.
    """
    if isinstance(db_or_path,(str,Path)):
        db = coldb.COLMAPDatabase(db_or_path)
    else:
        db = db_or_path

    images = db.get_images()
    imgid_colmapkpid_kp = {}
    for img_id in images.keys():
        descriptors = db.get_descriptors(img_id)
        keypoints = db.get_keypoints(img_id)

        colmapkpid_kp = {}
        for i, (desc, kp) in enumerate(zip(descriptors, keypoints)):
            identifier = CustomKeypoint.identify_from_descriptor(desc)
            if identifier:
                colmapkpid_kp[i] = CustomKeypoint(identifier, tuple(kp[:2]), i)
        imgid_colmapkpid_kp[img_id] = colmapkpid_kp
    
    if isinstance(db_or_path,(str,Path)):
        db.close()

    return imgid_colmapkpid_kp
    

def get_2d_keypoints(db_or_path: Union[coldb.COLMAPDatabase, Path]) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Returns a dictionary of the custom keypoint pixel positions for all images.
    Args:
        db_or_path: Either a colmap database or a path to a colmap database
    Returns:
        A dict with the image name (without the suffix) as key and a mapping from keypoint id (str) to the pixel position as value.
    """
    if isinstance(db_or_path,(str,Path)):
        db = coldb.COLMAPDatabase(db_or_path)
    else:
        db = db_or_path

    imgid_colmapkpid_kp = get_custom_keypoints_from_db(db)
    
    images = db.get_images()
    imgname_kpid_xy = {}
    for img_id, img in images.items():
        kpid_xy = {}
        for kp in imgid_colmapkpid_kp[img_id].values():
            kpid_xy[kp.id] = kp.point
        imgname_kpid_xy[Path(img['name']).stem] = kpid_xy
    
    if isinstance(db_or_path,(str,Path)):
        db.close()
    
    return imgname_kpid_xy
    

def get_3d_keypoints(db_or_path: Union[coldb.COLMAPDatabase, Path], recon: pycolmap.Reconstruction) -> Dict[str, Tuple[float,float,float]]:
    """Returns a dictionary of 3d points of custom keypoints
    Args:
        db_or_path: Either a colmap database or a path to a colmap database
        recon: COLMAP reconstruction
    Returns:
        A dict with the keypoint id as key and the xyz positions as value.
    """
    imgid_colmapkpid_kp = get_custom_keypoints_from_db(db_or_path)
    kpname_xyz = defaultdict(list)
    for k, p in recon.points3D.items():
        el = p.track.elements[0]
        kp = imgid_colmapkpid_kp[el.image_id].get(el.point2D_idx)
        if kp is None:
            continue
        kpname_xyz[kp.id].append((tuple(p.xyz), len(p.track.elements)))

    # colmap may create duplicate 3d points.
    # In this case return the 3d point with the most track elements.
    return {k: sorted(v,key=lambda x:x[-1], reverse=True)[0][0] for k,v in kpname_xyz.items()}


def merge_duplicate_keypoint_point3Ds(db_or_path: Union[coldb.COLMAPDatabase, Path], recon: pycolmap.Reconstruction):
    """Merges 3d points that describe the same custom keypoint
    Args:
        db_or_path: Either a colmap database or a path to a colmap database
        recon: COLMAP reconstruction
    Returns:
        Returns a reference to the modified input reconstruction.
    """
    imgid_colmapkpid_kp = get_custom_keypoints_from_db(db_or_path)
    kpname_p3Did = defaultdict(list)
    key_point3d_ids = set()
    for k, p in recon.points3D.items():
        el = p.track.elements[0]
        kp = imgid_colmapkpid_kp[el.image_id].get(el.point2D_idx)
        if kp is None:
            continue
        key_point3d_ids.add(k)
        kpname_p3Did[kp.id].append(k)

    points3D_replace = {}
    point3D_ids_delete = set()
    for k, p in recon.points3D.items():
        if k in key_point3d_ids:
            el = p.track.elements[0]
            kp = imgid_colmapkpid_kp[el.image_id].get(el.point2D_idx)
            point3D_ids = kpname_p3Did[kp.id]
            if k == min(point3D_ids):
                w_sum = 0
                xyz = np.zeros((3,), dtype=np.float64)
                for p3Did in point3D_ids:
                    p3D = recon.points3D[p3Did]
                    w = p3D.track.length()
                    w_sum += w
                    xyz += w*p3D.xyz
                xyz /= w_sum
                p.xyz[:] = xyz

                for p3Did in [x for x in point3D_ids if x != k]:
                    # print(k, p.track.length())
                    p.track.add_elements(recon.points3D[p3Did].track.elements)
            else:
                point3D_ids_delete.add(k)
                    
            if len(point3D_ids) > 1:
                points3D_replace[k] = min(point3D_ids)
    
            
    num_points_before_merge = len(recon.points3D)
    for k in point3D_ids_delete:
        del recon.points3D[k]
    num_points_after_merge = len(recon.points3D)

    for k, im in recon.images.items():
        for p in im.points2D:
            new_p3D_id = points3D_replace.get(p.point3D_id)
            if new_p3D_id:
                # print(p.point3D_id, '->', new_p3D_id)
                p.point3D_id = new_p3D_id

    recon.check()
    print(f'before merging points {num_points_before_merge}, after merge {num_points_after_merge}')
    return recon


def visualize_camera(R, t, path, time=0):
    """Visualizes a camera with the o3d rpc interface"""
    import open3d as o3d
    c = o3d.camera.PinholeCameraParameters()
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t
    c.extrinsic = T
    c.intrinsic.intrinsic_matrix = np.eye(3)
    o3d.io.rpc.set_legacy_camera(c, path, time=time)


def visualize_reconstruction(recon: pycolmap.Reconstruction, imgid_colmapkpid_kp=None, sg_prefix='recon'):
    """Visualize the reconsruction using o3d rpc interface.
    Args:
        recon: The colmap reconstruction.
        imgid_colmapkpid_kp: The dictionary for identifying custom keypoints as returned by get_custom_keypoints_from_db().
        sg_prefix: Scene graph prefix.
    """
    import open3d as o3d
    points = []
    colors = []
    keypoints = []
    
    for k, p in recon.points3D.items():
        el = p.track.elements[0]
        if imgid_colmapkpid_kp is not None and el.point2D_idx in imgid_colmapkpid_kp[el.image_id]:
            keypoints.append(p.xyz)
        else:
            points.append(p.xyz)
            colors.append(p.color)
    points = np.array(points)
    colors = np.array(colors)
    keypoints = np.array(keypoints)
    o3d.io.rpc.set_mesh_data(path=sg_prefix+'/points', vertices=points, vertex_attributes={'Colors': colors})
    if keypoints.shape[0] > 0:
        o3d.io.rpc.set_mesh_data(path=sg_prefix+'/keypoints', vertices=keypoints, vertex_attributes={'Colors': np.full_like(keypoints, fill_value=(255,0,0), dtype=np.uint8)})

    for k, im in sorted(recon.images.items(), key=lambda x: x[1].name):
        visualize_camera(im.rotation_matrix(), im.tvec, path=sg_prefix+'/cameras', time=k)


def compute_transform_to_custom_keypoints(recon_dir: Path, src_recon: pycolmap.Reconstruction, dst_kp: Dict[str,Tuple[float,float,float]]) -> Tuple[float, np.ndarray]:
    """Computes the SE3 transform to align the reconstruction with common custom keypoints
    
    Args:
        recon_dir: The root directory where the database and the reconstructions will be stored.
        src_recon: The pycolmap Reconstruction.
        dst_kp: Dictionary of 3D points.

    Returns:
        Tuple with scale and the 4x4 rotation and translation matrix.
        The SE3 transform with scale to align the reconstruction with the custom keypoints.
    """
    recon_dir = Path(recon_dir)
    db_path = recon_dir/'database.db'

    db = coldb.COLMAPDatabase(db_path)

    imgid_colmapkpid_kp = get_custom_keypoints_from_db(db)

    src = []
    dst = []
    for k, p in src_recon.points3D.items():
        el = p.track.elements[0]
        kp = imgid_colmapkpid_kp[el.image_id].get(el.point2D_idx)
        if kp is None:
            continue

        p2 = dst_kp.get(kp.id)
        if p2 is not None:
            src.append(p.xyz)
            dst.append(p2)

    src = np.stack(src)
    dst = np.array(dst)

    return umeyama(src, dst)

def set_keypoint_positions_in_reconstruction(recon_dir: Path, recon: pycolmap.Reconstruction, keypoints: Dict[str,Tuple[float,float,float]]):
    """Identifies the custom keypoints in the reconstruction and sets them to the specified xyz positions.

    Args:
        recon_dir: The root directory where the database and the reconstructions will be stored.
        recon: The pycolmap Reconstruction.
        keypoints: Dictionary of 3D points.
    
    Returns:
        Returns the set of colmap ids for the 3D point for which the xyz position has been set.
    """
    recon_dir = Path(recon_dir)
    db_path = recon_dir/'database.db'

    db = coldb.COLMAPDatabase(db_path)

    imgid_colmapkpid_kp = get_custom_keypoints_from_db(db)

    point3D_ids = set()

    for id, p in recon.points3D.items():
        el = p.track.elements[0]
        kp = imgid_colmapkpid_kp[el.image_id].get(el.point2D_idx)
        if kp is None:
            continue

        point3D_ids.add(id)

        p2 = keypoints.get(kp.id)
        if p2 is not None:
            p.xyz[:] = p2

    return point3D_ids
    



def colmap_to_neus(intermediate_env_dir: Path, output_dir: Path):
    """Converts the colmap reconstruction to the NeuS format.
    Args:
        intermediate_env_dir: The environment directory with the intermediate data that contains 
                              the COLMAP reconstruction directory.
                              The parent dir must have a subfolder 'images'.
        output_dir: The output directory.
    """
    
    div = 4
    def downsample(arr):
        """Downsample the images for NeuS"""
        new_size = (arr.shape[1]//div, arr.shape[0]//div)
        return cv2.resize(arr, new_size, interpolation=cv2.INTER_AREA)
    
    env_dir = Path(intermediate_env_dir).resolve()
    object_name = env_dir.parent.name
    env_name = env_dir.name
    source_data_dir = SOURCE_DATA_PATH/object_name/env_name

    write_neus_config_file(output_dir)

    recon = pycolmap.Reconstruction(env_dir/'recon'/'0')
    cameras = recon.cameras
    images = recon.images

    K = np.loadtxt(env_dir/'K.txt')
    K[0,0] /= 4
    K[1,1] /= 4
    K[0,2] /= 4
    K[1,2] /= 4

    dist_coeffs = np.loadtxt(env_dir/'dist_coeffs.txt')
    assert np.count_nonzero(dist_coeffs) == 0

    bounds = np.loadtxt(source_data_dir/'object_bounding_box.txt')
    bounds = bounds.reshape(-1,2)
    center = np.mean(bounds, axis=-1)
    radius = 0.5*np.linalg.norm(bounds.max(axis=-1) - bounds.min(axis=-1))
    print('radius', radius)

    scale_mat = np.eye(4)
    scale_mat[:3,:3] = radius*np.eye(3)
    scale_mat[:3,3] = center

    scale_mat_inv = np.eye(4)
    scale_mat_inv[:3,:3] = 1/radius*np.eye(3)
    scale_mat_inv[:3,3] = -center/radius

    image_output_dir = output_dir/'image'
    image_output_dir.mkdir(exist_ok=True, parents=True)
    mask_output_dir = output_dir/'mask'
    mask_output_dir.mkdir(exist_ok=True)
    
    
    world_mats = []
    for im_id, im in images.items():
        R = pycolmap.qvec_to_rotmat(im.qvec)
        Rt = np.concatenate((R,im.tvec[:,None]), axis=-1)
        P = K @ Rt
        world_mat = np.eye(4)
        world_mat[:3,:] = P
        world_mats.append((im_id, world_mat))

    world_mats.sort(key=lambda x: x[0])

    cameras_sphere = {}
    for i, (_, wm) in enumerate(world_mats):
        cameras_sphere[f'world_mat_{i}'] = wm.astype(np.float64)
        cameras_sphere[f'world_mat_inv_{i}'] = np.linalg.inv(wm).astype(np.float64)
        cameras_sphere[f'scale_mat_{i}'] = scale_mat
        cameras_sphere[f'scale_mat_inv_{i}'] = scale_mat_inv
    
    np.savez(output_dir/'cameras_sphere.npz', **cameras_sphere)

    colmap_image_dir = env_dir/'images'
    images_sorted = sorted([(k, v) for k,v in images.items()], key=lambda x: x[0])
    for i, (_, im) in enumerate(tqdm(images_sorted)):
        image_path = colmap_image_dir/im.name
        im = cv2.imread(str(image_path))
        im = downsample(im)
        cv2.imwrite(str(image_output_dir/f'{i:06d}.png'), im[...,:3])
        dummy_mask = np.full(shape=im.shape[:2]+(3,), dtype=np.uint8, fill_value=255)
        cv2.imwrite(str(mask_output_dir/f'{i:06d}.png'), dummy_mask)
