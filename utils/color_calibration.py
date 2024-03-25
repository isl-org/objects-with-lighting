# SPDX-License-Identifier: Apache-2.0
import numpy as np
from addict import Dict as ADict
from utils.hdri import *
from utils.constants import *

def get_colorchecker_squares(box, img):
    """Returns the center of the 24 individual color squares
    Args:
        box: The box as returned by cv2.mcc.ColorChecker.getBox()
        img: The corresponding image
        
    Returns:
        An array of points, the sizes of the squares in pixels and the average color.
    """
    A,B,C,D = box

    us = np.linspace(0,1,7)
    vs = np.linspace(0,1,5)

    us += us[1]/2
    vs += vs[1]/2

    def interp(u,v):
        return (1-v)*((1-u)*A + u*B) + v*((1-u)*D + u*C)
    
    points = []
    sizes = []
    colors = []
    for u in us[:-1]:
        for v in vs[:-1]:
            p = interp(u,v)
            p2 = interp(u+us[1]/2, v+vs[1]/2)
            size = np.abs(p-p2)
            safe_radius = size/4
            sq_min = np.round(p-safe_radius).astype(np.int32)
            sq_max = np.round(p+safe_radius).astype(np.int32)
            color = np.mean(img[sq_min[1]:sq_max[1],sq_min[0]:sq_max[0],...], axis=(0,1))
            
            points.append(p)
            sizes.append(size)
            colors.append(color)
            
    return np.stack(points), np.stack(sizes), np.stack(colors)


def extract_colorchecker_colors_from_raw(raws):
    """Extracts the 24 colors from the colorchecker
    Args:
       raws: A list of RawMeta objects read with hdri.read_raw_meta().
    
    Returns:
        An array with the 24 color values of the color checker.
        A dictionary with debug visualizations.
    """
    import cv2
    hdri = compute_hdri_from_raws(raws)
        
    debug = True
    dbg = {}
    
    def fn(im):
        tonemap = cv2.createTonemapReinhard()
        ldr = (tonemap.process(im.astype(np.float32))*255).astype(np.uint8)
        ldr_bgr = np.ascontiguousarray(ldr[...,[2,1,0]]) # RGB -> BGR

        # use the ldr for detection
        detector = cv2.mcc.CCheckerDetector.create()
        status = detector.process(ldr_bgr, cv2.mcc.MCC24)
        if not status:
            return
        
        # "zoom in". This helps to improve the detection for the fisheye images.
        cc = detector.getBestColorChecker()

        if debug:
            dbg_initial_detection = ldr.copy()
            draw = cv2.mcc.CCheckerDraw.create(cc, thickness=5)
            draw.draw(dbg_initial_detection)
            dbg['initial_detection'] = dbg_initial_detection
        roi_min = cc.getBox().min(axis=0)
        roi_max = cc.getBox().max(axis=0)
        roi_size = roi_max-roi_min
        m = 0.20 # margin for the zoom in
        roi_min = np.clip(np.round(roi_min-m*roi_size).astype(int), 0, None)
        roi_max = np.clip(np.round(roi_max+m*roi_size).astype(int), 0, np.array(im.shape[:2])[::-1]-1)        
        
        im = np.ascontiguousarray(im[roi_min[1]:roi_max[1],roi_min[0]:roi_max[0],...])
        ldr = (tonemap.process(im.astype(np.float32))*255).astype(np.uint8)

        dbg['roi_min'] = roi_min
        dbg['roi_max'] = roi_max
        dbg['zoom'] = ldr.copy()
        ldr_bgr = np.ascontiguousarray(ldr[...,[2,1,0]]) # RGB -> BGR

        # use the ldr for detection
        detector = cv2.mcc.CCheckerDetector.create()
        status = detector.process(ldr_bgr, cv2.mcc.MCC24)
        if not status:
            return

        cc = detector.getBestColorChecker()
        draw = cv2.mcc.CCheckerDraw.create(cc, thickness=5)

        dbg_img = ldr.copy()
        draw.draw(dbg_img)
        dbg['detection'] = dbg_img

        # for debugging
        if debug:
            points, sizes, colors = get_colorchecker_squares(cc.getBox(), ldr)
            dbg_colors = ldr.copy()
            for p, s, c in zip(points, sizes, colors):
                cv2.drawMarker(dbg_colors, p.astype(int), tuple(map(int,c)), markerSize=int(s[0]), thickness=3)
            dbg['colors'] = dbg_colors

            dbg_order = ldr.copy()
            for p, c in zip(cc.getBox(),[(0,0,0),(85,85,85),(170,170,170),(255,255,255)]):
                cv2.drawMarker(dbg_order, p.astype(int), c, markerSize=int(s[0]), thickness=3)
            dbg['order'] = dbg_order

        points, sizes, colors = get_colorchecker_squares(cc.getBox(), im)
        return colors
        
    
    if 'THETA' in raws[0].exif['Image Model'].printable:
        squares = extract_valid_square_from_dfe(hdri)
        for side, im in squares.items():
            colors = fn(im)
            if colors is not None:
                return colors, dbg
    else:
        colors = fn(hdri)
        if colors is not None:
            return colors, dbg

    return None, dbg


def compute_color_transform(src_colors, dst_colors):
    """Computes a 3x3 color transformation matrix that maps from the src to the dst colors.
    
    Args:
        src_colors: An array of shape (N,3) with the source colors.
        dst_colors: An array of shape (N,3) with the corresponding destination colors.
    
    Returns:
        A 3x3 color transformation matrix C. (Least squares solution to src_colors * C = dst_colors)
    """
    return np.linalg.lstsq(src_colors, dst_colors, rcond=None,)[0]


def apply_color_transform(arr: np.ndarray, color_transform: np.ndarray):
    """Applies a 3x3 color transformation matrix to the image."""
    assert arr.ndim == 3
    assert arr.shape[-1] == 3
    assert color_transform.shape == (3,3)
    assert arr.dtype in (np.float32, np.float64)
    assert color_transform.dtype in (np.float32, np.float64)
    return arr @ color_transform


def load_color_transforms():
    """Load all color transforms and return them as a dict"""
    result = ADict()
    for path in (CALIBRATION_PATH/'color').glob('eos*.txt'):
        result[path.stem] = np.loadtxt(path)
    return result
