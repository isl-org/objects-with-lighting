# SPDX-License-Identifier: Apache-2.0
import numpy as np
from pathlib import Path
from addict import Dict as ADict
import exifread
import rawpy
import cv2
from collections import namedtuple
from .filesystem import *
from typing import List
from scipy.interpolate import griddata
from scipy.ndimage import binary_dilation
import imageio
import re

RawMeta = namedtuple('RawMeta', ['raw', 'exif',  'exposure_time', 'fnumber', 'iso', 'ndfilter'])


def read_hdri(path: Path) -> np.ndarray:
    """Reads .exr or .hdr files and returns a numpy array. The channel order is RGB."""
    imageio.plugins.freeimage.download()
    return imageio.imread(path)


def write_hdri(arr: np.ndarray, path: Path) -> None:
    """Writes a color image as numpy array to a .exr or .hdr file.
    Args:
        arr: Array with dtype np.float32 and shape (h,w,3). The channel order is RGB.
        path: Output path.
    """
    path = Path(path)
    assert path.suffix.lower() in ('.hdr', '.exr')
    assert arr.ndim == 3 and arr.shape[-1] == 3
    imageio.plugins.freeimage.download()
    imageio.imwrite(path, arr.astype(np.float32))


def read_nd_filter_data(path: Path):
    """Reads the ND-filter metadata that is stored in separate files and returns it if the file is affected by the filter.
    Args:
        path: Path to the .DNG file.
    Returns:
        The f-number for the ND-filter and the stem of the images for which the filter applies.
        Returns None if there is no ND-filter information.
    """
    ndfilter_path = (path.parent/'nd-filter').resolve()

    ndfilter = None
    if ndfilter_path in read_nd_filter_data.cache:
        ndfilter = read_nd_filter_data.cache[ndfilter_path]
    elif ndfilter_path.exists():
        with open(ndfilter_path,'r') as f:
            data = f.read()
        data = data.split(' ')
        fstop = float(data[0])
        first, last = data[1].split('-')
        pattern = re.compile('([^\d]*)(\d+)')
        prefix, first_number = pattern.match(first).groups()
        num_digits = len(first_number)
        first_number = int(first_number)
        last_number = int(pattern.match(last).group(2))
        images_with_nd_filter = set([prefix+f'{{:0{num_digits}d}}'.format(x) for x in range(first_number,last_number+1)])
        ndfilter = ADict()
        ndfilter.fstop = fstop
        ndfilter.image_set = images_with_nd_filter
        mask_path = path.parent/'nd-filter_images_mask.png'
        if mask_path.exists():
            ndfilter.mask = cv2.imread(str(mask_path))
            ndfilter.mask_half = cv2.resize(ndfilter.mask, (ndfilter.mask.shape[1]//2, ndfilter.mask.shape[0]//2))
            ndfilter.mask = ndfilter.mask > 127
            ndfilter.mask_half = ndfilter.mask_half > 127
        else:
            ndfilter.mask = None
            ndfilter.mask_half = None
        read_nd_filter_data.cache[ndfilter_path] = ndfilter

    if ndfilter and path.stem in ndfilter.image_set:
        return ndfilter
    else:
        return None
read_nd_filter_data.cache = {}


def process_image_with_nd_filter_for_merging(im: np.ndarray, ndfilter):
    """Preprocesses the images that were taken with an ndfilter.
    This function denoises the image and applies a soft mask.
    Args:
        im: The image as returned by process_raw()
        ndfilter: The ndfilter information as returned by read_raw_and_meta()
    Returns:
        A denoised and masked version of the input image and the soft mask used.
    """
    # denoise
    # im = cv2.medianBlur(im, 3)
    # im = cv2.GaussianBlur(im, None, 0.5)
    im = cv2.erode(im, kernel=None)

    # apply soft mask
    dtype=im.dtype
    dtype_info = np.iinfo(dtype)
    mask = ndfilter.mask_half.astype(np.float32)
    soft_mask = cv2.GaussianBlur(mask, None, 10)
    im = im.astype(np.float64)*soft_mask
    im = np.clip(np.round(im), 0, dtype_info.max).astype(dtype)
    if dtype_info.max == 255:
        im[im <= 4] = 0
    elif dtype_info.max == 2**16-1:
        im[im <= 255] = 0

    # use only the green channel to avoid color shifts caused by the filter
    im[...,0] = im[...,1]
    im[...,2] = im[...,1]
    
    return im, soft_mask


def read_raw_and_meta(path: Path, meta_only: bool=False):
    """Reads a raw image and metadata
    
    Args:
        path: path to the raw file
        meta_only: If True read only the EXIF metadata.
    
    Returns:
        A dict with keys 'raw', 'exif', 'exposure_time', 'fnumber', and 'iso'. 
        The exif information may be read from a JPG file with the same name.
    """
    result = ADict()
    path = Path(path)
    if path.suffix == '.DNG':
        exif_path = path
    elif path.with_suffix('.JPG').exists():
        exif_path = path.with_suffix('.JPG')
    else:
        exif_path = None

    if exif_path:
        with open(exif_path, 'rb') as f:
            tags = exifread.process_file(f)
            result.exif = tags
        # extract some important tags
        v = tags['EXIF ExposureTime'].values[0]
        result.exposure_time = float(v)
        v = tags['EXIF FNumber'].values[0]
        result.fnumber = float(v)
        v = tags['EXIF ISOSpeedRatings'].values[0]
        result.iso = int(v)

    result.raw = None
    if not meta_only:
        result.raw = rawpy.imread(str(path))  
    
    result.ndfilter = read_nd_filter_data(path)
    
    return RawMeta(**result)


def extract_valid_square_from_dfe(dfe, edge_length=0.66):
    """Extracts two square images from the double fisheye image
    
    Args:
        dfe (np.ndarray): Double fisheye image
        edge_length: Edge length of the squares as a factor of the image height. 
        
    Returns:
        The returned arrays may be views of the original array.
    """
    h, w = dfe.shape[:2]
    assert 2*h==w
    s_2 = int(edge_length*h/2)
    center_left = (h//2, w//4)
    center_right = (h//2, 3*(w//4))
    return {'left': dfe[center_left[0]-s_2:center_left[0]+s_2,center_left[1]-s_2:center_left[1]+s_2],
            'right': dfe[center_right[0]-s_2:center_right[0]+s_2,center_right[1]-s_2:center_right[1]+s_2]}

    
def process_raw(raw, apply_cam_white_balance=True, dtype=np.uint8):
    """Processes a raw image to a RGB numpy array
    Args:
        raw: A RawPy object.
        apply_cam_white_balance: If True applies the white balance from the camera.
        dtype: The output dtype. Either uint8 or uint16.
        
    Returns:
        A numpy array with the processed image with shape [h,w,3].
        The color format is RGB.
        
    """
    assert dtype in (np.uint8, np.uint16)
    assert raw.raw_pattern.shape == (2,2)
    assert raw.raw_image_visible.dtype == np.uint16

    dtype_info = np.iinfo(dtype)
    
    red_idx = [i for i, x in enumerate(raw.color_desc) if x == ord(b'R')]
    green_idx = [i for i, x in enumerate(raw.color_desc) if x == ord(b'G')]
    blue_idx = [i for i, x in enumerate(raw.color_desc) if x == ord(b'B')]

    channels = []
    for idx in range(raw.raw_pattern.size):
        i, j = np.where(raw.raw_pattern == idx)
        channels.append(raw.raw_image_visible[i.item()::2,j.item()::2])
    img = np.stack(channels)

    if not raw.camera_white_level_per_channel is None:
        white_level = raw.camera_white_level_per_channel
    else:
        white_level = [raw.white_level]*len(channels)
    
    for i, (ch, black, white) in enumerate(zip(img,raw.black_level_per_channel, white_level)):
        np.clip(ch, black, white, out=ch)
        ch -= black

    new_white_level = np.array(white_level) - np.array(raw.black_level_per_channel, dtype=np.float64)
    img = img.astype(np.float64)

    if apply_cam_white_balance:
        white_balance = np.array(raw.camera_whitebalance)
        if len(green_idx) == 2:
            if white_balance[green_idx[1]] == 0.0:
                white_balance[green_idx[1]] = white_balance[green_idx[0]]
            assert white_balance[green_idx[0]] == white_balance[green_idx[1]]
            
        white_balance /= white_balance[green_idx[0]]
        img *= white_balance[...,None,None]

    for ch, white in zip(img, new_white_level):
        ch *= dtype_info.max/white

    reds = [img[i] for i in red_idx]
    greens = [img[i] for i in green_idx]
    blues = [img[i] for i in blue_idx]

    red = np.mean(reds, axis=0)
    green = np.mean(greens, axis=0)
    blue = np.mean(blues, axis=0)

    img = np.stack([red,green,blue])
    img = np.round(img)
    
    img = np.clip(img,None, dtype_info.max)
    return img.transpose([1,2,0]).astype(dtype)


def robertson_weights(ldr_size=256):
    """Weights for merging LDR images
    See  Mark A Robertson, Sean Borman, and Robert L Stevenson. Dynamic range improvement through multiple exposures. In Image Processing, 1999. ICIP 99. Proceedings. 1999 International Conference on, volume 3, pages 159–163. IEEE, 1999.
    
    Args:
        ldr_size: The size for the returned lookup table.
        
    Returns:
        A LUT with weights for merging LDR images to an HDRI.
    """
    q = (ldr_size-1) / 4
    e4 = np.exp(4)
    scale = e4/(e4-1)
    shift = 1/(1-e4)
    
    i = np.arange(ldr_size)
    value = i/q-2
    value = scale*np.exp(-value**2) + shift
    return np.clip(value,0,1)


def merge_robertson(images, times, response=None, masks=None):
    """Merge a set of LDR images to an HDRI.
    See  Mark A Robertson, Sean Borman, and Robert L Stevenson. Dynamic range improvement through multiple exposures. In Image Processing, 1999. ICIP 99. Proceedings. 1999 International Conference on, volume 3, pages 159–163. IEEE, 1999.
    
    Args:
        images: A list of numpy arrays with images. All images must have the same shape and dtype.
            dtype must be either uint8 or uint16
        times: A list of exposure times.
        response: The camera response function. If None a linear response is used.
        masks: A spatial mask with the same shape as the images. The range of the values must be n [0..1].
    
    Return:
        The merged HDRI and the mask of valid values
    """
    assert len(images) == len(times)
    if masks is not None:
        assert len(masks) == len(images)
    else:
        masks = len(images)*[None]

    dtype_info = np.iinfo(images[0].dtype)
    ldr_size = dtype_info.max+1

    # sort with respect to times
    images, times, masks = zip(*sorted(zip(images, times, masks), key=lambda x: x[1]))

    if response is None:
        response = 2*np.arange(ldr_size, dtype=np.float64)/ldr_size
    assert response.shape[0] == ldr_size
    
    result = np.zeros_like(images[0], dtype=np.float64)
    wsum = np.zeros_like(images[0], dtype=np.float64)
    
    if len(images)==1:
        # Special case for just 1 image. This is for compatibility.
        weights = np.ones((ldr_size,),dtype=np.float64)
    else:
        weights = robertson_weights(ldr_size)
    
    # dbg = {(1928//2,1272//2):[], (1000,670):[], (970,650):[]}
    dbg = {}
    for img, time, mask in zip(images, times, masks):
        assert img.dtype in (np.uint8, np.uint16) and img.dtype == images[0].dtype
        w = weights[img]
        if mask is not None:
            w *= mask
        im = response[img]
        result += time*w*im
        wsum += time**2 * w
        
        for (c,r), l in dbg.items():
            l.append(f'{c},{r} time {time:>25}, w {w[r,c,1]:>25}, im {im[r,c,1]:>25}, im/t {im[r,c,1]/time}')
    
    return result*(1/wsum), wsum == 0


def fill_nonfinite_nearest(im):
    """Fills all nan values with the spatially nearest valid value. This method manipulates the input array"""
    assert im.ndim == 3
    xy = np.stack(np.meshgrid(np.arange(im.shape[1]), np.arange(im.shape[0])), axis=-1)
    mask = np.isfinite(im)
    
    for c in range(im.shape[-1]):
        valid_xy = xy[mask[...,c]]
        invalid_xy = xy[~mask[...,c]]
        valid_values = im[...,c][mask[...,c]]
        nearest_values = griddata(valid_xy.reshape(-1,2), valid_values.flat, invalid_xy.reshape(-1,2), method='nearest')
        im[...,c][~mask[...,c]] = nearest_values


def smooth_filled_in_values(im, mask):
    """Smooth filled in values using squared L2 regularization
    Args:
        im: Float image with shape (h,w,3).
        mask: boolean mask with the pixels that should be smoothed set to True. The shape is (h,w,3).
        
    Returns:
        The smoothed image as numpy array and the loss values for each iteration.
    """
    import mitsuba as mi
    import drjit as dr
    mi.set_variant('llvm_ad_rgb')

    if np.count_nonzero(mask) == 0:
        return im
    
    opt = mi.ad.SGD(lr=0.1)
    mask2d = mask.any(axis=-1)
    masky = mask2d.any(axis=1)
    row_start = np.argmax(masky)
    row_end = masky.shape[0] - np.argmax(masky[::-1])
    row_start = max(0,row_start-1)
    row_end = min(masky.shape[0], row_end+1)

    maskx = mask2d.any(axis=0)
    col_start = np.argmax(maskx)
    col_end = maskx.shape[0] - np.argmax(maskx[::-1])
    col_start = max(0,col_start-1)
    col_end = min(maskx.shape[0], col_end+1)
    
    im_crop = np.ascontiguousarray(im[row_start:row_end,col_start:col_end,...])
    mask_crop = np.ascontiguousarray(mask[row_start:row_end,col_start:col_end,...])

    im_crop = np.nan_to_num(np.log(im_crop),nan=-20)
    
    x0 = mi.TensorXf(im_crop)
    m = mi.TensorXf(mask_crop.astype(np.float32))

    opt['x'] = x0
    loss_curve = np.empty((1000,))
    for i in range(len(loss_curve)):
        x = opt['x']
        dx = x[:-1,1:] - x[:-1,:-1]
        dy = x[1:,:-1] - x[:-1,:-1]
        reg = dx**2 + dy**2

        loss = dr.mean(reg*m[:-1,:-1])
        loss_curve[i] = loss[0]

        dr.backward(reg)
        opt.step()
        opt['x'] = dr.select(m>0, opt['x'], x0)

    ans = im.copy()
    ans[row_start:row_end,col_start:col_end,...] = np.exp(opt['x'].numpy())
    return ans.astype(im.dtype), loss_curve


def compute_hdri_from_raws(raws, apply_cam_white_balance=True):
    """Merge multiple raw images into an HDRI.
    Args:
        raws: A list of RawMeta objects.
        apply_cam_white_balance: If True applies the while balance from the camera.
        
    Returns:
        A numpy array with the HDRI with shape [h,w,3].
        The color format is RGB. 
    """
    images = []
    times = []
    ndfilter_soft_mask = None
    for x in raws:
        im = process_raw(x.raw, apply_cam_white_balance=apply_cam_white_balance, dtype=np.uint16)
        time = x.exposure_time
        if x.ndfilter is not None:
            time *= (1/2)**x.ndfilter.fstop
            im, ndfilter_soft_mask = process_image_with_nd_filter_for_merging(im, x.ndfilter)
            # skip images with ndfilter that have overexposed pixels
            if im.max() == np.iinfo(im.dtype).max:
                continue
        images.append(im)
        times.append(time)

    hdri, mask = merge_robertson(images, times)

    if np.count_nonzero(~np.isfinite(hdri)):
        fill_nonfinite_nearest(hdri)

    # special treatment for sequences with ndfilter with masks:
    if ndfilter_soft_mask is not None:
        mask = mask.any(axis=-1)
        mask = binary_dilation(mask)
        mask = np.tile(mask[...,None], [1,1,3])
        hdri, _ = smooth_filled_in_values(hdri, mask)

    return hdri
    

def compute_hdri_from_tiffs(tiffs: List[Path], apply_cam_white_balance=True):
    """Merge multiple tiff images into an HDRI.

    This function is used for the equirectangular images generated by the stitcher.

    Args:
        tiffs: A list of paths to the tiff images. For each image we assume that there is a .DNG file with the metadata.
        apply_cam_white_balance: If True applies the while balance from the camera.
        
    Returns:
        A numpy array with the HDRI with shape [h,w,3].
        The color format is RGB. 
    """
    images = []
    times = []
    for tiff_path in tiffs:
        raw = read_raw_and_meta(tiff_path.with_suffix('.DNG'), meta_only=True)
        times.append(raw.exposure_time)
        images.append(cv2.imread(str(tiff_path))[...,[2,1,0]])
    hdri, _ = merge_robertson(images, times)
    return hdri
    

def compute_ldr_from_hdri_opencv(hdri: np.ndarray, clip=(0,200)):
    """Computes an 8bit tonemapped image from an HDRI with OpenCV.
    This image can be used for detecting apriltags but it not meant to be used for the final dataset images.

    Args:
        hdri: Input hdr image
        clip: clip values for clipping the range before tonemapping. This is used because the tonemap algorithm has problems with very large values.

    Returns:
        LDR image after clipping and tonemapping.
    """
    hdri = np.clip(hdri, 0, 200)
    tonemap = cv2.createTonemapReinhard()
    ldr = (tonemap.process(hdri.astype(np.float32))*255).astype(np.uint8)
    return ldr

    
def compute_response_function_from_raw(raws):
    """Computes the camera response function from raw images.
    This function is used to check if the raw processing and reading/interpretation of exif data is sane.
    
    Args:
        raws: A list of RawMeta objects.
        
    Returns:
        The camera response function.
    """
    images, times = zip(*[(process_raw(x.raw), x.exposure_time) for x in raws])
    times = np.array(times, dtype=np.float32) # conversion to float32 is important here!
    calibrate = cv2.createCalibrateRobertson()
    response = calibrate.process(images, times)
    return response


def create_data_for_theta_stitcher(dng_paths, outdir):
    """Creates a directory that can be batch processed with the RICOH Theta Stitcher software.
    
    Args:
        dng_paths: A list of paths to the DNG files
        outdir: The output directory
        
    Returns:
        A list of paths with the corresponding tiff files.
    """
    import shutil
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    result = []
    for i, p in enumerate(dng_paths):        
        p = Path(p)
        dng = read_raw_and_meta(p)
        img = process_raw(dng.raw, True, dtype=np.uint16)
        # dst_shape = np.array(img.shape[:2])*2
        # img2 = cv2.resize(img, dst_shape[::-1]) # this alters the maximum intensities?
        img2 = np.kron(img, np.ones(dtype=np.uint16,shape=(2,2,1)))
        tiff_path = outdir/f'{p.stem}.tiff'
        cv2.imwrite(str(tiff_path), img2[...,[2,1,0]])
        out_path = outdir/p.name
        copyfile(p, out_path, overwrite=True)
        result.append(tiff_path)
    return result
    

def simple_downsample(arr):
    """Simple downsample function averaging a block of 2x2 pixels."""
    assert arr.dtype in (np.float32, np.float64)
    return 0.25*(arr[0::2,0::2,...] + arr[0::2,1::2,...] + arr[1::2,0::2,...] + arr[1::2,1::2,...])


def apply_exposure_and_gamma(arr: np.ndarray, exposure: float, gamma: float):
    """Applies exposure and gamma to an array.
    Args:
        arr: Image as numpy array
        exposure: The exposure value
        gamma: Gamma value
    """
    assert arr.dtype in (np.float32, np.float64)
    return np.power(np.clip(arr*2**exposure, 0, None), 1/gamma)

def convert_to_8bit(arr):
    """Converts a float array to a uint8 array by clipping the range and rounding to the nearest integer in [0,255]."""
    return np.clip(np.round(arr*255),0,255).astype(np.uint8)

def simple_tonemap(arr, exposure, show_overexposed_areas=False):
    """Applies the exposure change and gamma correction with gamma 2.2. Returns an 8 bit array."""
    if show_overexposed_areas:
        im = np.round(255*apply_exposure_and_gamma(arr, exposure, gamma=2.2))
        overexposed = np.any(im>255, axis=-1)
        im[overexposed,:] = (128,0,128) # purple
        return im.astype(np.uint8)
    return convert_to_8bit(apply_exposure_and_gamma(arr, exposure, gamma=2.2))