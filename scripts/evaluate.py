# SPDX-License-Identifier: Apache-2.0
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
print(sys.path)
import re
import numpy as np
from scipy.optimize import minimize_scalar, minimize
import argparse
import cv2
import json
from typing import Any, Dict, Set
from utils.metrics import METRICS
from utils.hdri import read_hdri, convert_to_8bit, apply_exposure_and_gamma
from utils.constants import DATASET_PATH, DEFAULT_GAMMA
from utils.tabelle import *


def read_image(p: Path):
    p = Path(p)
    if p.suffix == '.png':
        im = cv2.imread(str(p))[...,[2,1,0]] # convert to RGB
        im = np.clip(im.astype(np.float32)/255, 0, 1)
    elif p.suffix == '.npy':
        im = np.load(p)
        assert im.dtype in (np.float32, np.float64), f'npy array must have float dtype but is {im.dtype}'
        im = im.astype(np.float32)
    elif p.suffix == '.exr':
        im = read_hdri(p)[...,:3]
    else:
        raise Exception(f'Unsupported image format for file {str(p)}')
    return im

def read_mask(p: Path):
    im = cv2.imread(str(p))[...,1:2]
    im = np.clip(im.astype(np.float32)/255, 0, 1)
    return im


def fix_nonfinite(im):
    if np.count_nonzero(~np.isfinite(im)):
        print('  fixing nonfinite values!')
        return np.nan_to_num(im, posinf=0.0, neginf=0.0)
    else:
        return im

def resize_prediction(pr, gt):
    """Resizes the prediction to match the gt image size if necessary
    Args:
        gt: ground truth image as np.ndarray with shape (H,W,3).
        pr: prediction in the same format as the ground truth image.
    Returns:
        prediction with the same shape as the gorund truth
    """
    if pr.shape == gt.shape:
        return pr
    assert pr.shape[-1] == gt.shape[-1] and pr.ndim == 3, f'pr.shape {pr.shape}, gt.shape {gt.shape}'
    print(f'  resizing prediction {pr.shape[:2][::-1]} -> {gt.shape[:2][::-1]}')
    wh = (gt.shape[1], gt.shape[0])
    return cv2.resize(pr, wh, cv2.INTER_AREA)


def scale_and_tonemap_prediction_image(gt, mask, pr=None, pr_linear=None, balance_colors=True, autoexposure=True):
    """Scales the prediction to minimize the mean error in linear space.
    
    Args:
        gt: ground truth image as np.ndarray with shape (H,W,3), dtype float32 and range [0..1].
        mask: mask image as np.ndarray with shape (H,W,1), dtype float32 and values 0 or 1.
        pr: prediction in the same format as the ground truth image.
        pr_linear: linear prediction image before tonemapping.
        balance_colors: If True scale the color channels individually to reduce the MSE
        autoexposure: If True comput and apply the optimal exposure that reduces the MSE

    Returns:
        The tonemapped prediction with range [0..1] after applying the optimal scale.
    """
    assert (pr is None) ^ (pr_linear is None)

    gamma = DEFAULT_GAMMA
    if pr is not None:
        pr_linear = np.power(pr, gamma)

    # exclude overexposed values
    mask = (gt.max(axis=-1, keepdims=True) < 1).astype(np.float32)*mask
    mask = mask[...,0] > 0.9
    
    def fn(x):
        return np.abs(np.clip(apply_exposure_and_gamma(pr_linear[mask], exposure=x, gamma=gamma),0,1) - gt[mask]).mean()

    optimal_exposure = 0.0
    if autoexposure:
        # define upper search bound
        smallest_linear_value = pr_linear[mask].min(axis=-1)
        smallest_linear_value = smallest_linear_value[smallest_linear_value > 0].min()
        max_exposure = np.log2(0.95**gamma / smallest_linear_value)
        
        res = minimize_scalar(fn, (-10,max_exposure))
        optimal_exposure = res.x
    # print('optimal exposure', res.x)
    
    if balance_colors:
        def fn(x):
            tmp = pr_linear[mask].reshape(-1,3) * x[None,:]
            return np.abs(np.clip(apply_exposure_and_gamma(tmp, exposure=optimal_exposure, gamma=gamma),0,1) - gt[mask]).mean()

        res = minimize(fn, np.ones((3,), dtype=np.float32), bounds=3*[(0.5,2.0)])
        optimal_color_balance = res.x.astype(np.float32)
        pr_linear = pr_linear * optimal_color_balance[None,None,:]
        # print('color balance', tuple(optimal_color_balance))

    result = np.clip(apply_exposure_and_gamma(pr_linear, exposure=optimal_exposure, gamma=gamma), 0, 1)
    return result


def evaluate(gt: Path, pr: Path, output: Dict[str, Dict[str,Any]], subsets: Set[str], save_tonemapped=False, save_comparison=False, balance_colors=True, force_tonemapping=False, autoexposure=True):
    """Compute the metrics for the predictions.
    This function recursively compares the ground truth images with the corresponding predictions.

    Args:
        gt: Path to the dataset directory with the ground truth
        pr: Path to the directory with the predictions
        output: The output dictionary
        subsets: The subsets for which to evaluate. This is usually {'test'}.
        save_tonemapped: Save the tonemapped image to the disk
        save_comparison: Save an image for easy comparison to the ground truth to the disk
        balance_colors: Apply color balancing to the images to reduce the MSE
        force_tonemapping: If True redo the tonemapping for PNG images.
        autoexposure: If True apply the optimal exposure for linear images. Will not affect PNG images unless force_tonemapping is True.

    """
    prediction_extensions = [
        '.npy', # assume linear image
        '.exr', # assume linear image
        '.png', # assume tonemapped image
    ]

    gt_images = sorted(list(gt.glob('**/gt_image*.png')))
    for gt_im_path in gt_images:
        if subset:
            env_name = gt_im_path.parent.name
            if env_name not in subsets:
                continue
            obj_name = gt_im_path.parent.parent.name
            key = f'{obj_name}/{env_name}/{gt_im_path.name}'
            pr_im_path = pr/obj_name/env_name/gt_im_path.name.replace('gt_','pr_')
        else:
            env_name = None
            obj_name = gt_im_path.parent.name
            key = f'{obj_name}/{gt_im_path.name}'
            pr_im_path = pr/obj_name/gt_im_path.name.replace('gt_','pr_')
        mask_path = gt_im_path.parent/gt_im_path.name.replace('image', 'mask')

        if not mask_path.exists():
            print(f'skipping {str(gt_im_path)} because the mask file does not exist.')
            continue
        
        result = {}
        
        for ext in prediction_extensions:
            pr_im_path = pr_im_path.with_suffix(ext)
            if pr_im_path.exists():
                cached_output = output.get(key)
                update = True
                if cached_output:
                    mtime = os.path.getmtime(pr_im_path)
                    if cached_output['mtime'] == mtime and cached_output['prediction_path'] == str(pr_im_path):
                        update = False
                        result = cached_output
                        print('using cached result for ', gt_im_path, pr_im_path)
                if update:
                    mtime = os.path.getmtime(pr_im_path)
                    gt_im = read_image(gt_im_path)
                    pr_im = read_image(pr_im_path)
                    mask = read_mask(mask_path)
                    
                    print(f'\nevaluating d({pr_im_path}, {gt_im_path})')
                    pr_im = fix_nonfinite(resize_prediction(pr_im, gt_im))

                    if pr_im_path.suffix in ('.exr', '.npy'):
                        pr_im = scale_and_tonemap_prediction_image(gt=gt_im, mask=mask, pr_linear=pr_im, balance_colors=balance_colors, autoexposure=autoexposure)
                    if force_tonemapping and pr_im_path.suffix in ('.png',):
                        pr_im = scale_and_tonemap_prediction_image(gt=gt_im, mask=mask, pr=pr_im, balance_colors=balance_colors, autoexposure=autoexposure)
                    if save_tonemapped:
                        cv2.imwrite(str(pr_im_path.parent/(pr_im_path.name+'.tonemapped.png')), convert_to_8bit(pr_im)[...,[2,1,0]])

                    result['mtime'] = mtime
                    result['prediction_path'] = str(pr_im_path)
                    
                    if save_comparison:
                        col_mask = mask[...,0].any(axis=0)
                        start_col = np.argmax(col_mask)
                        end_col = col_mask.shape[0] - np.argmax(col_mask[::-1])
                        trimmed_pr = (mask*pr_im)[:,start_col:end_col,:]
                        trimmed_gt = (mask*gt_im)[:,start_col:end_col,:]
                        comparison_im = np.concatenate((pr_im, gt_im, trimmed_pr, trimmed_gt, np.sqrt(np.abs(trimmed_gt-trimmed_pr))),axis=1)
                        cv2.imwrite(str(pr_im_path.parent/(pr_im_path.name+'.comparison.png')), convert_to_8bit(comparison_im[...,[2,1,0]]))

                    for metric_name, metric in METRICS.items():
                        result[metric_name] = metric['fn'](prediction=pr_im, ground_truth=gt_im, mask=mask)
                        print(f'{metric_name}: {result[metric_name]:2.3f}', end='  ', flush=True)
                break # only evaluate prediction for the first extension found
        output[key] = result


def print_tables(data: Dict[str, Dict[str,Any]], subsets: Set[str], num_envs: int=3):
    """Print a table with the average metrics for each object and environment
    
    Args:
        data: This is the output of the evaluate function.
        subsets: The subsets for which to print a table. This is usually {'test'}.
        num_envs: The number of envs present in the test data. It is assumed that each env has the same number of test images.
    
    """
    objects = set()
    if subsets:
        for gt_path in data:
            objects.add(Path(gt_path).parent.parent.name)
    else:
        for gt_path in data:
            objects.add(Path(gt_path).parent.name)

    objects = sorted(list(objects))
    
    if not subsets:
        subsets = [None]
    for subset in subsets:
        table_values = np.full((len(objects)+1,(num_envs+1)*len(METRICS)), fill_value=np.nan)
        for obj_i, obj in enumerate(objects):
            if subset:
                keys = sorted([k for k in data if Path(k).parent == (Path(obj)/subset)])
            else:
                keys = sorted([k for k in data if Path(k).parent == Path(obj)])
            env_vals = np.array_split([data[k] for k in keys], num_envs)
            for metric_i, metric in enumerate(METRICS):
                allenv_vals = []
                for env_i, v in enumerate(env_vals):
                    table_values[obj_i, env_i*len(METRICS)+metric_i] = np.mean([x.get(metric,np.nan) for x in v])
                    allenv_vals.extend([x.get(metric,np.nan) for x in v])
                table_values[obj_i, num_envs*len(METRICS)+metric_i] = np.mean(allenv_vals)
                
        table_values[-1] = table_values[:-1].mean(axis=0)
                
        tab = Table((table_values.shape[0]+2, table_values.shape[1]+1))

        cols = []
        for i in list(range(num_envs))+['Mean']:
            cols.append(Cell( i if isinstance(i,str) else f'Env{i}', col_span=len(METRICS)))
            cols.extend((len(METRICS)-1)*[None])
        tab[0,1:] = cols
        tab[0,0].rowfmt.line = [(i+1,i+len(METRICS)) for i in range(0,(num_envs+1)*len(METRICS),len(METRICS))]

        tab[1,:] = ['Object']+(num_envs+1)*[v['latex'] for k, v in METRICS.items()]
        tab[1,0].rowfmt.line = True
        tab[-2,0].rowfmt.line = True
        # tab[0,0].colfmt.line = '|'
        tab[0,0].colfmt.align = 'l'
        for i in range(1,tab.shape[1]):
            tab[0,i].colfmt.num_format = '{:2.4f}'
            # tab[0,i].colfmt.auto_highlight = list(METRICS.values())[(i-1)%len(METRICS)]['best']
        tab[2:,0] = objects + ['Mean']
        tab[2:,1:] = table_values
        tab.set_topbottomrule(True)
        print('\n',subset)
        print(tab.str())




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script that computes the common error metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('output', type=Path, help="Path to the output json file")
    parser.add_argument('-g', '--ground_truth', type=Path, default=DATASET_PATH, help="Path to the dataset directory with the ground truth")
    parser.add_argument('-p', '--prediction', type=Path, required=True, help="Path to the directory with the predictions")
    parser.add_argument('--set', choices=set(['train', 'valid', 'test']), nargs="+", default=['test'], help="The subset to evaluate.")
    parser.add_argument('--name', type=str, help="A name for the method that is evaluated.")
    parser.add_argument('--save_tonemapped', action='store_true', help="If set the tonemapped images created from the linear images will be written to the prediction directory.")
    parser.add_argument('--save_comparison', action='store_true', help="If set an image with a side by side comparison will be written to the prediction directory.")
    parser.add_argument('--no_color_balance', action='store_true', help="If set disables color balancing. Color balancing reduces the mean error between the input image and the ground truth.")
    parser.add_argument('--force_tonemapping', action='store_true', help="If set redo the tonemapping for PNG images.")
    parser.add_argument('--disable_autoexposure', action='store_true', help="If set do not apply optimal exposure correction. This only affects .exr and .npy images. PNG images are affected too if --force_tonemapping is set.")
    parser.add_argument('--dataset', choices=set(['ours', 'synthetic4relight', 'dtu', 'bmvs']), default='ours', help="The dataset")
    parser.add_argument('--rm_empty', action='store_true', help="If set the json file will not store empty dictionaries for missing predictions.")

    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    print(args)

    output = {'data':{}}
    if args.name:
        output['name'] = args.name
    else:
        output['name'] = args.output.stem

    if args.dataset in ('synthetic4relight', 'synthetic4relight_nvs', 'dtu', 'bmvs'):
        subset = set()
    else:
        subset = args.set
    
    evaluate(gt=args.ground_truth, pr=args.prediction.resolve(), output=output['data'], subsets=args.set, save_tonemapped=args.save_tonemapped, save_comparison=args.save_comparison, balance_colors=not args.no_color_balance, force_tonemapping=args.force_tonemapping, autoexposure=not args.disable_autoexposure)

    num_envs = {
        'ours': 3,
        'synthetic4relight': 3,
        'dtu': 1,
        'bmvs': 1,
    }[args.dataset]
    print_tables(output['data'], subset, num_envs)

    if args.rm_empty:
        rm_data_keys = []
        for k,v in output['data'].items():
            if not v:
                rm_data_keys.append(k)
        for k in rm_data_keys:
            del output['data'][k]

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
