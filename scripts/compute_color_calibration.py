# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
import argparse
import numpy as np
import cv2
from utils import hdri
from utils import color_calibration
from utils.constants import *


def main():

    parser = argparse.ArgumentParser(
        description="Script that computes the color conversion matrices from the images in the calibration directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--show_debug_images", action='store_true', help="If True show intermediate images for debugging")
    parser.add_argument("--theta_scale", type=float, default=1/100, help="Scale the values for the Theta Z1 camera")

    args = parser.parse_args()
    print(args)



    outdir = CALIBRATION_PATH/'color'

    eos_image_paths = list((outdir/'eos90d').glob('*.CR3'))
    theta_image_paths = {
        'front': list((outdir/'theta'/'front').glob('*.DNG')),
        'back': list((outdir/'theta'/'front').glob('*.DNG')),
    }

    theta_images = {}
    for front_or_back, paths in theta_image_paths.items():
        theta_images[front_or_back] = sorted([ hdri.read_raw_and_meta(x) for x in paths ], key=lambda x: x.exposure_time)

    theta_colors_front, dbg = color_calibration.extract_colorchecker_colors_from_raw(theta_images['front'])
    # show debug images to visually check if the detections make sense
    if args.show_debug_images:
        cv2.imshow('initial_detection', dbg['initial_detection'][...,[2,1,0]])
        cv2.waitKey()
        cv2.imshow('zoom', dbg['zoom'][...,[2,1,0]])
        cv2.waitKey()
        cv2.imshow('detection', dbg['detection'][...,[2,1,0]])
        cv2.waitKey()
        cv2.imshow('colors', dbg['colors'][...,[2,1,0]])
        cv2.waitKey()
        cv2.imshow('order', dbg['order'][...,[2,1,0]])
        cv2.waitKey()
        cv2.destroyAllWindows()

    theta_colors_back, dbg = color_calibration.extract_colorchecker_colors_from_raw(theta_images['back'])
    # show debug images to visually check if the detections make sense
    if args.show_debug_images:
        cv2.imshow('initial_detection', dbg['initial_detection'][...,[2,1,0]])
        cv2.waitKey()
        cv2.imshow('zoom', dbg['zoom'][...,[2,1,0]])
        cv2.waitKey()
        cv2.imshow('detection', dbg['detection'][...,[2,1,0]])
        cv2.waitKey()
        cv2.imshow('colors', dbg['colors'][...,[2,1,0]])
        cv2.waitKey()
        cv2.imshow('order', dbg['order'][...,[2,1,0]])
        cv2.waitKey()
        cv2.destroyAllWindows()

    theta_colors = 0.5*(theta_colors_front + theta_colors_back)
    theta_colors *= args.theta_scale
    np.savetxt(outdir/f'theta_to_theta_scaled.txt', np.eye(3)*args.theta_scale)

    
    eos_images = [ hdri.read_raw_and_meta(x) for x in eos_image_paths ]

    eos_iso_speeds = sorted(list(set([x.iso for x in eos_images])))
    eos_images_iso = {}
    for iso in eos_iso_speeds:
        eos_images_iso[iso] = sorted([x for x in eos_images if x.iso == iso], key=lambda x: x.exposure_time)

        eos_colors, dbg = color_calibration.extract_colorchecker_colors_from_raw(eos_images_iso[iso])
        if args.show_debug_images:
            cv2.imshow('initial_detection', dbg['initial_detection'][...,[2,1,0]])
            cv2.waitKey()
            cv2.imshow('zoom', dbg['zoom'][...,[2,1,0]])
            cv2.waitKey()
            cv2.imshow('detection', dbg['detection'][...,[2,1,0]])
            cv2.waitKey()
            cv2.imshow('colors', dbg['colors'][...,[2,1,0]])
            cv2.waitKey()
            cv2.imshow('order', dbg['order'][...,[2,1,0]])
            cv2.waitKey()
            cv2.destroyAllWindows()

        eos2theta = color_calibration.compute_color_transform(eos_colors, theta_colors)
        np.savetxt(outdir/f'eos90d_iso_{iso}_to_theta.txt', eos2theta)
    
if __name__ == '__main__':
    main()
