#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
import multiprocessing 
from PIL import ImageTk, Image
import tkinter
from tkinter import Tk, ttk, Canvas, DoubleVar
import numpy as np
from utils import hdri
from utils.constants import *
from utils.color_calibration import apply_color_transform, load_color_transforms


def tonemap(arr: np.ndarray, exposure: float):
    return Image.fromarray(hdri.simple_tonemap(arr, exposure, show_overexposed_areas=True))


def main():

    parser = argparse.ArgumentParser(
        description="Tool for visualizing the exposure setting for multiple images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument( 'env_dir', type=Path, help="Paths to the environment folder that contains all images.")

    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    print(args)

    exposure_path = args.env_dir/'exposure.txt'
    init_exposure = 0.0
    if exposure_path.exists():
        print('using exposure value from text file')
        init_exposure = np.loadtxt(exposure_path).item()


    color_transforms = load_color_transforms()

    hdr_images = []
    dirs = sorted(list(args.env_dir.iterdir()))
    step = 8 # subsampling
    # load test images
    for test_dir in dirs:
        if test_dir.is_dir() and test_dir.name.startswith('test') and not 'env' in test_dir.name:
            im_paths = list(test_dir.glob('*.CR3'))
            raws = [hdri.read_raw_and_meta(x) for x in im_paths]
            im = hdri.compute_hdri_from_raws(raws)
            im = im[::step,::step,:].copy()
            iso = raws[0].iso
            ct = color_transforms[f'eos90d_iso_{iso}_to_theta']
            hdr_images.append(apply_color_transform(im, ct))
            print('.', end='', flush=True)
    num_test_images = len(hdr_images)

    # load some images from the image dir
    im_paths = [x[0] for x in np.array_split(sorted(list((args.env_dir/'images').glob('*.CR3'))), 3)]
    im_paths.append(sorted(list((args.env_dir/'images').glob('*.CR3')))[-1])
    for im_path in im_paths:
        raw = hdri.read_raw_and_meta(im_path)
        im = hdri.compute_hdri_from_raws([raw])
        im = im[::step,::step,:].copy()
        iso = raws[0].iso
        ct = color_transforms[f'eos90d_iso_{iso}_to_theta']
        hdr_images.append(apply_color_transform(im, ct))
        print('.', end='', flush=True)
    print('\n')

    root = Tk()
    root.title('Exposure visualization')
    root.geometry('1920x1080')
    root.grid()

    root.grid_rowconfigure(1, weight=1)
    root.grid_columnconfigure(0, weight=1)

    label = ttk.Label(root, text='Exposure')
    label.grid(row=0, column=0, sticky="E")
    exposure = DoubleVar(value=init_exposure)
    spinbox = ttk.Spinbox(root, from_=-20, to=20, textvariable=exposure)
    spinbox.grid(row=0, column=1, sticky="W")

    canvas = Canvas(root, background='grey')
    canvas.grid(row=1, column=0, columnspan=2, sticky="NWSE")
    
    canvas_image_ids = []
    tk_photos = []
    x = 0
    y = 0
    for i, hdr_im in enumerate(hdr_images):
        im8bit = tonemap(hdr_im,init_exposure)
        photo = ImageTk.PhotoImage(image=im8bit)
        tk_photos.append(photo)
        canvas_image_ids.append(canvas.create_image(x,y,anchor='nw',image=photo))
        x += hdr_im.shape[1]
        if i == num_test_images-1:
            x = 0
            y += hdr_im.shape[0]
    
    def update_images(*args):
        try:
            value = float(spinbox.get())
            for i, (hdr_im, im_id) in enumerate(zip(hdr_images, canvas_image_ids)):
                im8bit = tonemap(hdr_im,value)
                photo = ImageTk.PhotoImage(image=im8bit)
                tk_photos[i] = photo
                canvas.itemconfig(im_id, image=photo)
        except:
            pass
            

    exposure.trace('w', update_images)
    
    root.mainloop()

    with open(exposure_path, 'w') as f:
        f.write(str(exposure.get()))

    
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()
