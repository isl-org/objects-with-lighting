#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
import numpy as np
from utils import hdri
from utils.constants import *
import pyvista
import pycolmap


def main():

    parser = argparse.ArgumentParser(
        description="Tool for defining the object bounding box. This script loads and updates the corresponding bound in the source_data dir!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument( 'env_dir', type=Path, help="Paths to the environment folder with the colmap reconstruction.")
    parser.add_argument( '--reset', action='store_true', help="If set start with the default bounding box.")

    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    print(args)
    env_dir = args.env_dir.resolve()
    object_name = env_dir.parent.name
    env_name = env_dir.name
    
    bounding_box_path = SOURCE_DATA_PATH/object_name/env_name/'object_bounding_box.txt'
    if bounding_box_path.exists() and not args.reset:
        print(f'loading {str(bounding_box_path)}')
        bounds = np.loadtxt(bounding_box_path)
    else:
        bounds = np.array((-0.15,0.15, -0.15,0.15, -0.02, 0.2))

    recon_dir = env_dir/'recon'/'0'
    recon = pycolmap.Reconstruction(recon_dir)
    

    points = []
    colors = []
    for k,v in recon.points3D.items():
        points.append(v.xyz)
        colors.append(v.color)
    points = np.stack(points)
    colors = np.stack(colors)

    polydata = pyvista.PolyData(points)
    polydata['colors'] = colors
    # roughly remove points that are clearly outside of the object bounding box
    polydata = polydata.clip_box(bounds=(-0.25,0.25, -0.25,0.25, -0.02, 1), invert=False)


    def callback(*args):
        print(args[0].bounds)
        bounds[:] = args[0].bounds


    pl = pyvista.Plotter(window_size=(1920, 1080))
    pl.add_mesh(polydata, scalars='colors', rgb=True)
    box_widget = pl.add_box_widget(rotation_enabled=False, bounds=bounds, factor=1, callback=callback)
    pl.show()
    print('-->', bounds)

    print(f'writing {str(bounding_box_path)}')
    np.savetxt(bounding_box_path, bounds)
    


    
if __name__ == '__main__':
    main()
