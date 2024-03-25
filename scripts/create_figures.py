# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
print(sys.path)
import numpy as np
from PIL import Image
import argparse
import json
import shutil
from utils.constants import DATASET_PATH
from utils import hdri

def alter_mask_for_chest(mask: np.ndarray):
    """The chest object has a movable part that is masked out but makes images look confusing. 
    This function modifies the mask for visualization purposes only.
    
    Args:
        mask: 2D array with type bool
        
    Returns:
        New mask created with flood_fill
    """
    from skimage.morphology import flood_fill
    m = mask.astype(np.uint8)
    ans = flood_fill(m, (0,0), 100,)
    return ans != 100    


def crop_params_with_margin(mask: np.ndarray, margin: float=0):
    """Computes cropping paramters using the mask and a margin based on the fraction of the tightest crop
    Args:
        mask: 2D array with dtype bool
        margin: margin as fraction of the tightest crop of the smaller dim.
        
    Returns:
        Tuple (row_start, row_end, col_start, col_end)
    """
    col_mask = mask.any(axis=0)
    col_start = np.argmax(col_mask)
    col_end = col_mask.shape[0] - np.argmax(col_mask[::-1])

    row_mask = mask.any(axis=1)
    row_start = np.argmax(row_mask)
    row_end = row_mask.shape[0] - np.argmax(row_mask[::-1])
    
    # the largest possible margin
    largest_margin = min(col_start, row_start, col_mask.shape[0]-col_end, row_mask.shape[0]-row_end)
    
    smallest_size = min(col_end-col_start, row_end-row_start)
    m = min(int(np.floor(smallest_size*margin)), largest_margin)
    
    return (row_start-m, row_end+m, col_start-m, col_end+m)


def crop(arr, params):
    """Crop using the parameters from crop_params_with_margin()"""
    return arr[params[0]:params[1], params[2]:params[3], ...]


def add_padding(arr, fraction, value=255):
    """Adds padding based on the fraction of the shortest edge"""
    m = int(fraction*min(arr.shape[0], arr.shape[1]))
    p = [(m,m),(m,m)] + [(0,0) for i in range(arr.ndim-2)]
    return np.pad(arr, p, constant_values=value)


def process_img(im, mask):
    m = np.tile(mask[...,None], (1,1,3))
    if im.shape != m.shape:
        im = np.array(Image.fromarray(im).resize((mask.shape[1], mask.shape[0])))
    im[~mask] = 255
    crop_params = crop_params_with_margin(mask)
    return add_padding(crop(im, crop_params), 0.1)


def create_figure_table(table, output_dir, tablename, imwidth='2cm', maximheight='2cm', rotate_firstcol=0, table_env=False, preview=False, document=False, maximres=(800,800), selcols=None):
    """Creates a latex table from a numpy array with object dtype and PIL.Image.Image objects
    Args:
        table: 2D numpy array with dtype=object and str or PIL.Image.Image objects.
        output_dir: Output dir for the table.
        tablename: Name of the table. This will be used to create a subdir in output_dir.
        imwidth: The width of all images as latex length str.
        maximheight: The max height of an image as latex length str.
        rotate_firstcol: The rotation angle for the text objects in the first column.
        table_env: If True add the table env to the output.
        preview: If True wrap the tabular inside the preview env and use an adjustbox to scale to the page width
        document: If True add a document env to the output.
        maximres: The maximum image resolution in px
        selcols: List of col indices to show. If None show all cols
    """
    doc_header = r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{etoolbox}
\usepackage{siunitx}
\usepackage{multirow}
\usepackage[normalem]{ulem}
\usepackage{adjustbox}
\usepackage[active,tightpage]{preview}

\usepackage{array}
\newcolumntype{H}{>{\setbox0=\hbox\bgroup}c<{\egroup}@{}}
\graphicspath{{../}}

\begin{document}
"""
    table_env_header = r"""\begin{table}
\centering
"""

    preview_env_header = r"""\begin{preview}
\centering
\begin{adjustbox}{max width=\textwidth}
"""

    output_dir = Path(output_dir)
    outdir = output_dir/tablename
    outdir.mkdir(exist_ok=True, parents=True)
    
    lines = []
    if document:
        lines.append(doc_header)
    if table_env:
        lines.append(table_env_header)
    if preview:
        lines.append(preview_env_header)

    lines.append(f'\\def\\{tablename}imwidth{{{imwidth}}}')
    lines.append(f'\\def\\{tablename}maximheight{{{maximheight}}}')

    if selcols is None:
        selcols = set(range(table.shape[1]))
    else:
        selcols = set(selcols)
    zerocolspace = '@{}'
    coldefs = [
        'r@{\hskip 2pt}' if 0 in selcols else 'H@{}',
    ]
    for col_i in range(1,table.shape[1]):
        coldefs.append('c' if col_i in selcols else 'H')

    tabular_coldef = zerocolspace + coldefs[0] + zerocolspace.join(coldefs[1:]) + zerocolspace
    print(tabular_coldef)

    lines.append(r'\begin{tabular}'+'{'+tabular_coldef+'}')

    for i, row in enumerate(table):
        l = []
        for j, cell in enumerate(row):
            if isinstance(cell,str):
                angle = 0
                if j == 0:
                    angle = rotate_firstcol
                l.append(f'\\adjustbox{{angle={angle},valign=m}}{{'+cell+'}')
            elif isinstance(cell, Image.Image):
                p = outdir/f'{i}_{j}.jpg'
                if hasattr(cell,'outname'):
                    p = outdir/cell.outname
                if maximres:
                    cell.thumbnail(maximres)
                cell.save(p)
                l.append(f'\\adjustimage{{valign=m,width=\\{tablename}imwidth,max height=\\{tablename}maximheight,keepaspectratio}}'+f'{{{tablename}/{p.name}}}')
            elif cell is None:
                l.append('')

        lines.append(' & '.join(l)+r'\\')

    lines.append(r'\end{tabular}')
    if preview:
        lines.append(r'\end{adjustbox}')
        lines.append(r'\end{preview}')
    if table_env:
        lines.append(r'\end{table}')
    if document:
        lines.append(r'\end{document}')

    print('\n'.join(lines))

    with open(outdir/'table.tex','w') as f:
        f.write('\n'.join(lines))



def create_table(methods_data, subdir, num_gt_images, gt_root, include_envs):

    def read_im(p):
        with Image.open(p) as im:
            im.load()
        return np.array(im)
    

    num_methods = len(methods_data)

    object_names = set()
    for x in methods_data:
        for gt_image_path in x['data'].keys():
            if subdir:
                subdir_name = Path(gt_image_path).parent.name
                obj_name = Path(gt_image_path).parent.parent.name
            else:
                obj_name = Path(gt_image_path).parent.name
            object_names.add(obj_name)

    object_names = sorted(list(object_names))


    gt_image_names = ['gt_image_{:04d}.png'.format(x) for x in range(num_gt_images)]
    
    table = np.empty((num_methods+2, len(object_names)*num_gt_images+1), dtype=object)

    masks = {}

    if include_envs:
        table[0,0] = 'Envmap'
        table[1,0] = 'GT'
    else:
        table[0,0] = 'GT'
    col_i = 1
    for obj in object_names:
        for gt_im in gt_image_names:
            key = f'{obj}/{subdir}/{gt_im}'.replace('//','/')
            gt_im_path = gt_root/key
            mask_path = gt_im_path.parent/gt_im_path.name.replace('_image_','_mask_')
            masks[key] = read_im(mask_path).max(axis=-1)==255
            if obj in ('chest',):
                masks[key] = alter_mask_for_chest(masks[key])
            row_i = 1 if include_envs else 0
            table[row_i,col_i] = Image.fromarray(process_img(read_im(gt_im_path), masks[key]))
            table[row_i,col_i].outname = f'GT_{obj}_{gt_im_path.name}.jpg'
            
            if include_envs:
                env_path = (gt_im_path.parent/gt_im_path.name.replace('_image_','_env_')).with_suffix('.hdr')
                envim = hdri.simple_downsample(hdri.simple_downsample(hdri.read_hdri(env_path)))
                exposure = np.log2(0.3/np.median(envim))
                envim = hdri.simple_tonemap(envim, exposure)
                table[0,col_i] = Image.fromarray(envim)
                table[0,col_i].outname = f'env_{obj}_{gt_im_path.name}.jpg'
            col_i += 1

    row_offset = 2 if include_envs else 1
    for method_i, md in enumerate(methods_data):
        print(md['name'])
        table[method_i+row_offset,0] = md['name']
        col_i = 1
        row_i = method_i + row_offset
        for obj in object_names:
            for gt_im in gt_image_names:
                key = f'{obj}/{subdir}/{gt_im}'.replace('//','/')
                pr_im_path = Path(md['data'][key]['prediction_path'])
                if pr_im_path.suffix != '.png':
                    pr_im_path = pr_im_path.parent/(pr_im_path.name+'.tonemapped.png')
                table[row_i,col_i] = Image.fromarray(process_img(read_im(pr_im_path), masks[key]))
                table[row_i,col_i].outname = f'{md["name"].replace("+","_")}_{obj}_{pr_im_path.name}.jpg'
                col_i += 1

    return table




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script that creates latex figure with the prediction images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('inputs', type=Path, nargs='+', help="Path to the input json files")
    parser.add_argument('-g', '--ground_truth', type=Path, default=DATASET_PATH, help="Path to the dataset directory with the ground truth")
    parser.add_argument('--output', required=True, type=Path, help="Path to the output dir")
    parser.add_argument('--zip', action='store_true', help="If True create a zip in the same dir as the output dir")
    parser.add_argument('--dataset', choices=set(['ours', 'ours_nvs', 'synthetic4relight', ]), default='ours', help="The dataset")
    parser.add_argument('--selcols', type=int, nargs='+', help="Columns to show. If not specified all columns will be shown")
    parser.add_argument('--include_envs', action='store_true', help="If True add the GT environment maps")

    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    print(args)

    methods_data = []
    for p in args.inputs:
        with open(p,'r') as f:
            methods_data.append(json.load(f))

    num_gt_images = {
        'ours': 9,
        'ours_nvs': 3,
        'synthetic4relight': 39,
    }[args.dataset]
    subdir = {
        'ours': 'test',
        'ours_nvs': 'test',
        'synthetic4relight': '',
    }[args.dataset]

    table = create_table(methods_data=methods_data, subdir=subdir, num_gt_images=num_gt_images, gt_root=args.ground_truth, include_envs=args.include_envs)
    print(table.shape)

    output_dir = args.output.resolve().parent

    create_figure_table(table, output_dir, args.output.name, imwidth='2cm', maximheight='1.4cm', rotate_firstcol=0, table_env=False, preview=True, document=True, maximres=(400,400), selcols=args.selcols)

    if args.zip:
        shutil.make_archive(args.output, 'zip', root_dir=output_dir, base_dir=args.output.name)