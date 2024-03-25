# SPDX-License-Identifier: Apache-2.0
import sys
import argparse
from pathlib import Path
# from lxml.etree import parse
from defusedxml.lxml import parse
import numpy as np
import json
import re

def debug_vis(json_path):
    import open3d as o3d
    with open(json_path, 'r') as f:
        tagcorner_point = json.load(f)
    
    corner_color = np.array([[1,0,0], [0,1,0], [0,0,1], [1,1,1]], dtype=np.float32)
    tag_ids = np.unique(sorted([int(x.split('.')[0]) for x in tagcorner_point]))
    
    meshes = []
    for tag_id in tag_ids:
        points = []
        colors = []
        for corner_i in range(4):
            key = f'{tag_id}.{corner_i}'
            p = tagcorner_point[key]
            points.append(p)
            colors.append(corner_color[corner_i])
        mesh = o3d.t.geometry.TriangleMesh()
        mesh.vertex.positions = np.array(points).astype(np.float32)
        mesh.vertex.colors = np.stack(colors)
        mesh.triangle.indices = np.array([[0,1,2], [2,3,0]])
        meshes.append(mesh)
        text = o3d.t.geometry.TriangleMesh.create_text(f'id={tag_id}')
        text.translate(np.mean(points, axis=0))
        text.translate([0,0,1])
        text.scale(0.0005, center=np.mean(points, axis=0))
        meshes.append(text)

    o3d.visualization.draw(meshes)
    


def main():

    parser = argparse.ArgumentParser(
        description="Writes the coordinates of the corners of the apriltag to a json file. The unit of the output coordinates is meter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("input", type=Path, help="Input svg file.")
    parser.add_argument("output", type=Path, help="Output json file.")
    parser.add_argument("--debug", action='store_true', help="Visualized board in o3d.")


    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    print(args)

    tree = parse(args.input)
    root = tree.getroot()

    viewbox = list(map(float,root.attrib['viewBox'].split(' ')))
    img_width, img_height = viewbox[2], viewbox[3]
    assert viewbox[0] == 0 and viewbox[1] == 0, 'origin is expected to be at (0,0)'
    assert re.match('^\d+mm$', root.attrib['width']), 'unit must be millimeter'

    layer = root.xpath("//*[@id = 'layer1']")[0]

    corners = np.array([[-1,1], [1,1], [1,-1], [-1,-1]])
    tag_size_px = 8
    tag_border_px = 1
    corner_offset_px = 0.5*(tag_size_px - 2*tag_border_px)

    tag_keypoints = {}
    for el in layer:
        inkscape_label = '{'+root.nsmap['inkscape']+'}'+'label'
        if inkscape_label in el.attrib.keys():
            tag_id = el.attrib[inkscape_label]
            width = float(el.attrib['width'])
            height = float(el.attrib['height'])
            x = float(el.attrib['x'])
            y = float(el.attrib['y'])
            center = np.array((x+0.5*width, y+0.5*height))
            corner_offset = np.array([width,height])*(corner_offset_px/tag_size_px)
            # print(tag_id, x, y, width, height)

            for i, corner in enumerate(corners):
                point = center + corner_offset * corner
                # make the center of the image the origin and mirror y to get z+ as the up-axis
                point -= 0.5*np.array((img_width,img_height))
                point[1] = -point[1]
                # convert to meter
                point /= 1000
                tag_keypoints[f'{tag_id}.{i}'] = (*point,0)  
                print(f'tag_id {tag_id:2>} corner {i} at {point}')

    with open(args.output,'w') as f:
        json.dump(tag_keypoints, f, indent=4)


    if args.debug:
        debug_vis(args.output)

if __name__ == "__main__":
    main()
