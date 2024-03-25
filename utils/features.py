# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import re
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from collections import defaultdict 
from typing import List

# the tags on the default board are in the range [0,..,23]
DEFAULT_VALID_TAG_IDS = set(range(24))

class AprilTag:
    __slots__ = 'family', 'id', 'center', 'corners', 'margin'
    def __init__(self, family, id, center, corners, margin=None):
        self.family = family
        self.id = id
        self.center = center
        self.corners = corners
        self.margin = margin
    
    def __repr__(self) -> str:
        return f"AprilTag(family={self.family}, id={self.id}, center={self.center}, corners={self.corners}, margin={self.margin})"

class CustomKeypoint:
    """A keypoint in an image."""
    __slots__ = 'id', 'point', 'colmap_id'
    def __init__(self, id, point, colmap_id=-1):
        self.id = id
        self.point = point
        self.colmap_id = colmap_id

    def __repr__(self) -> str:
        return f"CustomKeypoint(id={self.id}, point={self.point}, colmap_id={self.colmap_id})"

    def create_dummy_descriptor(self):
        """Creates a dummy str descriptor with length 128."""
        desc = '<<<{:<122}>>>'.format(f'id="{self.id}"')
        desc = np.frombuffer(desc.encode(), dtype=np.uint8).copy()
        # desc[self.tag_id[0]*4+self.tag_id[1]+6] = 255
        assert len(desc) == 128
        return desc

    @staticmethod
    def identify_from_descriptor(desc):
        """Returns the tag_id and point_id from a descriptor."""
        assert len(desc) == 128
        try:
            dstr = desc.tobytes().decode('ascii')
            match = re.match(r'<<<id="(.*)"\s*>>>', dstr)
            if match:
                identifier = match.group(1)
                return identifier
        except:
            pass
        return None
    

def convert_tag_to_customkeypoints(tag):
    """Converts an april tag to a list of CustomKeypoints"""
    return [CustomKeypoint(f"{tag.family}.{tag.id}.{i}", tuple(co), -1) for i, co in enumerate(tag.corners)]


def get_grayscale_array_for_detection(img):
    """Returns a grayscale image as numpy array for detection.
    Args:
        img: The image. This is one of (str, Path, PIL.Image, np.ndarray)

    Returns:
        A 2D numpy array with the grayscale image.
    """
    if isinstance(img, (str,Path)):
        with Image.open(img) as im_color:
            im = ImageOps.grayscale(im_color)
            return np.asarray(im)
    elif isinstance(img, Image.Image):
        im = ImageOps.grayscale(im_color)
        return np.asarray(im)
    elif isinstance(img, np.ndarray):
        if img.ndim == 3 and img.dtype == np.uint8:
            im_color = Image.fromarray(img)
            im = ImageOps.grayscale(im_color)
            return np.asarray(im)
        elif img.ndim == 2 and img.dtype == np.uint8:
            return img
        else:
            raise ValueError(f"unsupported conversion {img.shape} {img.dtype}")
    else:
        raise ValueError("unsupported conversion")


def draw_cross(x: float, y: float, draw: ImageDraw, radius=20, width=5, fill='red', text=None):
    """Shortcut for drawing a cross with PIL.ImageDraw
    Args:
        x: x coordinate (column)
        y: y coordinate (row)
        draw: ImageDraw of a color image
        radius: The radius of the cross
        width: width of the lines
        fill: fill color
        text: Optional text
    """
    s = radius
    l = [(x-s, y), (x+s, y), (x,y), (x,y-s), (x,y+s)]
    draw.line(l, fill=fill, width=width)
    if text:
        draw.text((x+s,y+s), text)


def draw_tag(tag: AprilTag, draw: ImageDraw, radius=50, width=5, fill=['red','green','blue', 'white']):
    """Draws the corners of a tag with PIL.ImageDraw
    Args:
        tag: Tag as returned by the april tag detector.
        draw: ImageDraw of a color image corresponding to the detection.
    """
    s = radius
    if isinstance(fill, (tuple,list)):
        fill_colors = fill
    else:
        fill_colors = 4*[fill]
    for co, color in zip(tag.corners, fill_colors):
        c = tuple(co)
        draw_cross(*c, draw=draw, radius=radius, width=width, fill=color)
    draw.text(tuple(tag.center), f'id={tag.id}, m={tag.margin}')


class ApriltagDetector:
    def __init__(self, family: str='tag16h5', valid_ids=None):
        """Creates a detector for a specific Apriltag family.
        Args:
            family: The tag family, e.g. 'tag16h5'.
            valid_ids (set): A set of ids which limits the detections to this set. Set to None to allow all ids.
        """
        from apriltag import apriltag
        self._family = family
        self._detector = apriltag(family, decimate=4.0)
        self._valid_ids = valid_ids

    def detect(self, img: np.ndarray, min_size=150, min_margin=28, unique_ids=True, debug=False) -> List[AprilTag]:
        """Returns the detected tags for the image.
        Args:
            img (np.ndarray): 2D array with dtype uint8.
            min_size (int): The minimum size of a tag to be valid. The size is the length of the longest diagonal.
            min_margin (float): The minimum margin of a tag to be valid. Higher values correlate with higher confidence.
            unique_ids (bool): If True only the largest tag for each tag id will be kept.
            debug (bool): If True return an image with debug visualizations
        Returns:
            A list of tags that pass all tests.
        """
        tags = self._detector.detect(img)
        tags = [AprilTag(self._family, x['id'], x['center'], x['lb-rb-rt-lt'], x['margin']) for x in tags]
        result = []
        invalid = []
        id_tags = defaultdict(list)
        for t in tags:
            if self._valid_ids is None or t.id in self._valid_ids:
                id_tags[t.id].append(t)
            else:
                invalid.append(t)

        for tid, tags in id_tags.items():
            keep_i = []
            for i, t in enumerate(tags):
                valid = False
                size = max(np.linalg.norm(t.corners[0]-t.corners[2]), np.linalg.norm(t.corners[1]-t.corners[3]))
                if size > min_size and (self._valid_ids is None or t.id in self._valid_ids) and t.margin > min_margin:
                    valid = True
                if valid:
                    keep_i.append(i)
                else:
                    invalid.append(t)
            l = [tags[i] for i in keep_i]
            id_tags[tid] = l


        if unique_ids:
            for tid, tags in id_tags.items():
                keep_i = -1
                keep_size = -1
                for i, t in enumerate(tags):
                    size = max(np.linalg.norm(t.corners[0]-t.corners[2]), np.linalg.norm(t.corners[1]-t.corners[3]))
                    if size > keep_size:
                        keep_i = i
                        keep_size = size
                    else:
                        invalid.append(t)

                invalid.extend([ t for i, t in enumerate(tags) if i != keep_i ])
                if keep_i >= 0:
                    id_tags[tid] = [tags[keep_i]]

        for tid, tags in id_tags.items():
            result.extend(tags)

        if debug:
            dbg_img = Image.fromarray(np.stack([img]*3, axis=-1))
            draw = ImageDraw.ImageDraw(dbg_img)
            for t in result:
                draw_tag(t, draw)
            for t in invalid:
                draw_tag(t, draw, fill='#999999')
            return result, dbg_img

        return result

