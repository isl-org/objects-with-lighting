#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import sys
import argparse
from pathlib import Path
import multiprocessing 
from typing import List, Dict
from collections import OrderedDict
from tqdm import tqdm
import json

from PIL import ImageTk, Image
import tkinter
from tkinter.messagebox import askyesno
from tkinter import Tk, ttk, Canvas, Listbox, StringVar, IntVar, N, W, S, E, ALL
import numpy as np
import math
import time

DISTINCT_COLORS = [ '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#ffffff', ]


def triangulate_point(Pvec: np.ndarray, xvec: np.ndarray):
    """Triangulate a 3D point with multiple observations using the DLT method.
    Args:
        Pvec: (N,3,4) array with the projection matrices.
        xvec: (N,2) array with the 2d point observations.
    Returns:
        The least squared solution for the triangulated 3D point.
    """
    assert Pvec.shape[1:] == (3,4)
    assert xvec.shape[1:] == (2,)
    assert Pvec.shape[0] == xvec.shape[0]
    n = Pvec.shape[0]
    A = np.empty(shape=(2*n,3))
    b = np.empty(shape=(2*n,))
    for i, (P, point2d) in enumerate(zip(Pvec, xvec)):
        x, y = point2d
        z = 1
        A[i*2+0,:] = y*P[2,0:3] - z*P[1,0:3]
        A[i*2+1,:] = z*P[0,0:3] - x*P[2,0:3]
        b[2*i+0] = z*P[1,3] - y*P[2,3]
        b[2*i+1] = x*P[2,3] - z*P[0,3]

    x, res, rank, singular_values = np.linalg.lstsq(A,b, rcond=None)
    return x


def compute_keypoint_guess(kp, im_stem: str, im_width_height: np.ndarray):
    """Computes a guess for the keypoint position in the image.
    Args:
        kp: The keypoint
        im_stem: The name of the image without extension for looking up the projection matrix.
        im_width_height: width and height of the image
    Returns:
        A tuple (x,y) or None.
    """
    if compute_keypoint_guess.projection_matrices is None:
        return None

    projmat_dict = None
    for imname_P in compute_keypoint_guess.projection_matrices:
        if im_stem in imname_P:
            projmat_dict = imname_P
            break
    if projmat_dict is None:
        return None

    Pvec = []
    xvec = []
    for im_path, pos in kp.get_all_pos().items():
        P = projmat_dict.get(im_path.stem)
        if P is not None:
            Pvec.append(P)
            xvec.append(pos)
    if len(xvec)>=2:
        X = np.ones((4,1), dtype=np.float64)
        X[:3,0] = triangulate_point(np.stack(Pvec), np.array(xvec))
        P = projmat_dict[im_stem]
        x = np.squeeze(P @ X)
        wh = np.array(im_width_height)
        xy = x[:2] / x[-1]
        if np.all(x > 0) and np.all(xy<wh):
            return tuple(xy)
    return None
compute_keypoint_guess.projection_matrices = None


class MyListBox:
    """Simple wrapper for the listbox widget"""
    def __init__(self, parent):
        self._listvar = StringVar()
        self._listbox = Listbox(parent, listvariable=self._listvar, exportselection=0, font='TkFixedFont')
        self._listbox.grid(row=0, column=0, sticky="NSEW")
        self._selection_history = [None]

        def sel_change(event):
            self._selection_history.append(self.get_selection())
            while len(self._selection_history) > 5:
                self._selection_history.pop(0)
            print('select', self._selection_history)
        self._listbox.bind('<<ListboxSelect>>', sel_change, add='+')

    @property
    def mainwidget(self):
        return self._listbox

    def set_list(self, values):
        self._listvar.set(values)

    def get_previous_selection(self):
        return self._selection_history[-2]

    def get_selection(self):
        return self._listbox.curselection()

    def select(self, idx):
        self._selection_history.append((idx,))
        while len(self._selection_history) > 5:
            self._selection_history.pop(0)
        print('select2', self._selection_history)
        self._listbox.select_clear(0, tkinter.END)
        self._listbox.select_set(idx)

    def set_item_bg_color(self, idx, color):
        self._listbox.itemconfig(idx, bg=color)

    def __len__(self):
        return len(self._listbox)


class ImageList:
    """The widget for displaying the image paths"""
    def __init__(self, parent, image_paths):
        self._image_paths = image_paths
        self._kp = None
        self._keypoints = None
        self._change_callback = None
        self._mainframe = ttk.Frame(parent, padding="5 5 5 5")
        self._label = ttk.Label(self._mainframe, text="Images")
        
        self._listbox = MyListBox(self._mainframe)
        self._listbox.mainwidget.grid(row=1, column=0, sticky="NWSE")
        
        self._label.grid(row=0, column=0, sticky="NWSE")

        self._listbox.set_list([str(x) for x in self._image_paths])
        self._listbox.select(0)
        self._projection_matrices = None


        # self._mainframe.grid_rowconfigure(0, weight=1)
        self._mainframe.grid_rowconfigure(1, weight=1)
        self._mainframe.grid_columnconfigure(0, weight=1)
        self._listbox._listbox.bind('<<ListboxSelect>>', self._change, add='+')

    def _change(self, event):
        if self._change_callback is not None:
            self._change_callback(self.current_index)

    def set_change_callback(self, cb):
        self._change_callback = cb

    @property
    def mainwidget(self):
        return self._mainframe
        
    @property
    def current_index(self):
        return self._listbox.get_selection()[0]

    @property
    def current_image(self):
        return self._image_paths[self._listbox.get_selection()[0]]

    def set_current_keypoint(self, kp):
        self._kp = kp
        
    def update_list(self):
        if self._kp is None:
            return
        images = self._kp.get_all_pos().keys()
        l = []

        for x in self._image_paths:
            s = str(x)
            count_keypoints = 0
            for k in self._keypoints:
                if x in k.get_all_pos().keys():
                    count_keypoints += 1
            s += f' ({count_keypoints})'
            if x in images:
                s += '  X'
            l.append(s)
        self._listbox.set_list(l)
        
        # highlight images that have no projection matrix
        if self._projection_matrices is not None:
            for i, x in enumerate(self._image_paths):
                image_has_proj_matrix = False
                for proj_matrices_dict in self._projection_matrices:
                    if x.stem in proj_matrices_dict:
                        image_has_proj_matrix = True
                        break
                if not image_has_proj_matrix:
                    self._listbox.set_item_bg_color(i, '#ffaa00')
            


class DashedCircle:
    """A circle shaped marker for the Canvas"""
    def __init__(self, canvas: Canvas, x: float, y: float, size: int, color, width: int=2, text='A'):
        print(x,y,size,canvas)
        self._canvas = canvas
        self._pos = (x,y)
        self._size = size
        self._color = color
        self._text = text
        self._tags = [
            canvas.create_oval(x-size, y-size, x+size, y+size, fill='', outline=color, width=width, dash=(3,5)),
        ]

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, value):
        x, y = value
        size = self._size
        self._pos = value
        self._canvas.coords(self._tags[0], x-size, y-size, x+size, y+size)

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, size):
        self._size = size
        x, y = self._pos
        self._canvas.coords(self._tags[0], x-size, y-size, x+size, y+size)

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        self._color = value
        self._canvas.itemconfigure(self._tags[0], fill=value)

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, text):
        if self._text == text:
            return
        # self._canvas.itemconfigure(self._tags[2], text=text)


    def destroy(self):
        self._canvas.delete(*self._tags)



class Cross:
    """A cross shaped marker for the Canvas"""
    def __init__(self, canvas: Canvas, x: float, y: float, size: int, color, width: int=2, text='A'):
        self._canvas = canvas
        self._pos = (x,y)
        self._size = size
        self._color = color
        self._text = text
        self._tags = [
            canvas.create_line(x-size, y, x+size, y, fill=color, width=width),
            canvas.create_line(x, y-size, x, y+size, fill=color, width=width),
            canvas.create_text(x+size, y+size, fill=color, text=self._text),
        ]

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, value):
        x, y = value
        size = self._size
        self._pos = value
        self._canvas.coords(self._tags[0], x-size, y, x+size, y)
        self._canvas.coords(self._tags[1], x, y-size, x, y+size)
        self._canvas.coords(self._tags[2], x+size, y+size)

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, size):
        self._size = size
        x, y = self._pos
        self._canvas.coords(self._tags[0], x-size, y, x+size, y)
        self._canvas.coords(self._tags[1], x, y-size, x, y+size)
        self._canvas.coords(self._tags[2], x+size, y+size)

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        self._color = value
        self._canvas.itemconfigure(self._tags[0], fill=value)
        self._canvas.itemconfigure(self._tags[1], fill=value)
        self._canvas.itemconfigure(self._tags[2], fill=value)

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, text):
        if self._text == text:
            return
        self._canvas.itemconfigure(self._tags[2], text=text)


    def destroy(self):
        self._canvas.delete(*self._tags)


class Keypoint:
    """Represents a keypoint. Stores coordinates for each image."""
    def __init__(self, kplist, name:str, color):
        self._kplist = kplist
        self._name = name
        self._color = color
        self._image_pos = {}

    @property
    def name(self):
        return self._name

    @property
    def color(self):
        return self._color

    def get_all_pos(self):
        return self._image_pos

    def get_pos(self, image_key):
        return self._image_pos.get(image_key,None)
    
    def set_pos(self, image_key, value):
        self._image_pos[image_key] = value
        self._kplist._update_listbox()

    def remove_pos(self, image_key):
        if image_key in self._image_pos:
            del self._image_pos[image_key]
            self._kplist._update_listbox()

    def get_str(self, image_key=None) -> str:
        xy = self._image_pos.get(image_key, '')
        if xy:
            xy = f'(x={xy[0]:.2f}, y={xy[1]:.2f})'
        return f'{self._name:<5} {xy:<24} #images: {len(self._image_pos):>2}'

    def _set_from_dict(self, values):
        self._image_pos = {k: (v['x']-0.5, v['y']-0.5) for k,v in values.items()}


class KeypointList:
    """The widget for managing the keypoints."""
    def __init__(self, parent, keypoints_from_json=None):
        self._keypoints = []
        self._current_image = None

        self._change_callback = None
        self._mainframe = ttk.Frame(parent, padding="5 5 5 5")
        self._label = ttk.Label(self._mainframe, text="Keypoints")
        self._new_point_button = ttk.Button(self._mainframe, text="New point", command=self._add_point)
        self._remove_from_image_button = ttk.Button(self._mainframe, text="Remove point from image", command=self._remove_from_image)
        self._delete_point_button = ttk.Button(self._mainframe, text="Delete point", command=self._delete_point)
        
        self._listbox = MyListBox(self._mainframe)
        self._listbox.mainwidget.grid(row=1, column=0, columnspan=3, sticky="NWSE")
        
        self._label.grid(row=0, column=0, sticky="NWSE")
        self._new_point_button.grid(row=2, column=0, sticky="NWSE")
        self._remove_from_image_button.grid(row=2, column=1, sticky="NWSE")
        self._delete_point_button.grid(row=2, column=2, sticky="NWSE")

        if keypoints_from_json:
            self._add_points_from_dict(keypoints_from_json)
        else:
            self._add_point()
        self._listbox.select(0)


        # self._mainframe.grid_rowconfigure(0, weight=1)
        self._mainframe.grid_rowconfigure(1, weight=1)
        self._mainframe.grid_columnconfigure(0, weight=1)
        self._mainframe.grid_columnconfigure(1, weight=1)
        self._mainframe.grid_columnconfigure(2, weight=0)
        self._listbox._listbox.bind('<<ListboxSelect>>', self._change)

    def _change(self, event):
        if self._change_callback is not None:
            self._change_callback(self._listbox.get_selection(), self._keypoints)

    def set_change_callback(self, cb):
        self._change_callback = cb

    @property
    def keypoints(self):
        return self._keypoints

    @property
    def mainwidget(self):
        return self._mainframe
        
    @property
    def current_image(self):
        return self._current_image

    @current_image.setter
    def current_image(self, value):
        self._current_image = value
        self._update_listbox()
        
    def _update_listbox(self):
        self._listbox.set_list([x.get_str(self._current_image) for x in self._keypoints])
        
    def _add_point(self):
        used_names = set([x.name for x in self._keypoints])
        color = DISTINCT_COLORS[len(self._keypoints) % len(DISTINCT_COLORS)]

        name = 0
        while str(name) in used_names:
            name += 1
        kp = Keypoint(self, "{}".format(name), color)
        self._keypoints.append(kp)
        self._update_listbox()
        self._listbox.select(len(self._keypoints)-1)
        self._change(None)
        
    def _remove_from_image(self):
        sel = self._listbox.get_selection()
        for i in sel:
            self._keypoints[i].remove_pos(self._current_image)
        self._change(None)

    def _delete_point(self):
        if len(self._keypoints) == 1:
            return

        sel = self._listbox.get_selection()
        for i in reversed(sorted(sel)):
            yes = askyesno(title="Confirm deleting point", message="Are you sure?")
            if not yes:
                return
            if hasattr(self._keypoints[i], "crosses"):
                for c in self._keypoints[i].crosses:
                    c.destroy()
            del self._keypoints[i]
        self._update_listbox()
        self._listbox.select(0)
        self._change(None)

    def _add_points_from_dict(self, data):
        keys = sorted(list(data.keys()))
        for k in keys:
            d = data[k]
            color = DISTINCT_COLORS[len(self._keypoints) % len(DISTINCT_COLORS)]
            kp = Keypoint(self, k, color)
            kp._set_from_dict(d)
            self._keypoints.append(kp)
        self._update_listbox()


class AnnotationArea:
    """The widget for annotating images with keypoints. Shows already placed keypoints and provides 4 zoom levels."""
    zoom_factor = 3
    inactive_cross_size = 15
    cross_size = 20

    def __init__(self, parent) -> None:
        self._mainframe = ttk.Frame(parent,)
        self._canvases = []
        self._imagetags = []
        self._photos = {}
        self._active_keypoint_idx = 0
        self._keypoints = []
        self._c2i_transform = [np.eye(3)]*4 # transforms from canvas coordinates to the image coordinates
        self._click_callback = None

        for row in range(2):
            for col in range(2):
                canvas = Canvas(self._mainframe, background='grey')
                canvas.grid(row=row, column=col, sticky="NWSE")
                self._imagetags.append(canvas.create_image(0,0, anchor='nw'))
                self._canvases.append(canvas)
        
        for canvas in self._canvases:
            canvas.bind("<Button-1>", lambda x: self._click_canvas(x))

        self._mainframe.grid_rowconfigure(0, weight=1)
        self._mainframe.grid_rowconfigure(1, weight=1)
        self._mainframe.grid_columnconfigure(0, weight=1)
        self._mainframe.grid_columnconfigure(1, weight=1)

    @property
    def mainwidget(self):
        return self._mainframe

    @property
    def current_image(self):
        return self._current_image

    @property
    def active_keypoint(self):
        return self._keypoints[self._active_keypoint_idx]

    def set_click_callback(self, cb):
        self._click_callback = cb

    def _click_canvas(self, event):
        if not hasattr(self, '_current_image'):
            return
        idx = self._canvases.index(event.widget)
        c2i = self._c2i_transform[idx]
        xy = c2i @ np.array([event.x, event.y, 1])
        update_canvas = [i for i in range(1,4) if i != idx]
        self.set_active_keypoint(xy[0], xy[1], update_canvas)
        if self._click_callback is not None:
            self._click_callback(xy[0], xy[1])


    @staticmethod
    def _crop_around_point(x: float, y: float, w: float, h:float, image: Image):
        left = max(0,min(int(x-w/2), image.size[0]))
        upper = max(0,min(int(y-h/2), image.size[1]))
        right = max(0,min(int(x+w/2), image.size[0]))
        lower = max(0,min(int(y+h/2), image.size[1]))
        # print(left, upper, right, lower)
        crop = image.crop((left, upper, right, lower))
        c2i_transform = np.eye(3)
        c2i_transform[0,2] = left
        c2i_transform[1,2] = upper
        return crop, c2i_transform

    @staticmethod
    def _resize_to_canvas(image: Image, canvas: Canvas):
        image_aspect_ratio = image.size[0]/image.size[1]
        canvas_aspect_ratio = canvas.winfo_width()/canvas.winfo_height()
        if canvas_aspect_ratio > image_aspect_ratio:
            h = canvas.winfo_height()
            w = math.floor(image_aspect_ratio*h)
        else:
            w = canvas.winfo_width()
            h = math.floor(w/image_aspect_ratio)
        
        if w == 0:
            w = 1
        if h == 0:
            h = 1
        c2i_transform = np.eye(3)
        c2i_transform[0,0] = image.size[0]/w
        c2i_transform[1,1] = image.size[1]/h
        return image.resize((w,h),resample=Image.NEAREST), c2i_transform

    def set_image(self, key_image):
        key, image = key_image
        self._current_image = key
        self._image = image
        self._photos.clear()
        print('--')
        for i, (canvas, imgtag) in enumerate(zip(self._canvases, self._imagetags)):
            if i == 0:
                # full image
                full_image, c2i_transform = self._resize_to_canvas(image, canvas)
                self._c2i_transform[0] = c2i_transform
                # print(c2i_transform)
                im = ImageTk.PhotoImage(image=full_image)
                self._photos[i] = im
                canvas.itemconfig(imgtag, image=im)
            break

        self._update_zoomed_crops()
        for kp in self._keypoints:
            self.update_keypoint(kp)


    def _update_zoomed_crops(self, canvas_index=(1,2,3)):
        image = self._image
        for i in canvas_index:
            if i in self._photos:
                del self._photos[i]
        for i, (canvas, imgtag) in enumerate(zip(self._canvases, self._imagetags)):
            if i in canvas_index:
                canvas_aspect_ratio = canvas.winfo_width()/canvas.winfo_height()
                crop_height = max(64,min(image.size)//(self.zoom_factor**i))
                crop_width = max(64,math.floor(canvas_aspect_ratio*crop_height))
                pos = self._keypoints[self._active_keypoint_idx].get_pos(self._current_image)
                self.active_keypoint.guess = None
                if pos is None:
                    # try computing a guess
                    pos = compute_keypoint_guess(self.active_keypoint, self._current_image.stem, np.array(self._image.size))
                    self.active_keypoint.guess = pos
                if pos is None:
                    pos = (0,0)
                crop, translation = self._crop_around_point(*pos, crop_width, crop_height, image)
                crop_resized, scale = self._resize_to_canvas(crop, canvas)
                c2i = scale
                c2i[:,2] = translation[:,2]
                self._c2i_transform[i] = c2i
                # print(i, translation[:,2], scale.diagonal(), c2i)
                im = ImageTk.PhotoImage(image=crop_resized)
                self._photos[i] = im
                canvas.itemconfig(imgtag, image=im)
            

    def set_active_keypoint(self,x,y, update_canvas=(1,2,3)):
        kp = self._keypoints[self._active_keypoint_idx]
        kp.set_pos(self._current_image, (x,y))
        self._update_zoomed_crops(canvas_index=update_canvas)
        for kp in self._keypoints:
            self.update_keypoint(kp)

    def set_keypoints(self, keypoints=None):
        if keypoints is None:
            self._keypoints = []
        else:
            self._keypoints = keypoints


    def set_active_keypoint_idx(self, idx):
        self._active_keypoint_idx = idx
        self._update_zoomed_crops()
        for kp in self._keypoints:
            self.update_keypoint(kp)


    def update_keypoint(self, kp):
        idx = self._keypoints.index(kp)
        xy = kp.get_pos(self._current_image)
        print('kp', idx, xy, self._current_image.name)
        if xy is not None:
            poslist = []
            for c2i, canvas in zip(self._c2i_transform, self._canvases):
                inv_c2i = np.linalg.inv(c2i)
                pos = inv_c2i @ np.array([*xy,1])
                poslist.append(pos)
            if hasattr(kp, 'crosses'):
                for pos, cross in zip(poslist, kp.crosses):
                    cross.pos = tuple(pos[:2])
            else:
                crosses = [Cross(canvas, *pos[:2], self.cross_size, kp.color, text=kp.name) for pos, canvas in zip(poslist, self._canvases)]
                setattr(kp,'crosses', crosses)

            if hasattr(kp, 'circles'):
                for circle in kp.circles:
                    circle.destroy()
                delattr(kp, 'circles')
        else: # is None
            # visualize guess
            if hasattr(kp, 'crosses'):
                for cross in kp.crosses:
                    cross.destroy()
                delattr(kp, 'crosses')
            
            if kp is self._keypoints[self._active_keypoint_idx] and getattr(kp, 'guess', None) is not None:
                xy = kp.guess
                poslist = []
                for c2i, canvas in zip(self._c2i_transform, self._canvases):
                    inv_c2i = np.linalg.inv(c2i)
                    pos = inv_c2i @ np.array([xy[0], xy[1], 1])
                    poslist.append(pos)
                if hasattr(kp, 'circles'):
                    for pos, circle in zip(poslist, kp.circles):
                        circle.pos = tuple(pos[:2])
                else:
                    circles = [DashedCircle(canvas, pos[0], pos[1], self.cross_size, kp.color, text=kp.name) for pos, canvas in zip(poslist, self._canvases)]
                    setattr(kp,'circles', circles)
            else:
                if hasattr(kp, 'circles'):
                    for circle in kp.circles:
                        circle.destroy()
                    delattr(kp, 'circles')



class OtherImagesPanel:
    """A widget showing crops of oher images for a specific keypoint"""

    def __init__(self, parent, key_images, num_images=5):
        self._mainframe = ttk.Frame(parent,)
        self._key_images = dict(key_images)
        self._num_images = num_images
        self._canvases = []
        self._imagetags = []
        self._photos = {}
        self._markers = []
        self._c2i_transform = [np.eye(3)]*num_images # transforms from canvas coordinates to the image coordinates
        self._label = ttk.Label(self._mainframe, text="Other images with this keypoint")
        self._label.grid(row=0, column=0, columnspan=2, sticky="NW")
    
        self._label_zoom = ttk.Label(self._mainframe, text="Zoom factor")
        self._label_zoom.grid(row=1, column=0, columnspan=1, sticky="NW")
        self._zoom_factor = IntVar(value=6)
        def on_zoom_change(*args):
            self.update_images()
        self._zoom_factor.trace('w', on_zoom_change)
        self._spinbox = ttk.Spinbox(self._mainframe, from_=1, to=100, textvariable=self._zoom_factor)
        self._spinbox.grid(row=1, column=1, sticky="W")


        for row in range(num_images):
            canvas = Canvas(self._mainframe, background='grey')
            canvas.grid(row=row+2, column=0, columnspan=2, sticky="NWSE")
            self._imagetags.append(canvas.create_image(0,0, anchor='nw'))
            self._markers.append(Cross(canvas, -100, -100, 10, 'red'))
            self._canvases.append(canvas)
        
        self._mainframe.grid_columnconfigure(0, weight=1)
        self._mainframe.grid_rowconfigure(0, weight=0)
        for row in range(num_images):
            self._mainframe.grid_rowconfigure(row+2, weight=1)

        self._kp = None
        self._ignore_image_key = None


    @property
    def mainwidget(self):
        return self._mainframe

    def set_keypoint(self, kp: Keypoint, ignore_image_with_key):
        self._kp = kp
        self._ignore_image_key = ignore_image_with_key
        self.update_images()

    def update_images(self):
        kp = self._kp
        ignore_image_with_key = self._ignore_image_key
        if kp is None:
            return

        self._photos.clear()
        for canvas, imgtag, cross in zip(self._canvases, self._imagetags, self._markers):
            canvas.itemconfig(imgtag, image=None)
            cross.pos = (-100,-100) # hide marker

        key_pos = [(k,v) for k,v in kp.get_all_pos().items()][:4]
        key_pos = []
        for key, pos in kp.get_all_pos().items():
            if key != ignore_image_with_key:
                key_pos.append((key,pos))
        key_pos = key_pos[:self._num_images]
        for i, (canvas, imgtag, cross, (key, pos)) in enumerate(zip(self._canvases, self._imagetags, self._markers, key_pos)):
            image = self._key_images[key]
            canvas_aspect_ratio = canvas.winfo_width()/canvas.winfo_height()
            crop_height = max(64,min(image.size)//int(self._zoom_factor.get()))
            crop_width = max(64,math.floor(canvas_aspect_ratio*crop_height))
            crop, translation = AnnotationArea._crop_around_point(*pos, crop_width, crop_height, image)
            crop_resized, scale = AnnotationArea._resize_to_canvas(crop, canvas)
            c2i = scale
            c2i[:,2] = translation[:,2]
            self._c2i_transform[i] = c2i
            im = ImageTk.PhotoImage(image=crop_resized)
            self._photos[i] = im
            canvas.itemconfig(imgtag, image=im)

            inv_c2i = np.linalg.inv(c2i)
            cross_pos = inv_c2i @ np.array([*pos,1])
            cross.pos = tuple(cross_pos[:2])
            cross.color = kp.color
            cross.text = kp.name


    

def export_keypoints_to_json(path: Path, keypoints, is_autosave=False):
    # do not autosave if less than 300 seconds have passed
    if is_autosave and (time.time() - export_keypoints_to_json.autosave_time) < 300:
        return

    data = OrderedDict()
    for kp in keypoints:
        if kp.get_all_pos():
            data[kp.name] = {str(img_path): {'x': pos[0]+0.5, 'y': pos[1]+0.5} for img_path, pos in kp.get_all_pos().items() } 
    
    if is_autosave: 
        path = path.with_suffix('.autosave')
        print(f'autosaving to {str(path)}')
        export_keypoints_to_json.autosave_time = time.time()
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
export_keypoints_to_json.autosave_time = time.time()
    
            
def load_raw(x):
    import rawpy
    import cv2
    with rawpy.imread(str(x)) as raw:
        tonemap = cv2.createTonemapReinhard()
        im = raw.postprocess(user_flip=0, output_bps=16)
        im = np.nan_to_num(tonemap.process(im.astype(np.float32)/im.max()))
        im = np.clip(255*im,0,255).astype(np.uint8)
        print('*', end='', flush=True)
        return x, Image.fromarray(im)

def load_images(paths: List[Path]):
    """Load images handling raw files and the limit with opening files.
    
    Args:
        paths: List of image paths. Each path will be used in the output json file.

    Returns:
        List of PIL.Image objects.
    """
    import rawpy
    print("Loading images")
    supported_raw_files = ('.CR3', '.DNG')

    # use multiple processes if all images need raw postprocess
    if all(map(lambda x: x.suffix in supported_raw_files, paths)):
        print(len(paths)*'.', end='\r')
        with multiprocessing.Pool(processes=4) as pool:
            return pool.map(load_raw, paths)

    images = []
    open_file_handles = 0
    for x in tqdm(paths):
        if x.suffix in supported_raw_files:
            images.append(load_raw(x))
            with rawpy.imread(str(x)) as raw:
                images.append((x,Image.fromarray(raw.postprocess(user_flip=0))))
        else:
            # limit number of open files to 100
            if open_file_handles < 100:
                images.append((x, Image.open(x)))
            else:
                with Image.open(x) as im:
                    im.load()
                images.append((x,im))

    return images


def load_object_dir(path: Path):
    """Retrieve image paths from an object directory.

    An object directory has subdirs with different environments which are named 'test', 'train', 'valid'.
    
    Args:
        path: Path to an object directory.

    Returns:
        List of paths.
    """
    images = []
    for env_dir in path.iterdir():
        if len(env_dir.name) > 1 and env_dir.name not in ('train', 'valid', 'test'):
            if not env_dir.is_file():
                print(f'ignoring {str(env_dir)}')
            continue
        
        # ignore the equirectangular images which have an 'env' suffix
        # subdirs = [x for x in env_dir.iterdir() if x.is_dir() and not 'env' in x.name]
        
        subdirs = [x for x in env_dir.iterdir() if x.is_dir() and x.name in ['images', 'test1', 'test2', 'test3']]

        for d in subdirs:
            paths = sorted(list(d.glob('*.CR3')))
            # select the center of the exposure sequence for the test dirs.
            if 'test' in d.name:
                paths = paths[len(paths)//2:][:1]
            if len(paths):
                print(f'found {len(paths)} images in {str(d)}' )
                images.extend(paths)

    images = sorted(images)
    return images


def load_projection_matrices(colmap_recon_search_path: Path) -> List[Dict[str, np.ndarray]]:
    """Searches recursively for colmap sparse reconstructions and returns a dict with the projection matrices.

    Args:
        colmap_recon_search_path: Path to search

    Returns:
        A list of dicts with the image stem as key and the 3x4 projection matrix as value.
    """
    import pycolmap
    result = []
    for images_bin_path in list(colmap_recon_search_path.glob('**/images.bin')):
        cameras_bin_path = images_bin_path.parent/'cameras.bin'
        points3D_bin_path = images_bin_path.parent/'points3D.bin'
        if cameras_bin_path.exists() and points3D_bin_path.exists():
            print(f'found colmap sparse recon in {str(images_bin_path.parent)}')
            recon = pycolmap.Reconstruction(images_bin_path.parent)

            imname_P = {}
            for i,im in recon.images.items():
                R = im.rotation_matrix()
                t = im.tvec
                cam = recon.cameras[im.camera_id]
                K = cam.calibration_matrix()
                Rt = np.concatenate((R,t[...,None]), axis=-1)
                P = K@Rt
                imname_P[Path(im.name).stem] = P
            result.append(imname_P)
    
    return result


def update_input_keypoint_paths(input_keypoints_data: Dict[str, Dict[Path, Dict]], image_paths: List[Path]):
    """Returns a new dict with paths updated from image_paths using the unique image filename
    Args:
        input_keypoints_data: Dict with the keypoints and measurements in one or more images.
        image_paths: List of image paths.
    Returns:
        Dict of the same format as input_keypoint_data with updated paths.
    """
    image_filename_to_path = {x.name: x for x in image_paths}
    result = {}
    for kp_name, kp in input_keypoints_data.items():
        measurements = {}
        for key in kp.keys():
            if not key in image_paths and key.name in image_filename_to_path:
                measurements[image_filename_to_path[key.name]] = kp[key]
            elif not key in image_paths:
                assert key in image_paths, f"key '{key}' from the input file is not in the list of image paths!"
            else:
                measurements[key] = kp[key]
        result[kp_name] = measurements
    return result


def main():

    parser = argparse.ArgumentParser(
        description="Tool for annotating multiple images with keypoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument( 'images', nargs='+', type=Path, help="Paths to the images or a single directory for an object.")
    parser.add_argument("--output", type=Path, default=Path('object_keypoints.json'), help="Output json file.")
    parser.add_argument("--input", type=Path, help="Input json file.")
    parser.add_argument("--recon_search_path", type=Path, help="A directory to search for colmap reconstructions.")
    parser.add_argument("--update_paths_in_input_json", action='store_true', help="If set this will update the paths in the input keypoint file based on the image filename. Use this when the image directory has been renamed.")

    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    print(args)

    projection_matrices = None
    if args.recon_search_path is not None:
        projection_matrices = load_projection_matrices(args.recon_search_path)
        compute_keypoint_guess.projection_matrices = projection_matrices

    if len(args.images) == 1 and args.images[0].is_dir():
        images = load_object_dir(args.images[0])
    else:
        images = args.images

    input_keypoints_data = None
    if args.input:
        with open(args.input, 'r') as f:
            input_keypoints_data = json.load(f)
            input_keypoints_data = {kp_name : {Path(k): v for k,v in x.items()} for kp_name, x in input_keypoints_data.items()}
            if args.update_paths_in_input_json:
                input_keypoints_data = update_input_keypoint_paths(input_keypoints_data, images)
            for _, kp in input_keypoints_data.items():
                for key in kp.keys():
                    assert key in images, f"key '{key}' from the input file is not in the list of image paths!"

    images = load_images(images)

    root = Tk()
    root.title('Correspondence Annotation Tool     |     q,comma: prev image,  w,period: next image,  e: prev selected image')
    root.grid()

    imagelist = ImageList(root, [path for path, _ in images])
    imagelist.mainwidget.grid(row=0, column=0, sticky="NWSE")
    imagelist._projection_matrices = projection_matrices

    annotation_area = AnnotationArea(root, )
    annotation_area.mainwidget.grid(column=1, row=0, rowspan=2, sticky="NWSE")

    keypointlist = KeypointList(root, input_keypoints_data)
    keypointlist.mainwidget.grid(row=1, column=0, sticky="NWSE")

    otherimagespanel = OtherImagesPanel(root, images)
    otherimagespanel.mainwidget.grid(row=0, column=2, rowspan=3, sticky="NWSE")

    annotation_area.set_keypoints(keypointlist._keypoints)
    imagelist._keypoints = keypointlist.keypoints

    def keypointlist_change(idxs, keypoints):
        annotation_area.set_keypoints(keypoints)
        if len(idxs):
            annotation_area.set_active_keypoint_idx(idxs[0])
            otherimagespanel.set_keypoint(keypoints[idxs[0]], annotation_area.current_image)
            imagelist.set_current_keypoint(keypoints[idxs[0]])
            imagelist.update_list()


    def select_image(idx):
        print('select_image', idx)
        if idx < len(images):
            keypointlist.current_image = images[idx][0]
            annotation_area.set_image(images[idx])
            otherimagespanel.set_keypoint(annotation_area.active_keypoint, annotation_area.current_image)
            imagelist.update_list()

    def keypoint_pos_set(x,y):
        imagelist.update_list()

    annotation_area.set_click_callback(keypoint_pos_set)
    imagelist.set_change_callback(select_image)
    keypointlist.set_change_callback(keypointlist_change)
    
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=30)
    root.grid_columnconfigure(2, weight=4)
    root.grid_rowconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=1)
    
    # call once after the window is visible
    def init_callback(event):
        select_image(0)
        keypointlist._change(None)
        root.unbind('<Visibility>', init_callback.idstr)
    init_callback.idstr = root.bind('<Visibility>', init_callback)
    
    # some convenience shortcuts
    def goto_next_image(event):
        export_keypoints_to_json(args.output, keypointlist.keypoints, is_autosave=True)
        if event.keysym in ('e'): # goto previously selected image
            idx = imagelist._listbox.get_previous_selection()
            if idx:
                print('goto',idx)
                idx = idx[0]
            else:
                return
        else:
            if event.keysym in ('w', 'period'):
                delta = 1
            else: # comma, q
                delta = -1
            print( event, event.keysym, type(event.keysym))
            idx = (imagelist.current_index + delta) % len(images)
        imagelist._listbox.select(idx)
        select_image(idx)

    # root.bind('<Key-space>', goto_next_image) # may trigger one of the buttons in the lower left corner
    root.bind('<Key-q>', goto_next_image)
    root.bind('<Key-comma>', goto_next_image)
    root.bind('<Key-w>', goto_next_image)
    root.bind('<Key-period>', goto_next_image)
    root.bind('<Key-e>', goto_next_image)
    
    def select_keypoint(event):
        print(event)
        export_keypoints_to_json(args.output, keypointlist.keypoints, is_autosave=True)
        kp_name = str(event.keysym)
        if kp_name == 'grave':
            kp_name = '0'
        for idx, kp in enumerate(keypointlist.keypoints):
            if kp.name == kp_name:
                keypointlist._listbox.select(idx)
                keypointlist_change((idx,), keypointlist.keypoints)
                return
    # disbale keypoint shirtcuts due to a conflict with the zoom level spin box
    # for i in range(10):
        # root.bind(f'<Key-{i}>', select_keypoint)
    # root.bind(f'<Key-grave>', select_keypoint)
    
    root.mainloop()

    export_keypoints_to_json(args.output, keypointlist.keypoints)

    
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()
