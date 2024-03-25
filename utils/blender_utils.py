# SPDX-License-Identifier: Apache-2.0
import bpy
from bpy import data as D
from bpy import context as C
from bpy_extras.io_utils import ExportHelper
from bpy.props import StringProperty, BoolProperty, EnumProperty
from bpy.types import Operator
from mathutils import *
import re
import numpy as np
from pathlib import Path
import shutil


def delete_all_objects():
    """Helper function for deleting all objects in the scene."""
    # delete everything
    for obj in C.scene.objects:
        obj.select_set(True)
        bpy.ops.object.delete()


def create_camera(name, npz_path):
    """Creates a camera object from an npz file."""
    data = np.load(npz_path)

    T = data['T']
    R = T[:3, :3]
    t = T[:3, 3]
    cvcam = CVCamera(K=data['K'],
                     R=R,
                     t=t,
                     height_px=data['height'],
                     width_px=data['width'])

    bpy.ops.object.camera_add()
    cam = bpy.context.object
    cvcam.apply_to_blender(cam)
    cam['image_name'] = data['name']
    cam['image_id'] = data['id']
    cam['original_intrinsics'] = data['K'].reshape(-1)
    cam.name = name
    return cam


def create_mesh(name, npz_path, save_images=False):
    """Creates a mesh object from an npz file and sets up the pbr material nodes
    Args:
        name: Name of the new mesh
        npz_path: Path to the npz file
        save_images: If True save images to the current workdir
    
    """
    numpy_mesh = np.load(npz_path, allow_pickle=False)

    num_vertices = numpy_mesh['vertices'].shape[0]
    num_loops = numpy_mesh['triangles'].shape[0]
    loop_start = np.arange(num_loops, dtype=np.int32) * 3
    loop_total = np.full((num_loops,), 3, dtype=np.int32)

    mesh = bpy.data.meshes.new(name=name)

    mesh.vertices.add(num_vertices)
    mesh.vertices.foreach_set("co", numpy_mesh['vertices'].ravel())

    mesh.loops.add(
        num_loops *
        3)  # this is the size of the index array not the number of triangles
    mesh.loops.foreach_set("vertex_index", numpy_mesh['triangles'].ravel())

    mesh.polygons.add(num_loops)
    mesh.polygons.foreach_set("loop_start", loop_start)
    mesh.polygons.foreach_set("loop_total", loop_total)

    if 'base_color' in numpy_mesh:
        vertex_colors = mesh.vertex_colors.new(name='base_color')
        # vertex colors are rgba and per triangle index
        if numpy_mesh['base_color'].shape[-1] == 3:
            base_color = np.concatenate([
                numpy_mesh['base_color'],
                np.ones((num_vertices, 1), dtype=np.float32)
            ],
                                        axis=-1)
        else:
            base_color = numpy_mesh['base_color']
        base_color = base_color[numpy_mesh['triangles'].ravel()]
        vertex_colors.data.foreach_set('color', base_color.ravel())

    if 'metallic' in numpy_mesh and numpy_mesh['metallic'].ndim > 0:
        vertex_metallic = mesh.vertex_colors.new(name='metallic')
        if numpy_mesh['metallic'].ndim == 1:
            metallic = np.concatenate([
                np.zeros((num_vertices, 3), dtype=np.float32),
                numpy_mesh['metallic'][:, None]
            ],
                                      axis=-1)
        else:
            metallic = numpy_mesh['metallic']
        metallic = metallic[numpy_mesh['triangles'].ravel()]
        vertex_metallic.data.foreach_set('color', metallic.ravel())


    if 'roughness' in numpy_mesh and numpy_mesh['roughness'].ndim > 0:
        vertex_roughness = mesh.vertex_colors.new(name='roughness')
        if numpy_mesh['roughness'].ndim == 1:
            roughness = np.concatenate([
                np.zeros((num_vertices, 3), dtype=np.float32),
                numpy_mesh['roughness'][:, None]
            ],
                                       axis=-1)
        else:
            roughness = numpy_mesh['roughness']
        roughness = roughness[numpy_mesh['triangles'].ravel()]
        vertex_roughness.data.foreach_set('color', roughness.ravel())

    if 'vertexfeats' in numpy_mesh:
        for ch in range(numpy_mesh['vertexfeats'].shape[-1]):
            float_layer = mesh.vertex_layers_float.new(
                name='vertexfeats{}'.format(ch))
            channel = numpy_mesh['vertexfeats'][..., ch]
            float_layer.data.foreach_set('value', channel.ravel())

    # old files use 'features' instead of 'vertexfeats'
    if 'features' in numpy_mesh:
        for ch in range(numpy_mesh['features'].shape[-1]):
            float_layer = mesh.vertex_layers_float.new(
                name='vertexfeats{}'.format(ch))
            channel = numpy_mesh['features'][..., ch]
            float_layer.data.foreach_set('value', channel.ravel())

    if 'uvmap' in numpy_mesh:
        uv_layer = mesh.uv_layers.new()
        uv_layer.data.foreach_set(
            'uv', numpy_mesh['uvmap'].ravel().astype(np.float32))

    for k, arr in numpy_mesh.items():
        if k in ('vertexfeats', 'base_color', 'metallic', 'roughness',
                 'envfeats', 'features', 'uvmap'):
            continue

        if arr.ndim == 1 and arr.shape[0] == num_vertices:
            float_layer = mesh.vertex_layers_float.new(name=k)
            float_layer.data.foreach_set('value', arr)
        elif arr.ndim == 2 and arr.shape[0] == num_vertices and arr.shape[
                1] == 4:
            vertex_colors = mesh.vertex_colors.new(name=k)
            arr_loop = arr[numpy_mesh['triangles'].ravel()]
            vertex_colors.data.foreach_set('color', arr_loop.ravel())

    mesh.update()
    mesh.validate()

    obj = bpy.data.objects.new(name, mesh)

    # attach envfeats to the object
    if 'envfeats' in numpy_mesh:
        obj['envfeats'] = numpy_mesh['envfeats']

    if 'matrix_obj2world' in numpy_mesh:
        obj.matrix_world = Matrix(numpy_mesh['matrix_obj2world'])

    collection = bpy.context.collection
    collection.objects.link(obj)

    # setup materials
    # Deselect all
    bpy.ops.object.select_all(action='DESELECT')

    mat = bpy.data.materials.new("material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    mat_node = nodes['Material Output']
    bsdf_node = nodes.new('ShaderNodeBsdfPrincipled')
    mat.node_tree.links.new(bsdf_node.outputs['BSDF'],
                            mat_node.inputs['Surface'])

    if 'metallic' in numpy_mesh:
        if numpy_mesh['metallic'].ndim > 0:
            node_metal = nodes.new("ShaderNodeVertexColor")
            node_metal.layer_name
            node_metal.layer_name = 'metallic'
            mat.node_tree.links.new(node_metal.outputs['Alpha'],
                                    bsdf_node.inputs['Metallic'])
        else:
            bsdf_node.inputs['Metallic'].default_value = numpy_mesh['metallic'].item()

    if 'roughness' in numpy_mesh:
        if numpy_mesh['roughness'].ndim > 0:
            node_rough = nodes.new("ShaderNodeVertexColor")
            node_rough.layer_name
            node_rough.layer_name = 'roughness'
            mat.node_tree.links.new(node_rough.outputs['Alpha'],
                                    bsdf_node.inputs['Roughness'])
        else:
            bsdf_node.inputs['Roughness'].default_value = numpy_mesh['roughness'].item()

    if 'base_color' in numpy_mesh:
        node_base = nodes.new("ShaderNodeVertexColor")
        node_base.layer_name
        node_base.layer_name = 'base_color'
        mat.node_tree.links.new(node_base.outputs['Color'],
                                bsdf_node.inputs['Base Color'])

    if 'tex_albedo' in numpy_mesh:
        filename = 'albedo.jpg'
        tex = numpy_mesh['tex_albedo']
        if tex.shape[-1] == 3:
            tex = np.concatenate([tex, np.ones_like(tex[...,:1])], axis=-1)
        tex = np.flip(tex, axis=0) # flip v axis
        img = bpy.data.images.new(filename, width=tex.shape[1], height=tex.shape[0])
        img.filepath = filename
        img.colorspace_settings.name = 'sRGB'
        img.pixels = tex.ravel()
        img.file_format = 'JPEG'
        if save_images:
            img.save()
        node_tex = nodes.new("ShaderNodeTexImage")
        node_tex.image = img
        mat.node_tree.links.new(node_tex.outputs['Color'], bsdf_node.inputs['Base Color'])
        
    if 'tex_roughness' in numpy_mesh:
        filename = 'roughness.jpg'
        tex = numpy_mesh['tex_roughness']
        if tex.shape[-1] == 1:
            tex = np.tile(tex, (1,1,4))
            tex[...,-1] = 1
        tex = np.flip(tex, axis=0) # flip v axis
        img = bpy.data.images.new(filename, width=tex.shape[1], height=tex.shape[0])
        img.filepath = filename
        img.colorspace_settings.name = 'Raw' # 'Linear' does not work
        img.pixels = tex.ravel()
        img.file_format = 'JPEG'
        if save_images:
            img.save()
        node_tex = nodes.new("ShaderNodeTexImage")
        node_tex.image = img
        # node_split_rgb = nodes.new("ShaderNodeSeparateColor")
        # mat.node_tree.links.new(node_tex.outputs['Color'], node_split_rgb.inputs['Color'])
        # mat.node_tree.links.new(node_split_rgb.outputs['Red'], bsdf_node.inputs['Roughness'])
        mat.node_tree.links.new(node_tex.outputs['Color'], bsdf_node.inputs['Roughness'])
        
    if 'tex_metallic' in numpy_mesh:
        filename = 'metallic.jpg'
        tex = numpy_mesh['tex_metallic']
        if tex.shape[-1] == 1:
            tex = np.tile(tex, (1,1,4))
            tex[...,-1] = 1
        tex = np.flip(tex, axis=0) # flip v axis
        img = bpy.data.images.new(filename, width=tex.shape[1], height=tex.shape[0])
        img.filepath = filename
        img.colorspace_settings.name = 'Raw' # 'Linear' does not work
        img.pixels = tex.ravel()
        img.file_format = 'JPEG'
        if save_images:
            img.save()
        node_tex = nodes.new("ShaderNodeTexImage")
        node_tex.image = img
        # node_split_rgb = nodes.new("ShaderNodeSeparateColor")
        # mat.node_tree.links.new(node_tex.outputs['Color'], node_split_rgb.inputs['Color'])
        # mat.node_tree.links.new(node_split_rgb.outputs['Red'], bsdf_node.inputs['Metallic'])
        mat.node_tree.links.new(node_tex.outputs['Color'], bsdf_node.inputs['Metallic'])
        
    if 'tex_normal' in numpy_mesh:
        filename = 'normal.jpg'
        tex = numpy_mesh['tex_normal']
        if tex.shape[-1] == 3:
            tex = np.concatenate([tex, np.ones_like(tex[...,:1])], axis=-1)
        tex = np.flip(tex, axis=0) # flip v axis
        img = bpy.data.images.new(filename, width=tex.shape[1], height=tex.shape[0])
        img.filepath = filename
        img.colorspace_settings.name = 'Raw' # 'Linear' does not work
        img.pixels = tex.ravel()
        img.file_format = 'JPEG'
        if save_images:
            img.save()
        node_tex = nodes.new("ShaderNodeTexImage")
        node_tex.image = img
        node_normal_map = nodes.new("ShaderNodeNormalMap")
        mat.node_tree.links.new(node_tex.outputs['Color'], node_normal_map.inputs['Color'])
        mat.node_tree.links.new(node_normal_map.outputs['Normal'], bsdf_node.inputs['Normal'])
        
    
        
    
    obj.select_set(True)
    bpy.ops.object.shade_smooth()

    # set normals after shade_smooth()
    if 'vertex_normals' in numpy_mesh:
        mesh.use_auto_smooth = True # needs to be activated for setting normals
        mesh.normals_split_custom_set_from_vertices(numpy_mesh['vertex_normals'])

    return obj


def mesh_to_numpy(obj, apply_transform=True):
    result = {}

    if apply_transform:
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    else:
        matrix_obj2world = np.asarray(obj.matrix_world)
        result['matrix_obj2world'] = matrix_obj2world

    mesh = obj.data
    num_vertices = len(mesh.vertices)
    vertices = np.empty(num_vertices * 3, np.float32)
    mesh.vertices.foreach_get('co', vertices)
    result['vertices'] = vertices.reshape(-1, 3)

    normals = np.empty(num_vertices * 3, np.float32)
    mesh.vertices.foreach_get('normal', normals)
    result['vertex_normals'] = normals.reshape(-1, 3)

    num_loops = len(mesh.loops)
    loops = np.empty(num_loops, dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loops)

    num_polys = len(mesh.polygons)
    loop_start = np.empty(num_polys, dtype=np.int32)
    loop_total = np.empty(num_polys, dtype=np.int32)
    mesh.polygons.foreach_get("loop_start", loop_start)
    mesh.polygons.foreach_get("loop_total", loop_total)
    assert all(loop_total == 3), 'not all faces are triangles!'

    triangles = np.empty((num_polys, 3), dtype=np.int32)
    triangles[:, 0] = loops[loop_start]
    triangles[:, 1] = loops[loop_start + 1]
    triangles[:, 2] = loops[loop_start + 2]
    result['triangles'] = triangles.astype(np.int64)

    float_layers = {}
    for fl in mesh.vertex_layers_float:
        if fl.name.startswith('vertexfeats'):
            arr = np.empty(len(fl.data), dtype=np.float32)
            fl.data.foreach_get('value', arr)
            float_layers[fl.name] = arr
    if float_layers:
        result['vertexfeats'] = np.stack([
            float_layers['vertexfeats{}'.format(i)]
            for i in range(len(float_layers))
        ],
                                         axis=-1)

    # other float layers
    float_layers = {}
    for fl in mesh.vertex_layers_float:
        if not fl.name.startswith('vertexfeats'):
            arr = np.empty(len(fl.data), dtype=np.float32)
            fl.data.foreach_get('value', arr)
            result[fl.name] = arr

    # color layers
    for cl in mesh.vertex_colors:
        # vertex colors are rgba and per triangle index
        arr = np.empty(num_loops * 4, dtype=np.float32)
        cl.data.foreach_get('color', arr)
        arr = arr.reshape(-1, 4)
        vertex_color = np.empty((num_vertices, 4), dtype=np.float32)
        vertex_color[loops] = arr
        result[cl.name] = vertex_color

    if 'envfeats' in obj:
        result['envfeats'] = np.array(obj['envfeats'])

    # uvmap
    if len(obj.data.uv_layers):
        uvmap = obj.data.uv_layers[0]
        uvmap_arr = np.zeros(len(uvmap.data) * 2, dtype=np.float64)
        uvmap.data.foreach_get('uv', uvmap_arr)
        uvmap_arr = uvmap_arr.reshape(-1, 3, 2)
        result['uvmap'] = uvmap_arr.astype(np.float32)

    return result


class CVCamera:
    """Standard computer vision pinhole camera"""

    def __init__(self, K, R, t, height_px=960, width_px=1280):
        self._K = K
        self._R = R
        self._t = t
        self._height_px = height_px
        self._width_px = width_px

    @property
    def K(self):
        return self._K

    @property
    def R(self):
        return self._R

    @property
    def t(self):
        return self._t

    @property
    def T(self):
        ans = np.eye(4)
        ans[:3, :3] = self.R
        ans[:3, 3] = self.t
        return ans

    @property
    def width_px(self):
        return self._width_px

    @property
    def height_px(self):
        return self._height_px

    @property
    def horizontal_fov(self):
        return 2 * np.arctan(0.5 * self._width_px / self._K[0, 0])

    @property
    def vertical_fov(self):
        return 2 * np.arctan(0.5 * self._height_px / self._K[1, 1])

    @staticmethod
    def from_blender(blender_camera_obj, height=None, width=None):
        cam = blender_camera_obj
        assert cam.data.type == 'PERSP'
        assert cam.data.lens_unit == 'FOV'
        assert cam.data.sensor_fit == 'HORIZONTAL'
        width = bpy.context.scene.render.resolution_x if width is None else width
        height = bpy.context.scene.render.resolution_y if height is None else height
        fov = cam.data.angle
        K = np.eye(3)
        K[0, 0] = width / (2 * np.tan(0.5 * fov))
        K[1, 1] = K[0, 0]
        K[0, 2] = width / 2
        K[1, 2] = height / 2
        T = np.linalg.inv(np.asarray(cam.matrix_world))
        R_bcam2cv = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        R = R_bcam2cv @ T[:3, :3]
        t = T[:3, 3] @ R_bcam2cv
        return CVCamera(K=K, R=R, t=t, height_px=height, width_px=width)

    def apply_to_blender(self, blender_camera_obj):
        cam = blender_camera_obj
        cam.data.type = 'PERSP'
        cam.data.lens_unit = 'FOV'
        cam.data.angle = self.horizontal_fov
        cam.data.sensor_fit = 'HORIZONTAL'

        R_bcam2cv = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        R_cv2world = self._R.T
        rot = R_cv2world @ R_bcam2cv
        rot = Matrix(rot.tolist())
        loc = (-R_cv2world @ self._t.reshape(3, 1)).ravel()
        loc = Matrix.Translation(loc)
        cam.matrix_world = loc @ rot.to_4x4()

    def __str__(self):
        return "K\n{}\n\nR\n{}\n\nt\n{}\nwidth_px {},  height_px {}".format(
            self._K, self._R, self._t, self._width_px, self._height_px)


class ExportMeshNPZ(Operator, ExportHelper):
    """Exports a mesh in npz format"""
    bl_idname = "export_dsvbrdf.mesh_npz"
    bl_label = "Export Mesh in npz format"

    # ExportHelper mixin class uses this
    filename_ext = ".npz"

    filter_glob: StringProperty(
        default="*.npz",
        options={'HIDDEN'},
        maxlen=255,  # Max internal buffer length, longer would be clamped.
    )

    apply_transforms: BoolProperty(
        name="Apply transforms",
        description="If checked apply all transforms to the mesh",
        default=True,
    )

    def execute(self, context):
        filepath = Path(self.filepath)
        obj = context.object
        numpy_mesh = mesh_to_numpy(obj, self.apply_transforms)
        np.savez_compressed(filepath, **numpy_mesh)
        return {'FINISHED'}

    def menu_func_export(self, context):
        self.layout.operator(ExportMeshNPZ.bl_idname,
                             text="DSVBRDF Export Mesh")

    @classmethod
    def register_op(cls):
        bpy.utils.register_class(cls)
        bpy.types.TOPBAR_MT_file_export.append(cls.menu_func_export)


class ExportCamerasNPZ(Operator, ExportHelper):
    """Exports cameras in npz format for all frames"""
    bl_idname = "export_dsvbrdf.cameras_npz"
    bl_label = "Export Cameras in npz format"

    # ExportHelper mixin class uses this
    filename_ext = ".npz"

    filter_glob: StringProperty(
        default="*.npz",
        options={'HIDDEN'},
        maxlen=255,  # Max internal buffer length, longer would be clamped.
    )

    def execute(self, context):
        filepath = Path(self.filepath)
        cam = context.scene.camera
        frame_current = context.scene.frame_current
        for frame_i in range(context.scene.frame_start,
                             context.scene.frame_end + 1):
            context.scene.frame_set(frame_i)
            out_filepath = filepath.parent / (filepath.stem +
                                              '_{:03d}.npz'.format(frame_i))
            cvcam = CVCamera.from_blender(cam)
            # print(cvcam)
            np.savez(out_filepath,
                     K=cvcam.K,
                     T=cvcam.T,
                     width=cvcam.width_px,
                     height=cvcam.height_px)

        context.scene.frame_set(frame_current)
        return {'FINISHED'}

    def menu_func_export(self, context):
        self.layout.operator(ExportCamerasNPZ.bl_idname,
                             text="DSVBRDF Export Camera Animation")

    @classmethod
    def register_op(cls):
        bpy.utils.register_class(cls)
        bpy.types.TOPBAR_MT_file_export.append(cls.menu_func_export)


class ExportDatasetNPZ(Operator, ExportHelper):
    """Exports a dataset with a mesh and cameras in npz format"""
    bl_idname = "export_dsvbrdf.dataset"
    bl_label = "Export dataset to folder"

    # ExportHelper mixin class uses this
    filename_ext = ""
    use_filter_folder = True

    filter_glob: StringProperty(
        default="",
        options={'HIDDEN'},
        maxlen=255,  # Max internal buffer length, longer would be clamped.
    )

    write_images: BoolProperty(
        name="Write images",
        description="If checked copy images to the new dataset dir",
        default=True,
    )

    def execute(self, context):
        filepath = Path(self.filepath)

        mesh = None
        cams = []
        images = []
        for k, v in bpy.context.scene.objects.items():
            if isinstance(v.data, bpy.types.Camera) and 'original_path' in v:
                cams.append(v)
                idx = re.match('.*_(\d+)\.npz',
                               Path(v['original_path']).name).group(1)
                images.append(
                    Path(v['original_path']).parent / f'image_{idx}.png')
            elif isinstance(v.data, bpy.types.Mesh):
                mesh = v

        filepath.parent.mkdir(parents=True, exist_ok=True)

        for cam, img in zip(cams, images):
            cvcam = CVCamera.from_blender(cam)
            K = np.array(cam['original_intrinsics']).reshape(3, 3).astype(
                np.float32)
            out_filepath = filepath.parent / Path(cam['original_path']).name
            np.savez(out_filepath,
                     K=K,
                     T=cvcam.T,
                     width=cvcam.width_px,
                     height=cvcam.height_px)

            if self.write_images:
                out_filepath = filepath.parent / img.name
                if not out_filepath.exists() or not out_filepath.samefile(img):
                    shutil.copy2(img, out_filepath)

        numpy_mesh = mesh_to_numpy(mesh)
        np.savez_compressed(filepath.parent / 'mesh.npz', **numpy_mesh)
        return {'FINISHED'}

    def menu_func_export(self, context):
        self.layout.operator(ExportDatasetNPZ.bl_idname,
                             text="DSVBRDF Dataset Export")

    @classmethod
    def register_op(cls):
        bpy.utils.register_class(cls)
        bpy.types.TOPBAR_MT_file_export.append(cls.menu_func_export)


class OrientAndScaleScene(bpy.types.Operator):
    """Scales and orients the scene based on the cameras and the mesh"""
    bl_idname = "scene.orient_ans_scale_scene"
    bl_label = "Orient and scale scene"

    def execute(self, context):
        mesh = None
        cams = []
        for k, v in bpy.context.scene.objects.items():
            if isinstance(v.data, bpy.types.Camera) and 'original_path' in v:
                cams.append(v)
            elif isinstance(v.data, bpy.types.Mesh):
                mesh = v
        cam_locations = np.array([c.location for c in cams])

        # fix rotation
        upvec = robust_normal_estimation(cam_locations)
        cam_upvec_sum = np.zeros(shape=(3,), dtype=np.float64)
        for cam in cams:
            cam_upvec = np.array(cam.matrix_world)[:3, 1]
            cam_upvec_sum += cam_upvec
        if np.dot(upvec, cam_upvec_sum) < 0:
            upvec = -upvec

        xvec = Vector(upvec).orthogonal().normalized()
        yvec = Vector(upvec).cross(xvec).normalized()
        R = np.array([xvec, yvec, upvec])
        T = np.eye(4)
        T[:3, :3] = R
        mesh.matrix_world = (T @ np.array(mesh.matrix_world)).T
        for cam in cams:
            cam.matrix_world = (T @ np.array(cam.matrix_world)).T

        bpy.ops.object.select_all(action='DESELECT')
        mesh.select_set(True)
        bpy.context.view_layer.objects.active = mesh
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        # fix scale and translation
        mesh_bb = np.array(mesh.bound_box)
        mesh_height = mesh_bb.max(axis=0)[2] - mesh_bb.min(axis=0)[2]

        # scale such that the height of the mesh is 2
        scale = Matrix.Identity(4)
        scale[0][0] = 2 / mesh_height
        scale[1][1] = 2 / mesh_height
        scale[2][2] = 2 / mesh_height

        mesh.matrix_world = scale @ mesh.matrix_world
        for cam in cams:
            cam.location = scale[0][0] * np.array(cam.location)

        mesh_bb = np.array(mesh.bound_box)
        mesh_height = mesh_bb.max(axis=0)[2] - mesh_bb.min(axis=0)[2]
        mesh_bb_center = 0.5 * (mesh_bb.max(axis=0) + mesh_bb.min(axis=0))

        # translate
        translation = Vector(-mesh_bb_center * scale[0][0])
        mesh.location += translation
        for cam in cams:
            cam.location += translation

        bpy.ops.object.select_all(action='DESELECT')
        mesh.select_set(True)
        bpy.context.view_layer.objects.active = mesh
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        return {'FINISHED'}

    @classmethod
    def register_op(cls):
        bpy.utils.register_class(cls)


class UtilsPanel(bpy.types.Panel):
    """Creates a Panel in the scene context of the properties editor"""
    bl_label = "DSVBRDF Utils"
    bl_idname = "SCENE_PT_dsvbrdf"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "scene"

    def draw(self, context):
        layout = self.layout

        scene = context.scene

        layout.label(text="Operators:")
        row = layout.row()
        row.scale_y = 3.0
        row.operator(OrientAndScaleScene.bl_idname)

        row = layout.row()
        row.scale_y = 3.0
        row.operator(ExportDatasetNPZ.bl_idname)

        # these don't work in the context of the panel
        # row = layout.row()
        # row.operator(ExportMeshNPZ.bl_idname)

        # row = layout.row()
        # row.operator(ExportCamerasNPZ.bl_idname)

    @classmethod
    def register_panel(cls):
        bpy.utils.register_class(cls)

class SavePosePriors(bpy.types.Operator):
    """Saves the camera poses to a file that can be used as priors."""
    bl_idname = "scene.save_pose_priors"
    bl_label = "Save pose priors"

    def execute(self, context):
        cams = []
        for k, v in bpy.context.scene.objects.items():
            if isinstance(v.data, bpy.types.Camera) and 'image_id' in v:
                cams.append(v)
        
        im_path = Path(cams[0]['image_path'])
        envdir = im_path.parent.parent
        objdir = envdir.parent
        envname = envdir.name
        objname = objdir.name
        # compute the corresponding source path
        source_path = objdir.parent.parent/'source_data'/objname/envname

        data = {}
        for cam in cams:
            cvcam = CVCamera.from_blender(cam, cam['image_height'], cam['image_width'])
            prefix = cam['image_name']
            for k in ('R','t', 'width_px', 'height_px'):
                data[f'{prefix}:{k}'] = getattr(cvcam,k)
            K = np.array(cam['original_intrinsics']).reshape(3, 3).astype( np.float32)
            data[f'{prefix}:K'] = K
            
        np.savez(source_path/'pose_priors.npz', **data)

        return {'FINISHED'}

    @classmethod
    def register_op(cls):
        bpy.utils.register_class(cls)




class UtilsPanel(bpy.types.Panel):
    """Creates a Panel in the scene context of the properties editor"""
    bl_label = "Object Relighting Dataset Utils"
    bl_idname = "SCENE_PT_ord"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "scene"

    def draw(self, context):
        layout = self.layout

        scene = context.scene

        layout.label(text="Operators:")
        row = layout.row()
        row.scale_y = 3.0
        row.operator(SavePosePriors.bl_idname)


    @classmethod
    def register_panel(cls):
        bpy.utils.register_class(cls)
