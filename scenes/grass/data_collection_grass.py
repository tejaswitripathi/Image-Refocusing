import bpy
import os
import json
import random
import math
import numpy as np
from mathutils import Vector

# ----------------------------
# CONFIG
# ----------------------------

random.seed(42)

f_stops = [1.2, 1.4, 1.8, 2.0, 2.8, 4.0, 5.6, 6.3, 7.1, 8.0, 11.0, 16.0, 22.0]
focal_lengths = np.linspace(24, 135, 10).tolist()
focus_distances = np.linspace(2.799, 75.67, 10).tolist()  # meters

datadir = "//dataset/"  # relative to .blend file
sensor_width_mm = 35.0  # full-frame width

resolution_x = 512
resolution_y = 512
samples = 64

camera = bpy.context.scene.camera
scene = bpy.context.scene

# ----------------------------
# BASIC RENDER SETTINGS
# ----------------------------

scene.render.engine = "CYCLES"
scene.cycles.samples = samples
scene.cycles.use_denoising = True

scene.render.resolution_x = resolution_x
scene.render.resolution_y = resolution_y
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"

camera.data.sensor_width = sensor_width_mm

# ----------------------------
# HELPERS
# ----------------------------

def ensure_dir(path):
    os.makedirs(bpy.path.abspath(path), exist_ok=True)

def object_world_center(obj):
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return sum(corners, Vector()) / 8

def dist_to_camera(obj):
    return (object_world_center(obj) - camera.location).length

def get_candidate_objects(start, end):
    candidates = []

    for obj in scene.objects:
        if obj.type != "MESH":
            continue

        if obj.hide_render:
            continue

        d = dist_to_camera(obj)

        if start <= d < end:
            candidates.append(obj)

    return candidates

def assign_object_indices():
    idx = 1
    mapping = {}

    for obj in scene.objects:
        if obj.type == "MESH":
            obj.pass_index = idx
            mapping[obj.name] = idx
            idx += 1

    return mapping

def render_png(filepath):
    scene.render.filepath = filepath
    scene.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=True)

def setup_depth_nodes(output_dir):
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    render_layers = tree.nodes.new(type="CompositorNodeRLayers")

    depth_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_output.label = "Depth Output"
    depth_output.base_path = bpy.path.abspath(output_dir)
    depth_output.file_slots[0].path = "depth_"
    depth_output.format.file_format = "OPEN_EXR"
    depth_output.format.color_depth = "32"
    depth_output.format.color_mode = "BW"

    tree.links.new(render_layers.outputs["Depth"], depth_output.inputs[0])

def setup_mask_nodes(output_dir, subject_index):
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    render_layers = tree.nodes.new(type="CompositorNodeRLayers")

    id_mask = tree.nodes.new(type="CompositorNodeIDMask")
    id_mask.index = subject_index

    mask_output = tree.nodes.new(type="CompositorNodeOutputFile")
    mask_output.label = "Mask Output"
    mask_output.base_path = bpy.path.abspath(output_dir)
    mask_output.file_slots[0].path = "subject_mask_"
    mask_output.format.file_format = "PNG"
    mask_output.format.color_mode = "BW"
    mask_output.format.color_depth = "8"

    tree.links.new(render_layers.outputs["IndexOB"], id_mask.inputs[0])
    tree.links.new(id_mask.outputs["Alpha"], mask_output.inputs[0])

def disable_nodes():
    scene.use_nodes = False

# ----------------------------
# ENABLE PASSES
# ----------------------------

view_layer = bpy.context.view_layer
view_layer.use_pass_z = True
view_layer.use_pass_object_index = True

object_index_map = assign_object_indices()

# ----------------------------
# MAIN LOOP
# ----------------------------

print("Camera:", camera.name, camera.location)

mesh_distances = []

base_camera_location = camera.location.copy()

for obj in scene.objects:
    if obj.type == "MESH" and not obj.hide_render:
        d = dist_to_camera(obj)
        mesh_distances.append((obj.name, d))

mesh_distances = sorted(mesh_distances, key=lambda x: x[1])

print("Closest 20 objects:")
for name, d in mesh_distances[:20]:
    print(f"{name}: {d:.2f} m")

print("Farthest 20 objects:")
for name, d in mesh_distances[-20:]:
    print(f"{name}: {d:.2f} m")

ensure_dir(datadir)

img_i = 0

depth_bins = []
prev = 0.0
for fd in focus_distances:
    depth_bins.append((prev, fd))
    prev = fd

for start, end in depth_bins:

    candidates = get_candidate_objects(start, end)

    if not candidates:
        print(f"No objects found in depth range {start:.2f}m to {end:.2f}m")
        continue

    subject = random.choice(candidates)
    subject_index = subject.pass_index

    for focal_length in focal_lengths:

        fl_index = focal_lengths.index(focal_length)

        camera.location = base_camera_location.copy()
        camera.data.shift_y = 0.1 * fl_index

        curr_focus_distance = dist_to_camera(subject)

        for f_stop in f_stops:

            folder = os.path.join(
                datadir,
                f"img_{img_i:05d}_f{f_stop}_fl{focal_length}_fd{curr_focus_distance:.2f}"
            )
            ensure_dir(folder)

            camera.data.lens = focal_length
            camera.data.sensor_width = sensor_width_mm
            camera.data.dof.use_dof = True
            camera.data.dof.focus_distance = curr_focus_distance
            camera.data.dof.aperture_fstop = f_stop

            # ----------------------------
            # 1. Defocused render
            # ----------------------------

            disable_nodes()
            render_png(os.path.join(folder, "defocused.png"))

            # ----------------------------
            # 2. Sharp render
            # ----------------------------

            camera.data.dof.use_dof = False
            render_png(os.path.join(folder, "sharp.png"))

            # Restore DOF for metadata consistency
            camera.data.dof.use_dof = True
            camera.data.dof.focus_distance = curr_focus_distance
            camera.data.dof.aperture_fstop = f_stop

            # ----------------------------
            # 3. Depth + object index pass as multilayer EXR
            # ----------------------------

            scene.render.image_settings.file_format = "OPEN_EXR"
            scene.render.image_settings.color_depth = "32"

            scene.render.filepath = os.path.join(folder, "passes.exr")
            bpy.ops.render.render(write_still=True)

            # restore PNG for next renders
            scene.render.image_settings.file_format = "PNG"
            scene.render.image_settings.color_mode = "RGB"

            # ----------------------------
            # 4. Subject mask
            # ----------------------------

#            setup_mask_nodes(folder, subject_index)
#            bpy.ops.render.render(write_still=True)

#            disable_nodes()

            # ----------------------------
            # 5. Metadata JSON
            # ----------------------------

            metadata = {
                "image_id": img_i,
                "subject_name": subject.name,
                "subject_pass_index": subject_index,
                "depth_bin_m": [start, end],
                "focus_distance_m": curr_focus_distance,
                "focus_distance_cm": curr_focus_distance * 100,
                "f_stop": f_stop,
                "focal_length_mm": focal_length,
                "sensor_width_mm": sensor_width_mm,
                "resolution": [resolution_x, resolution_y],
                "cycles_samples": samples,
                "camera_location": list(camera.location),
                "camera_rotation_euler": list(camera.rotation_euler),
                "object_index_map": object_index_map,
            }

            with open(bpy.path.abspath(os.path.join(folder, "metadata.json")), "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"Saved {folder}")
            img_i += 1

    camera.data.shift_y = 0.0

print("Done.")