import argparse, sys, os
import json
import bpy
import mathutils
import numpy as np

# Constants
DEBUG_MODE = False
NUM_VIEWS = 500
IMAGE_RESOLUTION = 800
OUTPUT_DIR = 'render_results_500'
DEPTH_MAP_SCALE = 1.4
COLOR_DEPTH = 8
IMAGE_FORMAT = 'PNG'
RANDOMIZE_VIEWS = True
RESTRICT_UPPER_VIEWS = True
FIXED_START_ROTATION = (.3, 0, 0)

# File paths
output_path = bpy.path.abspath(f"//{OUTPUT_DIR}")

# Helper function to convert matrix to list
def matrix_to_list(matrix):
    return [list(row) for row in matrix]

# Create output directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Data to store in JSON file
output_data = {
    'camera_angle_x': bpy.data.objects['Camera'].data.angle_x,
}

# Render optimizations
bpy.context.scene.render.use_persistent_data = True

# Set up rendering of depth map
bpy.context.scene.use_nodes = True
node_tree = bpy.context.scene.node_tree
node_links = node_tree.links

# Set image format and color depth
bpy.context.scene.render.image_settings.file_format = IMAGE_FORMAT
bpy.context.scene.render.image_settings.color_depth = str(COLOR_DEPTH)

if not DEBUG_MODE:
    # Create input render layer node
    render_layers_node = node_tree.nodes.new('CompositorNodeRLayers')

    depth_output_node = node_tree.nodes.new(type="CompositorNodeOutputFile")
    depth_output_node.label = 'Depth Output'
    if IMAGE_FORMAT == 'OPEN_EXR':
        node_links.new(render_layers_node.outputs['Depth'], depth_output_node.inputs[0])
    else:
        # Remap depth for other types that cannot represent the full range
        map_node = node_tree.nodes.new(type="CompositorNodeMapValue")
        map_node.offset = [-0.7]
        map_node.size = [DEPTH_MAP_SCALE]
        map_node.use_min = True
        map_node.min = [0]
        node_links.new(render_layers_node.outputs['Depth'], map_node.inputs[0])
        node_links.new(map_node.outputs[0], depth_output_node.inputs[0])

    normal_output_node = node_tree.nodes.new(type="CompositorNodeOutputFile")
    normal_output_node.label = 'Normal Output'
    node_links.new(render_layers_node.outputs['Normal'], normal_output_node.inputs[0])

# Background settings
bpy.context.scene.render.dither_intensity = 0.0
bpy.context.scene.render.film_transparent = True

# Remove unnecessary objects
objects_to_delete = [obj for obj in bpy.context.scene.objects if obj.type in ('EMPTY') and 'Empty' in obj.name]
bpy.ops.object.delete({"selected_objects": objects_to_delete})

# Function to parent camera to an empty object
def parent_camera_to_empty(camera):
    origin = (0, 0, 0)
    empty_object = bpy.data.objects.new("Empty", None)
    empty_object.location = origin
    camera.parent = empty_object

    scene = bpy.context.scene
    scene.collection.objects.link(empty_object)
    bpy.context.view_layer.objects.active = empty_object
    return empty_object

# Scene setup
scene = bpy.context.scene
scene.render.resolution_x = IMAGE_RESOLUTION
scene.render.resolution_y = IMAGE_RESOLUTION
scene.render.resolution_percentage = 100

# Camera setup
camera = scene.objects['Camera']
camera.location = (0, 4.0, 0.5)
camera_constraint = camera.constraints.new(type='TRACK_TO')
camera_constraint.track_axis = 'TRACK_NEGATIVE_Z'
camera_constraint.up_axis = 'UP_Y'
empty_object = parent_camera_to_empty(camera)
camera_constraint.target = empty_object

scene.render.image_settings.file_format = 'PNG'  # Set output format to .png

# Rotation setup
rotation_step = 360.0 / NUM_VIEWS
rotation_mode = 'XYZ'

if not DEBUG_MODE:
    for output_node in [depth_output_node, normal_output_node]:
        output_node.base_path = ''

output_data['frames'] = []

if not RANDOMIZE_VIEWS:
    empty_object.rotation_euler = FIXED_START_ROTATION

# Main rendering loop
for i in range(NUM_VIEWS):
    if DEBUG_MODE:
        i = np.random.randint(0, NUM_VIEWS)
        empty_object.rotation_euler[2] += radians(rotation_step * i)
    if RANDOMIZE_VIEWS:
        scene.render.filepath = os.path.join(output_path, f'r_{i}')
        if RESTRICT_UPPER_VIEWS:
            rotation = np.random.uniform(0, 1, size=3) * (1, 0, 2 * np.pi)
            rotation[0] = np.abs(np.arccos(1 - 2 * rotation[0]) - np.pi / 2)
            empty_object.rotation_euler = rotation
        else:
            empty_object.rotation_euler = np.random.uniform(0, 2 * np.pi, size=3)
    else:
        print(f"Rotation {rotation_step * i}, {radians(rotation_step * i)}")
        scene.render.filepath = os.path.join(output_path, f'r_{i:03d}')

    if not DEBUG_MODE:
        bpy.ops.render.render(write_still=True)  # Render still image

    frame_data = {
        'file_path': scene.render.filepath,
        'rotation': radians(rotation_step),
        'transform_matrix': matrix_to_list(camera.matrix_world)
    }
    output_data['frames'].append(frame_data)

    if RANDOMIZE_VIEWS:
        if RESTRICT_UPPER_VIEWS:
            rotation = np.random.uniform(0, 1, size=3) * (1, 0, 2 * np.pi)
            rotation[0] = np.abs(np.arccos(1 - 2 * rotation[0]) - np.pi / 2)
            empty_object.rotation_euler = rotation
        else:
            empty_object.rotation_euler = np.random.uniform(0, 2 * np.pi, size=3)
    else:
        empty_object.rotation_euler[2] += radians(rotation_step)

# Save transformation data to JSON
if not DEBUG_MODE:
    with open(os.path.join(output_path, 'transforms.json'), 'w') as output_file:
        json.dump(output_data, output_file, indent=4)
 
