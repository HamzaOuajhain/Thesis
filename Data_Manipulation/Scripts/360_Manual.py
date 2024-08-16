import bpy
import os
from mathutils import Matrix
import sys

def save_transform_between_cameras(output_directory, scene, color_camera_name, depth_camera_name):
    """! Save the transformation matrix between the depth and color cameras to a file.
    @param[in] output_directory String path where the transform file will be saved.
    @param[in] scene Blender scene object.
    @param[in] color_camera_name String name of the color camera.
    @param[in] depth_camera_name String name of the depth camera.
    """

    world_matrix_color = get_object_world_matrix(color_camera_name)
    world_matrix_depth = get_object_world_matrix(depth_camera_name)
    transform_matrix = world_matrix_depth.inverted() @ world_matrix_color

    output_file = os.path.join(output_directory, "depth_to_color_transform.txt")
    save_matrix_to_file(output_file, transform_matrix)
    print(f"Saved: {output_file}")

def get_object_world_matrix(object_name):
    """! Get the 4x4 world matrix of an object.
    @param[in] object_name String name of the object.
    @return Normalized 4x4 homogeneous world matrix of the object.
    """
    world_matrix = bpy.data.objects[object_name].matrix_world
    normalized_matrix = world_matrix.copy()
    normalized_rotation = normalized_matrix.to_3x3().normalized()

    for i in range(3):
        for j in range(3):
            normalized_matrix[i][j] = normalized_rotation[i][j]

    return normalized_matrix

def calculate_camera_to_object_transform(camera_name, object_name):
    """! Calculate the transformation matrix from the camera to the object.
    @param[in] camera_name String name of the camera.
    @param[in] object_name String name of the object.
    @return 4x4 homogeneous matrix representing the camera-to-object transform.
    """
    correction_matrix = Matrix().to_4x4()
    correction_matrix[1][1] = -1
    correction_matrix[2][2] = -1

    world_matrix_camera = get_object_world_matrix(camera_name)
    world_matrix_object = get_object_world_matrix(object_name)

    camera_to_object_transform = correction_matrix @ world_matrix_camera.inverted() @ world_matrix_object
    return camera_to_object_transform

def save_matrix_to_file(filename, matrix):
    """! Save a 4x4 matrix to a text file.
    @param[in] filename String path of the file where the matrix will be saved.
    @param[in] matrix 4x4 matrix to save.
    """
    print(f"Saving to file: {filename}")
    with open(filename, 'w') as file:
        for row in matrix:
            file.write(" ".join(map(str, row)) + "\n")

def save_camera_pose(output_directory, scene, camera_name, object_name):
    """! Save the camera pose relative to the object for the current frame.
    @param[in] output_directory String path where the pose will be saved.
    @param[in] scene Blender scene object.
    @param[in] camera_name String name of the camera.
    @param[in] object_name String name of the object.
    """
    frame_num = scene.frame_current
    print(f"\n\nProcessing Frame: {frame_num}")
    camera_pose_matrix = calculate_camera_to_object_transform(camera_name, object_name)

    pose_filename = os.path.join(output_directory, f"{camera_name}_frame_{frame_num:04d}.txt")
    save_matrix_to_file(pose_filename, camera_pose_matrix)

def get_camera_intrinsics(camera_name):
    """! Get the intrinsic parameters of a camera.
    @param[in] camera_name String name of the camera.
    @return Tuple containing image dimensions, focal length ratios, and principal point coordinates.
    """
    camera_data = bpy.data.objects[camera_name].data
    focal_length = camera_data.lens
    scene = bpy.context.scene
    res_x = scene.render.resolution_x
    res_y = scene.render.resolution_y
    scale_factor = scene.render.resolution_percentage / 100
    sensor_width = camera_data.sensor_width
    sensor_height = camera_data.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

    if camera_data.sensor_fit == 'VERTICAL':
        su = res_x * scale_factor / sensor_width / pixel_aspect_ratio
        sv = res_y * scale_factor / sensor_height
    else:
        su = res_x * scale_factor / sensor_width
        sv = res_y * scale_factor * pixel_aspect_ratio / sensor_height

    px = focal_length * su
    py = focal_length * sv
    u0 = res_x * scale_factor / 2
    v0 = res_y * scale_factor / 2

    return res_x, res_y, px, py, u0, v0

def save_camera_intrinsics(filename, camera_name, width, height, px, py, u0, v0):
    """! Save camera intrinsic parameters to an XML file.
    @param[in] filename String path where the intrinsics will be saved.
    @param[in] camera_name String name of the camera.
    @param[in] width Image width.
    @param[in] height Image height.
    @param[in] px, py Focal length ratios.
    @param[in] u0, v0 Principal point coordinates.
    """
    print(f"Saving intrinsics to file: {filename}")
    with open(filename, 'w') as file:
        file.write("<?xml version=\"1.0\"?>\n")
        file.write("<camera_intrinsics>\n")
        file.write(f"  <camera_name>{camera_name}</camera_name>\n")
        file.write(f"  <image_width>{width}</image_width>\n")
        file.write(f"  <image_height>{height}</image_height>\n")
        file.write("  <projection>\n")
        file.write("    <type>perspective</type>\n")
        file.write(f"    <px>{px}</px>\n")
        file.write(f"    <py>{py}</py>\n")
        file.write(f"    <u0>{u0}</u0>\n")
        file.write(f"    <v0>{v0}</v0>\n")
        file.write("  </projection>\n")
        file.write("</camera_intrinsics>\n")

if __name__ == '__main__':
    """! Main script to save camera poses, images, and transforms between color and depth cameras."""
    color_camera = "Camera_L"
    depth_camera = "Camera_R"
    tracked_object = "teabox"
    base_output_dir = "/tmp/teabox/"
    pose_output_dir = os.path.join(base_output_dir, "ground-truth/")
    color_image_dir = os.path.join(base_output_dir, "color/")
    depth_image_dir = os.path.join(base_output_dir, "depth/")
    image_format = "JPEG"

    if not os.path.exists(base_output_dir):
        print(f"Creating directory: {base_output_dir}")
        os.makedirs(base_output_dir)

    if not os.path.exists(pose_output_dir):
        print(f"Creating directory: {pose_output_dir}")
        os.makedirs(pose_output_dir)

    scene = bpy.context.scene

    save_transform_between_cameras(base_output_dir, scene, color_camera, depth_camera)

    color_intrinsics = get_camera_intrinsics(color_camera)
    color_intrinsics_file = os.path.join(base_output_dir, f"{color_camera}.xml")
    save_camera_intrinsics(color_intrinsics_file, color_camera, *color_intrinsics)

    depth_intrinsics = get_camera_intrinsics(depth_camera)
    depth_intrinsics_file = os.path.join(base_output_dir, f"{depth_camera}.xml")
    save_camera_intrinsics(depth_intrinsics_file, depth_camera, *depth_intrinsics)

    for frame in range(scene.frame_start, scene.frame_end):
        scene.frame_set(frame)

        if not os.path.exists(color_image_dir):
            os.makedirs(color_image_dir)
        scene.render.filepath = os.path.join(color_image_dir, f"frame_{frame:04d}")
        scene.render.image_settings.file_format = image_format
        bpy.ops.render.render(write_still=True)

        save_camera_pose(pose_output_dir, scene, color_camera, tracked_object)

        depth_image_file_R = os.path.join(color_image_dir, f"frame_{frame:04d}_R{scene.render.file_extension}")
        if os.path.exists(depth_image_file_R):
            print(f"Removing file: {depth_image_file_R}")
            os.remove(depth_image_file_R)

        depth_image_file_L = os.path.join(depth_image_dir, f"DepthImage_{frame:04d}_L.exr")
        if os.path.exists(depth_image_file_L):
            print(f"Removing file: {depth_image_file_L}")
            os.remove(depth_image_file_L)
 
