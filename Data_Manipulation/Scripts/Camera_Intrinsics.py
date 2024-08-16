import bpy
from mathutils import Matrix

def compute_intrinsics(camera_data):
    """Compute the intrinsic camera matrix from Blender camera settings."""
    focal_length = camera_data.lens
    scene_data = bpy.context.scene
    res_x = scene_data.render.resolution_x
    res_y = scene_data.render.resolution_y
    scale_factor = scene_data.render.resolution_percentage / 100
    sensor_w = camera_data.sensor_width
    sensor_h = camera_data.sensor_height
    pixel_ratio = scene_data.render.pixel_aspect_x / scene_data.render.pixel_aspect_y
    
    if camera_data.sensor_fit == 'VERTICAL':
        s_u = res_x * scale_factor / sensor_w / pixel_ratio
        s_v = res_y * scale_factor / sensor_h
    else:
        s_u = res_x * scale_factor / sensor_w
        s_v = res_y * scale_factor * pixel_ratio / sensor_h

    p_u = focal_length * s_u
    p_v = focal_length * s_v
    c_x = res_x * scale_factor / 2
    c_y = res_y * scale_factor / 2
    
    K_matrix = Matrix((
        (p_u, 0, c_x),
        (0, p_v, c_y),
        (0, 0, 1)
    ))
    
    return K_matrix

if __name__ == "__main__":
    """Script to extract intrinsic camera parameters from Blender."""
    cam_name = "Camera_R"
    intrinsic_matrix = compute_intrinsics(bpy.data.objects[cam_name].data)
    print(f"Intrinsics for {cam_name}: \n{intrinsic_matrix}")
 
