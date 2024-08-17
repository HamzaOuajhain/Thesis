import os
from multiprocessing import Process, Queue
from queue import Empty

import numpy as np
import open3d as o3d
import torch


def normalize_vector(vector):
    return vector / np.linalg.norm(vector)


def create_camera_representation(index, is_ground_truth=False, scale=0.005):
    camera_points = scale * np.array([
        [0, 0, 0],
        [-1, -1, 1.5],
        [1, -1, 1.5],
        [1, 1, 1.5],
        [-1, 1, 1.5],
        [-0.5, 1, 1.5],
        [0.5, 1, 1.5],
        [0, 1.2, 1.5]])

    camera_lines = np.array([[1, 2], [2, 3], [3, 4], [4, 1], [1, 3], [2, 4],
                             [1, 0], [0, 2], [3, 0], [0, 4], [5, 7], [7, 6]])
    points = []
    
    for line in camera_lines:
        start_point, end_point = camera_points[line[0]], camera_points[line[1]]
        transition_values = np.linspace(0., 1., 100)
        interpolated_points = start_point[None, :] * (1. - transition_values)[:, None] + end_point[None, :] * transition_values[:, None]
        points.append(interpolated_points)
    
    points = np.concatenate(points)
    color = (0.0, 0.0, 0.0) if is_ground_truth else (1.0, .0, .0)
    
    camera_actor = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points))
    camera_actor.paint_uniform_color(color)

    return camera_actor


def render_trajectory(queue, output_path, initial_pose, camera_scale,
                      save_rendering, near_clip, estimated_poses, ground_truth_poses):
    render_trajectory.queue = queue
    render_trajectory.camera_actors = {}
    render_trajectory.points_actors = {}
    render_trajectory.index = 0
    render_trajectory.warmup = 0
    render_trajectory.mesh_actor = None
    render_trajectory.frame_index = 0
    render_trajectory.trajectory_actor = None
    render_trajectory.trajectory_actor_gt = None
    
    if save_rendering:
        os.system(f"rm -rf {output_path}/tmp_rendering")

    def animation_callback(visualizer):
        camera_params = visualizer.get_view_control().convert_to_pinhole_camera_parameters()
        
        while True:
            try:
                data = render_trajectory.queue.get_nowait()
                if data[0] == 'pose':
                    idx, pose, is_gt = data[1:]
                    if is_gt:
                        idx += 100000

                    if idx in render_trajectory.camera_actors:
                        cam_actor, previous_pose = render_trajectory.camera_actors[idx]
                        pose_change = pose @ np.linalg.inv(previous_pose)

                        cam_actor.transform(pose_change)
                        visualizer.update_geometry(cam_actor)

                        if idx in render_trajectory.points_actors:
                            point_actor = render_trajectory.points_actors[idx]
                            point_actor.transform(pose_change)
                            visualizer.update_geometry(point_actor)
                    else:
                        cam_actor = create_camera_representation(idx, is_gt, camera_scale)
                        cam_actor.transform(pose)
                        visualizer.add_geometry(cam_actor)

                    render_trajectory.camera_actors[idx] = (cam_actor, pose)

                elif data[0] == 'mesh':
                    mesh_file = data[1]
                    if render_trajectory.mesh_actor is not None:
                        visualizer.remove_geometry(render_trajectory.mesh_actor)
                    render_trajectory.mesh_actor = o3d.io.read_triangle_mesh(mesh_file)
                    render_trajectory.mesh_actor.compute_vertex_normals()
                    visualizer.add_geometry(render_trajectory.mesh_actor)

                elif data[0] == 'trajectory':
                    idx, is_gt = data[1:]

                    color = (0.0, 0.0, 0.0) if is_gt else (1.0, .0, .0)
                    trajectory_actor = o3d.geometry.PointCloud(
                        points=o3d.utility.Vector3dVector(
                            ground_truth_poses[1:idx, :3, 3] if is_gt else estimated_poses[1:idx, :3, 3]))
                    trajectory_actor.paint_uniform_color(color)

                    if is_gt:
                        if render_trajectory.trajectory_actor_gt is not None:
                            visualizer.remove_geometry(render_trajectory.trajectory_actor_gt)
                            tmp = render_trajectory.trajectory_actor_gt
                            del tmp
                        render_trajectory.trajectory_actor_gt = trajectory_actor
                        visualizer.add_geometry(render_trajectory.trajectory_actor_gt)
                    else:
                        if render_trajectory.trajectory_actor is not None:
                            visualizer.remove_geometry(render_trajectory.trajectory_actor)
                            tmp = render_trajectory.trajectory_actor
                            del tmp
                        render_trajectory.trajectory_actor = trajectory_actor
                        visualizer.add_geometry(render_trajectory.trajectory_actor)

                elif data[0] == 'reset':
                    render_trajectory.warmup = -1

                    for idx in render_trajectory.points_actors:
                        visualizer.remove_geometry(render_trajectory.points_actors[idx])

                    for idx in render_trajectory.camera_actors:
                        visualizer.remove_geometry(render_trajectory.camera_actors[idx][0])

                    render_trajectory.camera_actors = {}
                    render_trajectory.points_actors = {}

            except Empty:
                break

        if len(render_trajectory.camera_actors) >= render_trajectory.warmup:
            visualizer.get_view_control().convert_from_pinhole_camera_parameters(camera_params)

        visualizer.poll_events()
        visualizer.update_renderer()
        
        if save_rendering:
            render_trajectory.frame_index += 1
            os.makedirs(f'{output_path}/tmp_rendering', exist_ok=True)
            visualizer.capture_screen_image(
                f'{output_path}/tmp_rendering/{render_trajectory.frame_index:06d}.jpg')

    visualizer = o3d.visualization.Visualizer()

    visualizer.register_animation_callback(animation_callback)
    visualizer.create_window(window_name=output_path, height=1080, width=1920)
    visualizer.get_render_option().point_size = 4
    visualizer.get_render_option().mesh_show_back_face = False

    view_control = visualizer.get_view_control()
    view_control.set_constant_z_near(near_clip)
    view_control.set_constant_z_far(1000)

    initial_camera_params = view_control.convert_to_pinhole_camera_parameters()
    initial_pose[:3, 3] += 2 * normalize_vector(initial_pose[:3, 2])
    initial_pose[:3, 2] *= -1
    initial_pose[:3, 1] *= -1
    initial_pose = np.linalg.inv(initial_pose)

    initial_camera_params.extrinsic = initial_pose
    view_control.convert_from_pinhole_camera_parameters(initial_camera_params)

    visualizer.run()
    visualizer.destroy_window()


class SLAMFrontend:
    def __init__(self, output_path, initial_pose, camera_scale=1, save_rendering=False,
                 near_clip=0, estimated_poses=None, ground_truth_poses=None):
        self.queue = Queue()
        self.process = Process(target=render_trajectory, args=(
            self.queue, output_path, initial_pose, camera_scale, save_rendering,
            near_clip, estimated_poses, ground_truth_poses))

    def update_pose(self, index, pose, is_ground_truth=False):
        if isinstance(pose, torch.Tensor):
            pose = pose.cpu().numpy()

        pose[:3, 2] *= -1
        self.queue.put_nowait(('pose', index, pose, is_ground_truth))

    def update_mesh(self, file_path):
        self.queue.put_nowait(('mesh', file_path))

    def update_cam_trajectory(self, trajectory_list, is_ground_truth):
        self.queue.put_nowait(('trajectory', trajectory_list, is_ground_truth))

    def reset(self):
        self.queue.put_nowait(('reset',))

    def start(self):
        self.process.start()
        return self

    def join(self):
        self.process.join()
