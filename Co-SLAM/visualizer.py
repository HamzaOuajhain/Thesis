import argparse
import os
import time

import numpy as np
import torch
from tqdm import tqdm

import config
from datasets.dataset import get_dataset
from tools.viz import SLAMFrontend


def convert_dict_to_numpy(pose_dict: dict) -> np.ndarray:
    pose_array = pose_dict[0].numpy()[np.newaxis, :]
    for idx in range(1, len(pose_dict)):
        pose_array = np.vstack((pose_array, pose_dict[idx].numpy()[np.newaxis, :]))
    return pose_array


def convert_list_tensor_to_numpy(tensor_list: list) -> np.ndarray:
    numpy_array = [tensor.numpy() for tensor in tensor_list]
    return np.array(numpy_array)


if __name__ == '__main__':
    print('Starting the visualizer...')
    
    parser = argparse.ArgumentParser(description='Arguments for running the visualizer.')
    parser.add_argument('--config_file', type=str, help='Path to the configuration file.')
    parser.add_argument('--input_directory', type=str,
                        help='Input directory; overrides the config file if provided.')
    parser.add_argument('--output_directory', type=str,
                        help='Output directory; overrides the config file if provided.')
    parser.add_argument('--use_ground_truth', type=bool,
                        help='Use ground truth pose or not', default=True)
    arguments = parser.parse_args()

    configuration = config.load_config(arguments.config_file)
    
    if arguments.input_directory is not None:
        configuration['data']['datadir'] = arguments.input_directory
    if arguments.output_directory is not None:
        configuration['data']['output'] = arguments.output_directory

    ground_truth_poses = convert_list_tensor_to_numpy(get_dataset(configuration).poses)
    estimated_poses = None

    output_path = os.path.join(configuration['data']['output'], configuration['data']['exp_name'])
    
    if os.path.exists(output_path):
        checkpoints = [os.path.join(output_path, file)
                       for file in sorted(os.listdir(output_path)) if 'checkpoint' in file]
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            print('Loading checkpoint:', latest_checkpoint)
            checkpoint_data = torch.load(latest_checkpoint, map_location=torch.device('cpu'))
            estimated_poses = checkpoint_data['pose']
    
    estimated_poses = convert_dict_to_numpy(estimated_poses)

    visualizer = SLAMFrontend(output_path, initial_pose=estimated_poses[0], camera_scale=0.3,
                              save_rendering=False, near_clip=0,
                              estimated_poses=estimated_poses, ground_truth_poses=ground_truth_poses).start()

    for idx in tqdm(range(0, len(estimated_poses))):
        time.sleep(0.03)
        mesh_file = f'{output_path}/mesh_track{idx}.ply'
        
        if os.path.isfile(mesh_file):
            visualizer.update_mesh(mesh_file)
        
        visualizer.update_pose(1, estimated_poses[idx], is_ground_truth=False)

        if arguments.use_ground_truth:
            visualizer.update_pose(1, ground_truth_poses[idx], is_ground_truth=True)

        if idx % 10 == 0:
            visualizer.update_cam_trajectory(idx, is_ground_truth=False)
            if arguments.use_ground_truth:
                visualizer.update_cam_trajectory(idx, is_ground_truth=True)
