# Co-SLAM Implementation

This repository contains an improved implementation of Co-SLAM (Coordinate and Sparse Parametric Encodings for Neural Real-Time SLAM) along with tools for generating synthetic datasets using Blender.

## Overview

Co-SLAM is a neural RGB-D SLAM system that performs robust camera tracking and high-fidelity surface reconstruction in real-time. This implementation aims to test and evaluate Co-SLAM's performance on synthetic datasets. And further improve on the system.

## Repository Structure

- `Co-SLAM/`: Core implementation of Co-SLAM algorithm
- `Data_Manipulation/`: Scripts for generating synthetic datasets using Blender and Jupyter notebook to further manipulate data in order to generate Replica like datasets.

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch 1.10+
- Blender 2.93+ (for dataset generation)
- Additional dependencies listed in `requirements.txt`

### Installation

1. Clone this repository.
2. Refer to the README.md inside the Co-SLAM folder for the instalation of the necessary dependencies.

### Generating Synthetic Datasets

1. Open Blender and load the provided scene file.
2. Run the Blender script to generate RGB and depth images.
3. Use the provided Jupyter Notebook to post-process the generated data.

### Running Co-SLAM

1. Prepare your dataset in the required format.
2. Configure parameters in `config.yaml`
3. Run Co-SLAM:
   ```
   python coslam.py --config './configs/{Dataset}/{scene}.yaml'
   ```

### Running the Visualizer

1. Ensure you have completed a Co-SLAM run for the dataset you want to visualize.
2. Make sure the configuration file for the dataset is properly set up.
3. Run the visualizer:
   ```
   python visualizer.py --config './configs/{Dataset}/{scene}.yaml'
   ```

## Acknowledgments

This project builds upon and is inspired by the following works:

1. Wang, H., Wang, J., & Agapito, L. (2023). Co-SLAM: Joint Coordinate and Sparse Parametric Encodings for Neural Real-Time SLAM. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
   ```
   @inproceedings{wang2023coslam,
       title={Co-SLAM: Joint Coordinate and Sparse Parametric Encodings for Neural Real-Time SLAM},
       author={Wang, Hengyi and Wang, Jingwen and Agapito, Lourdes},
       booktitle={CVPR},
       year={2023}
   }
   ```
   [GitHub Repository](https://github.com/HengyiWang/Co-SLAM)