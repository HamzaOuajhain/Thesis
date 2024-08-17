#TODO: Explain modifications/ Add some visualisations


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