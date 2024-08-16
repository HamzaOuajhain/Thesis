# In This section I will provide some Scripts that I created during my thesis in Slam.


## 360.py 

### Blender Multi-View Rendering Script

This Python script automates the process of rendering multiple views of a 3D scene in Blender, generating color images, depth maps, and normal maps.

### Features

- Renders a specified number of views (default: 500)
- Supports randomized or fixed-step camera rotations
- Generates color images, depth maps, and normal maps
- Outputs camera transformation data in JSON format
- Configurable image resolution, format, and color depth
- Option to restrict upper views for more natural camera angles

### Usage

1. Open your Blender project
2. Load this script into Blender's Text Editor
3. Adjust constants at the top of the script as needed
4. Run the script within Blender

### Output

- Rendered images: `render_results_500/r_[index].png`
- Depth maps: `render_results_500/r_[index]_Depth.png`
- Normal maps: `render_results_500/r_[index]_Normal.png`
- Camera data: `render_results_500/transforms.json`

### Requirements

- Blender 2.8+
- Python 3.7+


## Camera_Intrinsics.py

A Blender Python script to compute and extract intrinsic camera parameters.

### Features

- Calculates the intrinsic camera matrix from Blender camera settings
- Supports both vertical and horizontal sensor fit modes
- Accounts for pixel aspect ratio and render resolution settings

### Usage

1. Open your Blender project
2. Load this script into Blender's Text Editor
3. Modify the `cam_name` variable if needed (default: "Camera_R")
4. Run the script within Blender

### Output

Prints the intrinsic camera matrix for the specified camera to the Blender console.

### Requirements

- Blender 2.8+
- Python 3.7+


## 360_Manual.py

A Blender Python script for automated camera pose extraction, image rendering, and intrinsic parameter calculation.



### Features

- Saves camera poses relative to a tracked object
- Renders color images for each frame
- Calculates and saves camera intrinsic parameters
- Computes and saves the transformation between color and depth cameras
- Handles both color and depth cameras

### Usage

1. Open your Blender project
2. Set up your scene with cameras named "Camera_L" (color) and "Camera_R" (depth)
3. Ensure there's an object named "teabox" in the scene
4. Adjust the `base_output_dir` variable if needed
5. Run the script within Blender

### Output

- Camera poses: `/tmp/teabox/ground-truth/Camera_L_frame_XXXX.txt`
- Color images: `/tmp/teabox/color/frame_XXXX.jpg`
- Camera intrinsics: `/tmp/teabox/Camera_L.xml` and `/tmp/teabox/Camera_R.xml`
- Depth-to-color transform: `/tmp/teabox/depth_to_color_transform.txt`

### Credits
This script is inspired by and adapts concepts from the Open Source Visual Servoing Platform (ViSP). ViSP is a modular cross platform library that allows prototyping and developing applications using visual tracking and visual servoing techniques.

### Requirements

- Blender 2.8+
- Python 3.7+
