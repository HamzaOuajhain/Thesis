# In This section I will provide some Scripts notebooks that I created during my thesis in Slam.


### 360.py 

#### Blender Multi-View Rendering Script

This Python script automates the process of rendering multiple views of a 3D scene in Blender, generating color images, depth maps, and normal maps.

##### Features

- Renders a specified number of views (default: 500)
- Supports randomized or fixed-step camera rotations
- Generates color images, depth maps, and normal maps
- Outputs camera transformation data in JSON format
- Configurable image resolution, format, and color depth
- Option to restrict upper views for more natural camera angles

##### Usage

1. Open your Blender project
2. Load this script into Blender's Text Editor
3. Adjust constants at the top of the script as needed
4. Run the script within Blender

##### Output

- Rendered images: `render_results_500/r_[index].png`
- Depth maps: `render_results_500/r_[index]_Depth.png`
- Normal maps: `render_results_500/r_[index]_Normal.png`
- Camera data: `render_results_500/transforms.json`


### Camera_Intrinsics.py

A Blender Python script to compute and extract intrinsic camera parameters.

## Features

- Calculates the intrinsic camera matrix from Blender camera settings
- Supports both vertical and horizontal sensor fit modes
- Accounts for pixel aspect ratio and render resolution settings

## Usage

1. Open your Blender project
2. Load this script into Blender's Text Editor
3. Modify the `cam_name` variable if needed (default: "Camera_R")
4. Run the script within Blender

## Output

Prints the intrinsic camera matrix for the specified camera to the Blender console.

## Requirements

- Blender 2.8+
- Python 3.7+