# In This section I will provide some Scripts/ Jupyter notebooks that I created during my thesis in Slam.


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

##### Requirements

- Blender 2.8+
- Python 3.7+

##### Note

Ensure your Blender scene is set up with a camera named 'Camera' before running the script.