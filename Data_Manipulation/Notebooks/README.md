# In This section I will provide some Jupyter Notebooks that I created during my thesis in Slam.


# 3D Image Processing and Visualization Notebook

This Jupyter notebook contains a collection of Python scripts for processing and visualizing 3D image data, particularly focusing on EXR file handling, depth map conversion, and camera trajectory analysis.

## Contents

1. **Imports**: The notebook begins with importing necessary libraries for image processing, 3D visualization, and data analysis.

2. **EXR to PNG Conversion**: 
   - Function `depth_exr_to_png()` for converting depth EXR images to PNG format.
   - Function `list_exr_channels()` to display available channels in an EXR file.
   - Batch conversion function `convert_all_exr_in_folder()` to process multiple EXR files.

3. **Image Display**: 
   - Code cells for loading and displaying PNG images using matplotlib.

4. **File Renaming**:
   - Scripts for renaming depth and RGB image files in specified directories.

5. **3D Point Cloud Visualization**:
   - Code for loading and visualizing 3D point clouds using Open3D.
   - Functions for rotating the view and capturing frames for animation.

6. **Camera Trajectory Analysis**:
   - Functions for reading and processing camera transformation matrices.
   - `extract_translation_components()` to extract camera positions from matrices.
   - Plotting functions for visualizing camera movement in 2D and 3D space.

7. **Data Visualization**:
   - Various matplotlib and Open3D based visualizations for analyzed data.

## Usage

To use this notebook:

1. Ensure all required libraries are installed (numpy, matplotlib, OpenEXR, cv2, open3d, etc.).
2. Adjust file paths in the code cells to match your directory structure.
3. Run cells sequentially, as later cells may depend on outputs from earlier ones.
4. Modify parameters and file paths as needed for your specific dataset.

## Key Functions

- `depth_exr_to_png(input_file, output_file, depth_channel='V')`: Converts a single EXR depth file to PNG.
- `convert_all_exr_in_folder(input_folder, output_folder, depth_channel='V')`: Batch converts EXR files in a folder.
- `read_and_reshape_matrices(file_name)`: Reads camera transformation matrices from a file.
- `extract_translation_components(matrices)`: Extracts camera positions from transformation matrices.
- `plot_camera_movement_3d(translations)`: Visualizes camera movement in 3D space.

## Notes

- This notebook is designed for processing and analyzing 3D image data, particularly from depth cameras or 3D reconstruction pipelines.
- Some cells may require long processing times depending on the size of your dataset.
- Ensure you have sufficient computational resources when working with large point clouds or high-resolution images.

## Customization

Feel free to modify the code to suit your specific needs. You may need to adjust parameters, file paths, or add additional processing steps depending on your particular dataset and analysis requirements.