# ZED-YOLO Person Detection and Deepfake Integration

This part of the project project combines the Stereolabs ZED 2 camera and YOLO object detection. It aims to detect people in live 3D video, finds the closest to the camera, and overlays future deepfake-processed images.

---

## Project Files Overview

### `person_detector.py`
First Implementation
  - Detects people in the live feed using synchronized YOLO object detection.
  - Tracks their positions in 3D space using ZED camera depth data.
  - Depth-based selection of the closest person.
  - Captures zoomed-in images (bounding boxes) of the closest detected person.

 ### `image_video_integration.py`
 Second Implementation
  - Testing integartion of a sigle predefined image (replacement_images) to the detected bounding box of the closest person

### `deep_fake_video_integration.py`
Third implementation to be completed with the deepfake model
  - Process bounding boxes images with a forthcoming deepfake model.
  - Overlays the modified results back into the live feed.
  - Final implementation will directly inject deepfake-enhanced outputs into the main pipeline.

### `cv_viewer` and `ogl_viewer`
- **Origin**: [Stereolabs ZED-YOLO repository](https://github.com/stereolabs/zed-yolo).
- **cv_viewer**:
  - Renders 2D visualizations, such as bounding boxes and tracking paths, on the live feed.
  - Highlights detected objects and tracks their movement.
- **ogl_viewer**:
  - Renders 3D visualizations, including point clouds and object detection data.
  - Useful for debugging and visualizing 3D data from the ZED camera.

---

## Installation and Setup

### Prerequisites
1. **Python 3.8+**
2. **Required Libraries**:
   - `torch`2.5.1+cu124
   - `ultralytics` (YOLOv8)
   - `pyzed` (Stereolabs ZED SDK bindings)
   - `opencv-python`
   - `numpy` 1.26.3
3. **Stereolabs ZED SDK**:
   - Install the [ZED SDK](https://www.stereolabs.com/zed-sdk/) for camera functionality.
