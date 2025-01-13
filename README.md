# DepthDeepFake


## Faceswap:

# Faceswap Pipeline with Docker (GPU)

This repository contains a pre-configured pipeline for face extraction and conversion using a trained Faceswap model. The project is designed to run in a Docker container with GPU support to ensure reproducibility and ease of use.

---

## Prerequisites

Before proceeding, make sure you have the following installed on your system:

1. **Docker**: Install Docker by following [this official guide](https://docs.docker.com/get-docker/).
2. **Nvidia GPU Drivers**: Ensure that your Nvidia GPU drivers are installed and up-to-date.
3. **Nvidia Container Toolkit**: Configure Docker to use your GPU by following [this guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

---

## How to Use the Pipeline

### 1. Clone the Repository

First, clone this repository to your local machine

### 2. Build the Docker Image

To build the Docker image for the pipeline, run the following command in the repository's root directory:
```bash
docker build -f Dockerfile.gpu -t faceswap-pipeline .
```

This will create a Docker image named `faceswap-pipeline` using the GPU-enabled `Dockerfile.gpu`.


### 3. Run the Pipeline

To run the pipeline, execute the following command:
```bash
docker run --gpus all -it faceswap-pipeline
```

This will start the Docker container and automatically execute the `pipeline.py` script, which performs both face extraction and conversion.


### 4. Customize Input, Output, and Model

If you want to customize the input folder, output folder, or model used in the pipeline, edit the `pipeline.py` file before building the Docker image. Update the following variables in `pipeline.py`:

- **Input folder**: The folder containing images from the ZED camera to process (by default: /srv/src/christina')
- **Output folder**: The folder where results will be saved (by default: '/srv/output_christinalou') 
- **Model folder**: The path to the model that we trained with our faces.

After making changes, rebuild the Docker image using the command in Step 2.

### Results

The processed images will be saved in the output directory specified in the `pipeline.py` script.

## Project Files

- **Dockerfile.gpu**: Dockerfile for building the GPU-enabled environment.
- **pipeline.py**: The main script that automates face extraction and conversion.
- **models/**: Contains the trained model for Faceswap.
- **src/**: The folder with source images for face extraction and conversion.


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
   - Install the [ZED SDK](https://www.stereolabs.com/en-fr/developers/release) for camera functionality.
