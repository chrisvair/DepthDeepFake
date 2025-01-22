#!/usr/bin/env python3
import logging
import sys
import numpy as np

import argparse
import torch
import cv2
from pyzed import sl

from threading import Lock, Thread
from time import sleep

import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer

import os
import globals


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", 
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", 
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", 
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", 
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", 
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
    "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse", 
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", 
    "toothbrush"
]

def xywh2abcd(xywh, im_shape):
    output = np.zeros((4, 2))

    # Center / Width / Height -> BBox corners coordinates
    x_min = (xywh[0] - 0.5*xywh[2]) #* im_shape[1]
    x_max = (xywh[0] + 0.5*xywh[2]) #* im_shape[1]
    y_min = (xywh[1] - 0.5*xywh[3]) #* im_shape[0]
    y_max = (xywh[1] + 0.5*xywh[3]) #* im_shape[0]

    # A ------ B
    # | Object |
    # D ------ C

    output[0][0] = x_min
    output[0][1] = y_min

    output[1][0] = x_max
    output[1][1] = y_min

    output[2][0] = x_min
    output[2][1] = y_max

    output[3][0] = x_max
    output[3][1] = y_max
    return output

def detections_to_custom_box(detections, im0):
    output = []
    for i, det in enumerate(detections):
        xywh = det.xywh[0]

        # Creating ingestable objects for the ZED SDK
        obj = sl.CustomBoxObjectData()
        obj.bounding_box_2d = xywh2abcd(xywh, im0.shape)
        obj.label = det.cls.item()
        # class_id needs to be an integer
        #class_id = int(det.cls)
        #obj.label = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else "unknown"
        obj.probability = det.conf.item()
        obj.is_grounded = False
        output.append(obj)
    return output


def torch_thread(weights, img_size, lock, conf_thres=0.2, iou_thres=0.45):

    print("Intializing Network...")

    from ultralytics import YOLO
    model = YOLO(weights)

    while not globals.exit_signal:
        if globals.run_signal:
            lock.acquire()

            img = cv2.cvtColor(globals.image_net, cv2.COLOR_BGRA2RGB)
            # https://docs.ultralytics.com/modes/predict/#video-suffixes
            det = model.predict(img, save=False, imgsz=img_size, conf=conf_thres, iou=iou_thres)[0].cpu().numpy().boxes

            # ZED CustomBox format (with inverse letterboxing tf applied)
            globals.detections = detections_to_custom_box(det, globals.image_net)
            lock.release()

            globals.run_signal = False

        sleep(0.01)


def main(output_folder, opt, lock):
    capture_thread = Thread(target=torch_thread, kwargs={'weights': opt.weights, 'img_size': opt.img_size, 'lock':lock,"conf_thres": opt.conf_thres})
    capture_thread.start()

    print("Initializing Camera...")

    zed = sl.Camera()

    input_type = sl.InputType()
    if opt.svo is not None:
        input_type.set_from_svo_file(opt.svo)

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # QUALITY
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_maximum_distance = 50

    runtime_params = sl.RuntimeParameters()
    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status),"switching to builtin camera")
        main_builtin(output_folder, lock)

    main_zed(zed, init_params, runtime_params, output_folder, lock)

def main_builtin(output_folder, lock):
    
    print("Initializing Camera...")

    # Initialize the built-in camera
    cap = cv2.VideoCapture(0)  # Use 0 for default camera, or change if needed

    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Initialized Camera")

    while not globals.exit_signal:
        if not globals.extract_ready:
            logging.info("Waiting for extraction thread to start")
            sleep(1)
            continue

        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Process the frame
        lock.acquire()
        globals.image_net = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        lock.release()
        globals.run_signal = True

        # Wait for globals.detections
        logging.info("Waiting for Torch thread to complete")
        while globals.run_signal:
            sleep(0.01)

        lock.acquire()
        # Process globals.detections
        for det in globals.detections:
            if det.label == 0:  # Checking if it's a person
                bbox = det.bounding_box_2d
                x_min, y_min = int(bbox[0][0]), int(bbox[0][1])
                x_max, y_max = int(bbox[3][0]), int(bbox[3][1])

                # Draw bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Zoom into the person
                globals.zoomed_in_image = frame[y_min:y_max, x_min:x_max]

                # Save the zoomed-in image
                zoomed_in_filename = os.path.join(output_folder, f"zoomed_person_{globals.frame_counter}.jpg")
                cv2.imwrite(zoomed_in_filename, globals.zoomed_in_image)
                print(f"Saved zoomed-in image as {zoomed_in_filename}")

        globals.frame_counter += 1
        lock.release()

        globals.extract_signal = True

        # Wait for extraction
        while globals.extract_signal:
            sleep(1)
            logging.info("Waiting for extraction thread to finish")

        # Display the frame
        cv2.imshow("Camera Feed", frame)

        # Check for exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            globals.exit_signal = True

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    globals.exit_signal = True

def main_zed(zed, init_params, runtime_params, output_folder, lock):

    image_left_tmp = sl.Mat()

    print("Initialized Camera")

    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static, uncomment the following line to have better performances and boxes sticked to the ground.
    # positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)

    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = True
    zed.enable_object_detection(obj_param)

    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()

    # Display
    camera_infos = zed.get_camera_information()
    camera_res = camera_infos.camera_configuration.resolution
    # Create OpenGL viewer
    viewer = gl.GLViewer()
    point_cloud_res = sl.Resolution(min(camera_res.width, 720), min(camera_res.height, 404))
    point_cloud_render = sl.Mat()
    viewer.init(camera_infos.camera_model, point_cloud_res, obj_param.enable_tracking)
    point_cloud = sl.Mat(point_cloud_res.width, point_cloud_res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    image_left = sl.Mat()
    # Utilities for 2D display
    display_resolution = sl.Resolution(min(camera_res.width, 1280), min(camera_res.height, 720))
    image_scale = [display_resolution.width / camera_res.width, display_resolution.height / camera_res.height]
    image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239, 255], np.uint8)

    # Utilities for tracks view
    camera_config = camera_infos.camera_configuration
    tracks_resolution = sl.Resolution(400, display_resolution.height)
    track_view_generator = cv_viewer.TrackingViewer(tracks_resolution, camera_config.fps, init_params.depth_maximum_distance)
    track_view_generator.set_camera_calibration(camera_config.calibration_parameters)
    image_track_ocv = np.zeros((tracks_resolution.height, tracks_resolution.width, 4), np.uint8)
    # Camera pose
    cam_w_pose = sl.Pose()

    while viewer.is_available() and not globals.exit_signal:
        if not globals.extract_ready:
            print("Waiting for extract signal")
            continue

        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # -- Get the image
            lock.acquire()
            zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
            globals.image_net = image_left_tmp.get_data()
            lock.release()
            globals.run_signal = True

            # -- Detection running on the other thread
            while globals.run_signal:
                sleep(0.001)

            # Wait for globals.detections
            lock.acquire()
            # -- Ingest globals.detections
            zed.ingest_custom_box_objects(globals.detections)
            lock.release()
            zed.retrieve_objects(objects, obj_runtime_param)
            
            # depth map on zed thread and globals.detections on yolo thread
            lock.acquire()
            # Retrieve depth map and print for each detected person
            depth_map = sl.Mat()  # Create a Mat object to store depth data
            zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)  # Correct usage

            # Variables to keep track of the closest person
            closest_person_idx = -1
            min_depth = float('inf')
            person_count = 0
            for det in globals.detections:
                if det.label == 0 : # Checking if it's a person
                    # Get the bounding box of the person (det.bounding_box_2d)
                    bbox = det.bounding_box_2d
                    x_center = (bbox[0][0] + bbox[1][0]) / 2  # X center of the bbox
                    y_center = (bbox[0][1] + bbox[2][1]) / 2  # Y center of the bbox

                    # Get depth from depth_map at the center of the bounding box
                    x_center = int(x_center)
                    y_center = int(y_center)
                    depth_value = depth_map.get_value(x_center, y_center)
                    depth_value = depth_value[1]

                    # Compare to find the closest person
                    if depth_value < min_depth:
                        min_depth = depth_value
                        closest_person_idx = person_count

                    print(f"Person {person_count + 1}: Depth = {depth_value:.2f} meters")
                    person_count += 1

            if closest_person_idx != -1:
                # Zoom into the closest person
                closest_person = globals.detections[closest_person_idx]
                bbox = closest_person.bounding_box_2d
                x_min, y_min = bbox[0][0], bbox[0][1]
                x_max, y_max = bbox[3][0], bbox[3][1]
                
                # Crop the image to the bounding box
                globals.zoomed_in_image = globals.image_net[int(y_min):int(y_max), int(x_min):int(x_max)]

                # Save the zoomed-in image
                zoomed_in_filename = os.path.join(output_folder, f"zoomed_person_{globals.frame_counter}.jpg")
                cv2.imwrite(zoomed_in_filename, globals.zoomed_in_image)
                print(f"Saved zoomed-in image of person {closest_person_idx + 1} as {zoomed_in_filename}")
            else:
                print("No persons detected.")
            globals.frame_counter += 1
            lock.release()

            globals.extract_signal = True

            # Wait for extraction
            while globals.extract_signal:
                sleep(0.01)

            # -- Display
            # Retrieve display data
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, point_cloud_res)
            point_cloud.copy_to(point_cloud_render)
            zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)

            # 3D rendering
            viewer.updateData(point_cloud_render, objects)
            # 2D rendering
            np.copyto(image_left_ocv, image_left.get_data())
            cv_viewer.render_2D(image_left_ocv, image_scale, objects, obj_param.enable_tracking)
            global_image = cv2.hconcat([image_left_ocv, image_track_ocv])
            # Tracking view
            track_view_generator.generate_view(objects, cam_w_pose, image_track_ocv, objects.is_tracked)

            cv2.imshow("ZED | 2D View and Birds View", global_image)
            key = cv2.waitKey(10)
            if key == 27:
                globals.exit_signal = True
        else:
            globals.exit_signal = True

    viewer.exit()
    globals.exit_signal = True
    zed.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8m.pt', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--output_folder', type=str, default="./salah", help='output folder')

    opt = parser.parse_args()

    # Folder to save zoomed images
    output_folder = "salah"
    os.makedirs(output_folder, exist_ok=True)

    lock = Lock()

    with torch.no_grad():
        main(output_folder, opt, lock)