#!/usr/bin/env python3

import argparse
import cv2
import os
import numpy as np
import pyzed.sl as sl

# Folder to save the frames
output_folder = "christina_all_frames"
os.makedirs(output_folder, exist_ok=True)

def main():
    global frame_counter

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
        print(repr(status))
        exit()

    print("Initialized Camera")

    image_left = sl.Mat()
    frame_counter = 0

    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Retrieve the left image
            zed.retrieve_image(image_left, sl.VIEW.LEFT)
            image_data = image_left.get_data()

            # Save the frame as an image
            frame_filename = os.path.join(output_folder, f"frame_{frame_counter}.jpg")
            cv2.imwrite(frame_filename, image_data)
            print(f"Saved frame {frame_counter} as {frame_filename}")
            frame_counter += 1

            # Display the frame
            cv2.imshow("ZED | Left Image", image_data)
            key = cv2.waitKey(10)
            if key == 27:  # Exit on pressing 'ESC'
                break
        else:
            print("Failed to grab frame. Exiting...")
            break

    zed.close()
    print("Camera closed.")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    opt = parser.parse_args()

    main()