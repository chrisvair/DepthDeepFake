import argparse
import torch
import os, sys
sys.path.append("./zed_yolov8")

from threading import Lock, Thread
from argparse import Namespace
from zed_yolov8.person_detector import main
from lib.config import generate_configs
from scripts.extract import Filter, Extractor, _Extract
from scripts.convert import Convert
import logging
import globals
from time import sleep
import shutil
logging.basicConfig(level=logging.DEBUG)

def delete_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' and all its contents have been deleted.")
    except FileNotFoundError:
        print(f"Folder '{folder_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred while deleting the folder: {e}")

def convert_thread(camera_folder, lock, model_folder):

    globals.convert_ready = False
    extract_folder = os.path.join(camera_folder, "extract")
    convert_folder = os.path.join(camera_folder, "convert")

    ### Init Convert service
    convert_args = {'exclude_gpus': None, 'configfile': None, 'loglevel': 'INFO', 'logfile': None,
                    'redirect_gui': False, 'depr_logfile_LF_F': None,
                    'input_dir': extract_folder, 'output_dir': convert_folder, 'alignments_path': None,
                    'depr_alignments_al_p': None, 'reference_video': None, 'model_dir': model_folder,
                    'color_adjustment': 'avg-color', 'mask_type': 'extended', 'writer': 'opencv',
                    'output_scale': 100, 'frame_ranges': None, 'face_scale': 0.0, 'input_aligned_dir': None,
                    'nfilter': None, 'filter': None, 'ref_threshold': 0.4, 'jobs': 0, 'on_the_fly': False,
                    'keep_unchanged': False, 'swap_model': False, 'singleprocess': False,
                    'depr_singleprocess_sp_P': False, 'depr_reference-video_ref_r': None,
                    'depr_frame-ranges_fr_R': None, 'depr_output-scale_osc_O': None, 'depr_on-the-fly_otf_T': False}
    convert = None
    try:
        convert = Convert(Namespace(**convert_args))
    except Exception as e:
        logging.debug(f"Error initializing Convert service: {e}")

    globals.convert_ready = True
    while not globals.exit_signal:
        if globals.convert_signal:
            lock.acquire()
            if convert is not None:
                convert.init_images(convert_args.input_dir)
                convert.process()
            else:
                logging.DEBUG("Convert service is not initialized. Skipping conversion.")
            lock.release()
            globals.convert_signal = False


def extract_thread(camera_folder, lock):

    globals.extract_ready = False

    extract_folder = os.path.join(camera_folder,"extract")

    ### Init extract service
    extract_args = Namespace(**{'exclude_gpus': None, 'configfile': None, 'loglevel': 'INFO', 'logfile': None,
                        'redirect_gui': False, 'depr_logfile_LF_F': None, 'input_dir': camera_folder,
                        'output_dir': extract_folder, 'alignments_path': None, 'depr_alignments_al_p': None,
                        'batch_mode': False, 'detector': 's3fd', 'aligner': 'fan', 'masker': None,
                        'normalization': 'none', 're_feed': 0, 're_align': False, 'rotate_images': None,
                        'identity': False, 'min_size': 0, 'nfilter': None, 'filter': None, 'ref_threshold': 0.6,
                        'size': 512, 'extract_every_n': 1, 'save_interval': 0, 'debug_landmarks': False,
                        'singleprocess': False, 'skip_existing': False, 'skip_faces': False, 'skip_saving_faces': False,
                        'depr_min-size_min_m': None, 'depr_extract-every-n_een_N': None,
                        'depr_normalization_nm_O': None, 'depr_re-feed_rf_R': None, 'depr_size_sz_z': None,
                        'depr_save-interval_si_v': None, 'depr_debug-landmarks_dl_B': False,
                        'depr_singleprocess_sp_P': False, 'depr_skip-existing-faces_sf_e': False,
                        'depr_skip-saving-faces_ssf_K': False})

    configfile = extract_args.configfile if hasattr(extract_args, "configfile") else None
    normalization = None if extract_args.normalization == "none" else extract_args.normalization
    maskers = ["components", "extended"]
    maskers += extract_args.masker if extract_args.masker else []
    recognition = ("vgg_face2"
                   if extract_args.identity or extract_args.filter or extract_args.nfilter
                   else None)

    _extractor = Extractor(extract_args.detector,
                           extract_args.aligner,
                           maskers,
                           recognition=recognition,
                           configfile=configfile,
                           multiprocess=not extract_args.singleprocess,
                           exclude_gpus=extract_args.exclude_gpus,
                           rotate_images=extract_args.rotate_images,
                           min_size=extract_args.min_size,
                           normalize_method=normalization,
                           re_feed=extract_args.re_feed,
                           re_align=extract_args.re_align)

    extract = _Extract(_extractor, extract_args)
    globals.extract_ready = True


    while not globals.exit_signal:
        if globals.extract_signal:
            lock.acquire()
            extract._loader.init_images(extract_args.input_dir)
            extract.process()
            lock.release()
            globals.extract_signal = False
            globals.convert_signal = True
            print(globals.zoomed_in_image.shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8m.pt', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--output_folder', type=str, default="./salah", help='output folder')
    parser.add_argument('--model_folder', type=str, default="./models/model_christina_lou", help='Faceswap model folder')

    opt = parser.parse_args()

    # Folder to save zoomed images
    output_folder = "salahV2"
    delete_folder(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    lock = Lock()
    generate_configs()


    exit_signal = False
    run_signal = False
    extract_signal = False
    extract_ready = False

    extract_t = Thread(target=extract_thread, kwargs={'camera_folder':output_folder, 'lock': lock})
    extract_t.start()

    convert_t = Thread(target=convert_thread, kwargs={'camera_folder': output_folder, 'lock': lock,
                                                           "model_folder": opt.model_folder})
    convert_t.start()

    with torch.no_grad():
        main(output_folder, opt, lock)