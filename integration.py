import argparse
import torch
import os, sys
sys.path.append("./zed_yolov8")

from threading import Lock, Thread
from argparse import Namespace
from zed_yolov8.person_detector import main
from lib.config import generate_configs
from scripts.extract import Filter, Extractor, _Extract
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

def extract_thread(camera_folder, lock):

    globals.extract_ready = False
    output_folder = os.path.join(camera_folder,"extract")
    args = Namespace(**{'exclude_gpus': None, 'configfile': None, 'loglevel': 'INFO', 'logfile': None,
                        'redirect_gui': False, 'depr_logfile_LF_F': None, 'input_dir': camera_folder,
                        'output_dir': output_folder, 'alignments_path': None, 'depr_alignments_al_p': None,
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

    configfile = args.configfile if hasattr(args, "configfile") else None
    normalization = None if args.normalization == "none" else args.normalization
    maskers = ["components", "extended"]
    maskers += args.masker if args.masker else []
    recognition = ("vgg_face2"
                   if args.identity or args.filter or args.nfilter
                   else None)

    _extractor = Extractor(args.detector,
                           args.aligner,
                           maskers,
                           recognition=recognition,
                           configfile=configfile,
                           multiprocess=not args.singleprocess,
                           exclude_gpus=args.exclude_gpus,
                           rotate_images=args.rotate_images,
                           min_size=args.min_size,
                           normalize_method=normalization,
                           re_feed=args.re_feed,
                           re_align=args.re_align)

    _filter = Filter(args.ref_threshold,
                              args.filter,
                              args.nfilter,
                              _extractor)

    extract = _Extract(_extractor, args)
    size = args.size if hasattr(args, "size") else 256
    globals.extract_ready = True
    while not globals.exit_signal:
        if globals.extract_signal:
            lock.acquire()
            extract._loader.init_images(args.input_dir)
            extract.process()
            lock.release()
            globals.extract_signal = False

            print(globals.zoomed_in_image.shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8m.pt', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--output_folder', type=str, default="./salah", help='output folder')

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

    extract_thread = Thread(target=extract_thread, kwargs={'camera_folder':output_folder, 'lock': lock})
    extract_thread.start()

    with torch.no_grad():
        main(output_folder, opt, lock)