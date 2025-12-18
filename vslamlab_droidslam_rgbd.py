from pathlib import Path
from tqdm import tqdm
import pkg_resources
import pandas as pd
import numpy as np
import argparse
import torch
import yaml
import time
import csv
import cv2
import sys
import os

sys.path.append('droid_slam') 
from droid_slam.droid import Droid 

timestamps = [] 

def show_image(image): 
    image = image.permute(1, 2, 0).cpu().numpy() 
    cv2.imshow('image', image / 255.0) 
    cv2.waitKey(1)

def load_calibration(calibration_yaml: Path, cam_name: str):
    with open(calibration_yaml, 'r') as file:
        data = yaml.safe_load(file)
    cameras = data.get('cameras', [])
    for cam_ in cameras:
        if cam_['cam_name'] == cam_name:
            cam = cam_;
            break;
    print(f"\nCamera Name: {cam['cam_name']}")
    print(f"Camera Type: {cam['cam_type']}")
    print(f"Depth Name: {cam['depth_name']}")
    print(f"Camera Model: {cam['cam_model']}")
    print(f"Focal Length: {cam['focal_length']}")
    print(f"Principal Point: {cam['principal_point']}")
    has_dist = ('distortion_type' in cam) and ('distortion_coefficients' in cam)
    if has_dist:
        print(f"Distortion Type Dimension: {cam['distortion_type']}")
        print(f"Distortion Coefficients: {cam['distortion_coefficients']}")
    print(f"Depth Factor: {cam['depth_factor']}")    
    print(f"Image Dimension: {cam['image_dimension']}")
    print(f"Fps: {cam['fps']}")

    K = np.array([[cam['focal_length'][0], 0,  cam['principal_point'][0]],
                  [0,  cam['focal_length'][1], cam['principal_point'][1]],
                  [0,  0,   1]], dtype=np.float32)
    
    if has_dist:
        dist = np.array(cam['distortion_coefficients'], dtype=np.float32)
    else:
        dist = 0.0

    return K, dist, cam['depth_name'], cam['depth_factor'], cam['image_dimension'][0], cam['image_dimension'][1]

def image_stream(sequence_path: Path, rgb_csv: Path, calibration_yaml: Path, 
                 cam_name: str = "rgb0", target_pixels: int = 384*512):
    """ image generator """ 
    global timestamps 
    K, dist, depth_name, depth_factor, w0 ,h0 = load_calibration(calibration_yaml, cam_name)

    # Load rgb images 
    df = pd.read_csv(rgb_csv)
    image_list = df[f'path_{cam_name}'].to_list()
    timestamps = df[f'ts_{cam_name} (s)'].to_list() 
    depth_list = df[f'path_{depth_name}'].to_list() 
    
    # Undistort and resize images
    h = (int(h0 * np.sqrt(target_pixels / (h0 * w0))) // 32) * 32
    w = (int(w0 * np.sqrt(target_pixels / (h0 * w0))) // 32) * 32
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w0, h0), 0, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, new_K, (w, h), cv2.CV_32FC1)
    intrinsics = torch.tensor([new_K[0,0], new_K[1,1], new_K[0,2], new_K[1,2], ], dtype=torch.float32)

    for t, imrel in enumerate(image_list): 
        impath = sequence_path / imrel
        depthpath = sequence_path / depth_list[t]

        image = cv2.imread(impath) 
        depth = cv2.imread(depthpath, cv2.IMREAD_ANYDEPTH) / depth_factor

        image = cv2.remap(image, mapx, mapy, interpolation=cv2.INTER_LINEAR)
        depth = cv2.remap(depth, mapx, mapy, interpolation=cv2.INTER_NEAREST)

        h, w, _ = image.shape 
        
        image = image[:h-h%8, :w-w%8] 
        image = torch.as_tensor(image).permute(2, 0, 1) 

        depth = torch.as_tensor(depth)
        depth = depth[:h-h%8, :w-w%8] 

        yield t, image[None], depth, intrinsics.clone() 

def main(): 
    print("\nRunning vslamlab_droidslam_rgbd.py ...\n")  

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--sequence_path", type=Path, required=True)
    parser.add_argument("--calibration_yaml", type=Path, required=True)
    parser.add_argument("--rgb_csv", type=Path, required=True)
    parser.add_argument("--exp_folder", type=Path, required=True)
    parser.add_argument("--exp_it", type=str, default="0")
    parser.add_argument("--settings_yaml", type=Path, default=None)
    parser.add_argument("--verbose", type=str, help="verbose")
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--weights", type=Path, default=None)

    args, _ = parser.parse_known_args()

    verbose = bool(int(args.verbose))
    args.disable_vis = not bool(int(args.verbose))
    args.upsample = bool(int(args.upsample))

    settings_path = args.settings_yaml
    if not os.path.exists(settings_path):
        settings_path = pkg_resources.resource_filename(
            'droid_slam.configs', 'vslamlab_droidslam-dev_settings.yaml'
        )

    with open(settings_path, 'r') as f:
        settings = yaml.safe_load(f)
    S = settings.get('settings', {})

    # Attach required settings to args
    args.t0 = int(S.get('t0', 0))
    args.stride = int(S.get('stride', 3))
    args.buffer = int(S.get('buffer', 512))
    args.beta = float(S.get('beta', 0.3))
    args.filter_thresh = float(S.get('filter_thresh', 2.4))
    args.warmup = int(S.get('warmup', 8))
    args.keyframe_thresh = float(S.get('keyframe_thresh', 4.0))
    args.frontend_thresh = float(S.get('frontend_thresh', 16.0))
    args.frontend_window = int(S.get('frontend_window', 25))
    args.frontend_radius = int(S.get('frontend_radius', 2))
    args.frontend_nms = int(S.get('frontend_nms', 1))
    args.backend_thresh = float(S.get('backend_thresh', 22.0))
    args.backend_radius = int(S.get('backend_radius', 2))
    args.backend_nms = int(S.get('backend_nms', 3))

    args.stereo = False 
    args.depth = True
    cam_name = str(S.get('cam_rgbd', "rgb0"))

    torch.multiprocessing.set_start_method('spawn') 
    
    droid = None
    for (t, image, depth, intrinsics) in tqdm(image_stream(args.sequence_path, args.rgb_csv, args.calibration_yaml, 
                                                            cam_name = cam_name)): 
        if t < args.t0: 
            continue 

        if verbose: 
            show_image(image[0]) 

        if droid is None: 
            args.image_size = [image.shape[2], image.shape[3]] 
            droid = Droid(args) 
            time.sleep(5)
            
        droid.track(t, image, depth, intrinsics=intrinsics)
    
    traj_est = droid.terminate(image_stream(args.sequence_path, args.rgb_csv, args.calibration_yaml, 
                                            cam_name = cam_name)) 

    keyframe_csv = args.exp_folder / f"{args.exp_it.zfill(5)}_KeyFrameTrajectory.csv"
    with open(keyframe_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
        for i in range(len(timestamps)):
            ts = timestamps[i]
            tx, ty, tz, qx, qy, qz, qw = traj_est[i][:7]
            writer.writerow([ts, tx, ty, tz, qx, qy, qz, qw])
    
    time.sleep(10)
    import signal
    os.killpg(os.getpgrp(), signal.SIGTERM)

if __name__ == '__main__': 
    main()