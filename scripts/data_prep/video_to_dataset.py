# ===== IMPORTS ======

import os
import sys
import shutil
import json
import numpy as np
import math
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ===== PATHS =====

IMAGE_DIR = Path("./images")
COLMAP_DIR = Path("./colmap")
COLMAP_DB_PATH = COLMAP_DIR / "colmap.db"
COLMAP_SPARSE_DIR = COLMAP_DIR / "sparse"
COLMAP_TEXT_DIR = COLMAP_DIR / "text"
OUTPUT_JSON = "transforms.json"

# print(IMAGE_DIR)
# print(COLMAP_DIR)
# print(COLMAP_DB_PATH)
# print(COLMAP_SPARSE_DIR)
# print(COLMAP_TEXT_DIR)
# print(OUTPUT_JSON)

# ===== HElPER FUNCTIONS ======

def qvec2rotmat(qvec):
    return np.array([
        [
            1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
        ], [
            2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
        ], [
            2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
        ]
    ])

def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if c < -1 + 1e-10:
        return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

def do_system(arg):
    """
    Helper function to run a shell command and exit if it fails.
    """
    print(f"==== RUNNING: {arg}")

    # Set the Qt platform to 'offscreen' to prevent GUI-related crashes
    # on headless servers like Lightning AI.
    cmd = f"QT_QPA_PLATFORM=offscreen {arg}"

    err = os.system(cmd)
    if err:
        print("FATAL: command failed")
        sys.exit(err)

def extract_images():
    """
    Uses FFMPEG to extract images from video files.
    """

    INPUT_VIDEO_FILE = os.getenv("INPUT_VIDEO_FILE")
    VIDEO_FPS = os.getenv("VIDEO_FPS")

    print(f"1. Extracting images from {INPUT_VIDEO_FILE}")
    if os.path.exists(IMAGE_DIR):
        print(f"Found existing {IMAGE_DIR} folder. Deleting it.")
        shutil.rmtree(IMAGE_DIR)
    os.makedirs(IMAGE_DIR)

    do_system(f"ffmpeg -i {INPUT_VIDEO_FILE} -qscale:v 1 -qmin 1 -vf \"fps={VIDEO_FPS}\" {IMAGE_DIR}/frame_%04d.jpg")
    print("Image extraction complete.")

def run_colmap():
    """
    Runs COLMAP's feature extractor, matcher, and mapper to find camera poses.
    """

    print("2. Running COLMAP")

    if os.path.exists(COLMAP_DIR):
        print(f"Found existing {COLMAP_DIR} folder. Deleting it.")
        shutil.rmtree(COLMAP_DIR)

    os.makedirs(COLMAP_SPARSE_DIR)
    os.makedirs(COLMAP_TEXT_DIR)

    # 1. Feature extraction
    do_system(f"colmap feature_extractor \
        --database_path {COLMAP_DB_PATH} \
        --image_path {IMAGE_DIR} \
        --ImageReader.camera_model OPENCV \
        --ImageReader.single_camera 1 \
        --SiftExtraction.use_gpu 0")

    # 2. Feature matching (sequential is good for videos)
    do_system(f"colmap sequential_matcher \
        --database_path {COLMAP_DB_PATH} \
        --SiftMatching.use_gpu 0")

    # 3. Mapping (bundle adjustment)
    do_system(f"colmap mapper \
        --database_path {COLMAP_DB_PATH} \
        --image_path {IMAGE_DIR} \
        --output_path {COLMAP_SPARSE_DIR}")

    # 4. Convert binary models to text
    do_system(f"colmap model_converter \
        --input_path {COLMAP_SPARSE_DIR}/0 \
        --output_path {COLMAP_TEXT_DIR} \
        --output_type TXT")
    print("COLMAP processing complete.")

def create_transforms_json():
    """
    Reads the COLMAP text files (cameras.txt, images.txt) and converts
    them into the transforms.json format required by NeRF.
    """
    print(f"3. Creating {OUTPUT_JSON}")
    
    # --- Read camera intrinsics from cameras.txt ---
    cameras = {}
    with open(os.path.join(COLMAP_TEXT_DIR, "cameras.txt"), "r") as f:
        for line in f:
            if line[0] == "#":
                continue
            els = line.split(" ")
            camera_id = int(els[0])
            camera = {}
            camera["w"] = float(els[2])
            camera["h"] = float(els[3])
            camera["fl_x"] = float(els[4])
            camera["fl_y"] = float(els[5])
            camera["cx"] = float(els[6])
            camera["cy"] = float(els[7])
            # NeRF assumes simple pinhole model, so we ignore distortion params
            camera["camera_angle_x"] = math.atan(camera["w"] / (camera["fl_x"] * 2)) * 2
            camera["camera_angle_y"] = math.atan(camera["h"] / (camera["fl_y"] * 2)) * 2
            cameras[camera_id] = camera

    if not cameras:
        print("FATAL: No cameras found in cameras.txt. COLMAP may have failed.")
        sys.exit(1)
    
    # Use the first camera's intrinsics for all frames
    # (since we used ImageReader.single_camera=1)
    main_cam = list(cameras.values())[0]
    out = {
        "camera_angle_x": main_cam["camera_angle_x"],
        "camera_angle_y": main_cam["camera_angle_y"],
        "fl_x": main_cam["fl_x"],
        "fl_y": main_cam["fl_y"],
        "cx": main_cam["cx"],
        "cy": main_cam["cy"],
        "w": main_cam["w"],
        "h": main_cam["h"],
        "aabb_scale": os.getenv("AABB_SCALE"),
        "frames": [],
    }

    # --- Read poses from images.txt ---
    up = np.zeros(3)
    bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
    
    with open(os.path.join(COLMAP_TEXT_DIR, "images.txt"), "r") as f:
        # Read two lines at a time (image metadata, then 3D points)
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            line = lines[i]
            if line[0] == "#":
                continue

            elems = line.split(" ")
            image_id = int(elems[0])
            qvec = np.array(tuple(map(float, elems[1:5])))
            tvec = np.array(tuple(map(float, elems[5:8])))
            image_name = elems[9].strip() # e.g., frame_0001.jpg
            
            # Convert COLMAP pose (camera-to-world) to NeRF (world-to-camera)
            R = qvec2rotmat(-qvec)
            t = tvec.reshape([3, 1])
            m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
            c2w = np.linalg.inv(m)

            # --- Reorient the scene for NeRF ---
            c2w[0:3, 2] *= -1  # flip the z axis
            c2w[0:3, 1] *= -1  # flip the y axis
            c2w = c2w[[1, 0, 2, 3], :] # swap y and z
            c2w[2, :] *= -1    # flip whole world upside down
            
            up += c2w[0:3, 1] # Accumulate "up" vectors

            frame = {
                "file_path": os.path.join(IMAGE_DIR, image_name),
                "transform_matrix": c2w
            }
            out["frames"].append(frame)

    nframes = len(out["frames"])
    if nframes == 0:
        print("FATAL: No frames were processed. COLMAP may have failed.")
        sys.exit(1)

    # --- Final scene re-centering and scaling ---
    
    # 1. Re-orient "up" to be [0, 0, 1]
    up = up / np.linalg.norm(up)
    R = rotmat(up, [0, 0, 1]) # rotate up vector to [0,0,1]
    R = np.pad(R, [0, 1])
    R[-1, -1] = 1
    for f in out["frames"]:
        f["transform_matrix"] = np.matmul(R, f["transform_matrix"])

    # 2. Find center of cameras
    totp = np.zeros(3)
    for f in out["frames"]:
        totp += f["transform_matrix"][0:3, 3]
    center = totp / nframes
    print(f"Computed camera center: {center}")
    
    # 3. Recenter cameras around the origin
    for f in out["frames"]:
        f["transform_matrix"][0:3, 3] -= center

    # 4. Scale scene to fit in a cube
    avglen = 0.
    for f in out["frames"]:
        avglen += np.linalg.norm(f["transform_matrix"][0:3, 3])
    avglen /= nframes
    print(f"Average camera distance from origin: {avglen}")
    
    scale_factor = 4.0 / avglen # "nerf-sized"
    for f in out["frames"]:
        f["transform_matrix"][0:3, 3] *= scale_factor
        f["transform_matrix"] = f["transform_matrix"].tolist() # Convert numpy to list for JSON

    # Sort frames by file path
    out["frames"].sort(key=lambda f: f["file_path"])

    # --- Write the final JSON file ---
    with open(OUTPUT_JSON, "w") as outfile:
        json.dump(out, outfile, indent=2)

    print(f"Successfully created {OUTPUT_JSON} with {nframes} frames.")

if __name__ == "__main__":
    # Check for FFMPEG and COLMAP
    if shutil.which("ffmpeg") is None:
        print("FATAL: ffmpeg is not in your PATH. Please install it.")
        sys.exit(1)
    if shutil.which("colmap") is None:
        print("FATAL: colmap is not in your PATH. Please install it.")
        sys.exit(1)

    extract_images()
    run_colmap()
    create_transforms_json()

    print(f"\n=== Pipeline Finished ===\n")
    print(f"Your NeRF dataset is ready!")
    print(f"Images: ./{IMAGE_DIR}")
    print(f"Transforms: ./{OUTPUT_JSON}\n")