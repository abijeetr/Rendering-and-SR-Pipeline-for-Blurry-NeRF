#!/usr/bin/env python3
import subprocess
import os
import sys
import shutil

# ===== CONFIG =====

# The name of the model
EXPERIMENT_NAME = "my-scene-nerf"

# The directory to save the rendered low-res frames
OUTPUT_DIR = "low_res_frames"

# ===== HELPER FUNCTIONS =====
def do_system(arg):
    """
    Helper function to run a shell command and exit if it fails.
    """
    print(f"==== RUNNING: {arg}")
    
    cmd = f"QT_QPA_PLATFORM=offscreen {arg}"
    
    err = os.system(cmd)
    if err:
        print("FATAL: command failed")
        sys.exit(err)

# ===== RENDERING ====

if __name__ == "__main__":
    print(f"--- 1. Starting NeRF Render for: {EXPERIMENT_NAME} ---")

    # ---  Find the saved model config ---
    config_path = r"outputs/my-scene-nerf/instant-ngp/2025-11-02_075531/config.yml"
    
    if not os.path.exists(config_path):
        print(f"FATAL: Could not find trained config at {config_path}")
        sys.exit(1)

    # ---  Clean up old renders and create output directory ---
    if os.path.exists(OUTPUT_DIR):
        print(f"Found existing {OUTPUT_DIR} folder. Deleting it.")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    # ---  Run the render command ---
    # This command renders a smooth interpolated path between 
    # all the training camera poses.
    render_command = f"ns-render interpolate --load-config {config_path} --output-path {OUTPUT_DIR} --output-format images"
    # This will create 'camera_path.json' and a folder
    # 'low_res_frames' full of images (e.g., frame_00000.png)

    do_system(render_command)

    print(f"--- Rendering complete! ---")
    print(f"Low-res frames saved to: ./{OUTPUT_DIR}")