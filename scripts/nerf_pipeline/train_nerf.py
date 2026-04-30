import subprocess
import os
import sys

# ===== CONFIG =====
# The name for the specific model. 
# This determines where the output config is saved.
EXPERIMENT_NAME = "my-scene-nerf"

# The path to the dataset 
# (the folder containing 'transforms.json' and 'images/')
DATA_DIR = "./nerf_lego/lego/"

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

# ===== MAIN TRAINING =====

if __name__ == "__main__":
    print(f"Starting NeRF Training for: {EXPERIMENT_NAME}")

    # This is the main training command
    train_command = f"""
    ns-train instant-ngp \
    --experiment-name {EXPERIMENT_NAME} \
    --vis tensorboard \
    --pipeline.model.near-plane 0.05 \
    --pipeline.model.far-plane 1000 \
    --max-num-iterations 30000 \
    --pipeline.datamanager.train-num-rays-per-batch 2048 \
    --steps-per-eval-batch 0 \
    --steps-per-eval-image 0 \
    blender-data \
    --data {DATA_DIR}
    """
    # Note: 30,000 iterations is the default. 
    # For a quick test use --max-num-iterations 5000

    do_system(train_command)

    # Nerfstudio saves model config in 'outputs/EXPERIMENT_NAME/MODEL_NAME/config.yml'
    config_path = os.path.join("outputs", EXPERIMENT_NAME, "instant-ngp", "config.yml")
    
    if os.path.exists(config_path):
        print(f"--- Training complete! ---")
        print(f"Model config saved to: {config_path}")
    else:
        print(f"FATAL: Could not find trained config at {config_path}")
