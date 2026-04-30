import torch
import cv2
import numpy as np
import torchvision.transforms as T
import argparse
import os
import sys
from glob import glob
from tqdm import tqdm

# Import our custom model
from sr_model import ESPCN

def upscale_image(model, lr_image_path, device):
    """Upscales a single image using the loaded model."""
    
    # 1. Load and process the low-res image
    lr_image = cv2.imread(lr_image_path)
    if lr_image is None:
        print(f"Warning: Could not read {lr_image_path}, skipping.")
        return None
    lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
    
    to_tensor = T.ToTensor()
    lr_tensor = to_tensor(lr_image).unsqueeze(0).to(device) # Add batch dimension
    
    # 2. Run inference
    with torch.no_grad():
        sr_tensor = model(lr_tensor)
        
    # 3. Convert back to an image
    # Clamp values to [0, 1] before converting back to [0, 255]
    sr_tensor = sr_tensor.squeeze(0).cpu().clamp(0, 1) 
    sr_image_np = (sr_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    sr_image_bgr = cv2.cvtColor(sr_image_np, cv2.COLOR_RGB2BGR)
    
    return sr_image_bgr

def main():
    parser = argparse.ArgumentParser(description='Upscale images using ESPCN model.')
    parser.add_argument('--input', type=str, required=True, help='Path to low-res image or folder.')
    parser.add_argument('--output', type=str, required=True, help='Path to save SR image or folder.')
    parser.add_argument('--weights', type=str, default='escpn_lego.pth', help='Model weights file.')
    parser.add_argument('--upscale_factor', type=int, default=4, help='Upscale factor (must match model).')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Model
    if not os.path.exists(args.weights):
        print(f"FATAL: Weights file not found at {args.weights}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Loading model from {args.weights}...")
    model = ESPCN(upscale_factor=args.upscale_factor).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval() # Set to evaluation mode

    # 2. Check if input is a single file or a folder
    if os.path.isfile(args.input):
        print(f"Upscaling single image: {args.input}")
        sr_image = upscale_image(model, args.input, device)
        if sr_image is not None:
            cv2.imwrite(args.output, sr_image)
            print(f"Saved to {args.output}")
        
    elif os.path.isdir(args.input):
        print(f"Upscaling all images in folder: {args.input}")
        os.makedirs(args.output, exist_ok=True)
        
        lr_files = sorted(glob(os.path.join(args.input, "*.png")))
        lr_files.extend(sorted(glob(os.path.join(args.input, "*.jpg"))))
        
        for lr_path in tqdm(lr_files, desc="Upscaling frames"):
            sr_image = upscale_image(model, lr_path, device)
            
            if sr_image is not None:
                file_name = os.path.basename(lr_path)
                save_path = os.path.join(args.output, file_name)
                cv2.imwrite(save_path, sr_image)
            
        print(f"All frames upscaled and saved to {args.output}")
    
    else:
        print(f"Error: Input path not found: {args.input}", file=sys.stderr)

if __name__ == "__main__":
    main()