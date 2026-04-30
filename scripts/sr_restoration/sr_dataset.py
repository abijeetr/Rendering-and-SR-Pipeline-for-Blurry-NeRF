import torch
import cv2
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset
from glob import glob
import random
import os

class SRDataset(Dataset):
    def __init__(self, hr_image_dir, upscale_factor=4, patch_size=128):
        """

        Args:
        hr_image_dir: Directory containing high-res images.
        upscale_factor: How much to scale (e.g., 4x).
        patch_size: The size of the high-res patch to crop (e.g., 128x128).
        """
        super(SRDataset, self).__init__()
        
        # Find all .png or .jpg files
        self.image_files = glob(os.path.join(hr_image_dir, "*.png"))
        if not self.image_files:
            self.image_files = glob(os.path.join(hr_image_dir, "*.jpg"))
            
        print(f"Found {len(self.image_files)} images for SR training in {hr_image_dir}")
        self.upscale_factor = upscale_factor
        self.hr_patch_size = patch_size
        self.lr_patch_size = patch_size // upscale_factor
        
        # Define the image-to-tensor transformation
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        # 1. Load the high-res image
        hr_image = cv2.imread(self.image_files[index])
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB) # Convert to RGB
        
        # 2. Crop a random patch from the high-res image
        h, w, _ = hr_image.shape
        rand_h = random.randint(0, h - self.hr_patch_size)
        rand_w = random.randint(0, w - self.hr_patch_size)
        
        hr_patch = hr_image[
            rand_h : rand_h + self.hr_patch_size,
            rand_w : rand_w + self.hr_patch_size
        ]

        # 3. Create the low-res patch by downscaling
        lr_patch = cv2.resize(
            hr_patch, 
            (self.lr_patch_size, self.lr_patch_size), 
            interpolation=cv2.INTER_CUBIC # Standard downscaling
        )
        
        # 4. Convert to PyTorch tensors
        hr_tensor = self.to_tensor(hr_patch)
        lr_tensor = self.to_tensor(lr_patch)
        
        return lr_tensor, hr_tensor