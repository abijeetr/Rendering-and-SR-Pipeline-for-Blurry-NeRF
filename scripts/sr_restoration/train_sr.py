import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# Import our custom model and dataset
from sr_model import ESPCN
from sr_dataset import SRDataset

# ===== CONFIG =====

HR_IMAGE_DIR = "nerf_lego/lego/train" 

UPSCALE_FACTOR = 4
PATCH_SIZE = 128     # We'll use 128x128 HR patches
BATCH_SIZE = 64
EPOCHS = 100          
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = "escpn_lego.pth"

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Setup Dataset and DataLoader
    print("Loading dataset...")
    train_dataset = SRDataset(HR_IMAGE_DIR, UPSCALE_FACTOR, PATCH_SIZE)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4
    )

    # Setup Model, Loss, and Optimizer
    print("Setting up model...")
    model = ESPCN(upscale_factor=UPSCALE_FACTOR).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    print(f"--- Starting Training for {EPOCHS} Epochs ---")
    for epoch in range(EPOCHS):
        model.train() 
        epoch_loss = 0.0
        
        #tqdm for progress bar
        for lr_batch, hr_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)

            # --- Forward Pass ---
            optimizer.zero_grad()
            sr_batch = model(lr_batch)
            
            # --- Calculate Loss ---
            loss = criterion(sr_batch, hr_batch)
            
            # --- Backward Pass ---
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} Complete. Average Loss: {avg_loss:.6f}")

    # Save the trained model
    print(f"--- Training Finished ---")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()