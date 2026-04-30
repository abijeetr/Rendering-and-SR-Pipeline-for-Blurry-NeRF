import torch
import torch.nn as nn

class ESPCN(nn.Module):
    def __init__(self, upscale_factor=4):
        """
        Initializes the ESPCN model.
        :param upscale_factor: The factor to upscale the image (e.g., 4x)
        """
        super(ESPCN, self).__init__()
        self.upscale_factor = upscale_factor

        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding='same')
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding='same')
        
        # This is the key layer. It creates (upscale_factor^2) channels
        # for each output channel (RGB). For 4x scaling, 3 * (4*4) = 48 channels.
        self.conv3 = nn.Conv2d(32, 3 * (self.upscale_factor ** 2), kernel_size=3, padding='same')
        
        # The Pixel Shuffle layer
        self.pixel_shuffle = nn.PixelShuffle(self.upscale_factor)
        
        # ACtivation Fxn
        self.relu = nn.ReLU()

    def forward(self, x):
        # x is the low-res image
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        
        # Output Feature Maps
        x = self.conv3(x) 
        
        # Rearrange the channels into the final, upscaled image
        x = self.pixel_shuffle(x)
        
        return x