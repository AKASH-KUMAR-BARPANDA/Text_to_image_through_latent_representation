import torch.nn as nn

import torch.nn.functional as F

""" IMAGE ENCODER CLASS """
class ImageDecoder(nn.Module):

    def __init__(self, latent_dim=512):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 7 * 7)
        # 7 -> 14
        self.deconv1 = nn.ConvTranspose2d(128, 64,kernel_size=4,stride=2,padding=1)  
        # 14 -> 28
        self.deconv2 = nn.ConvTranspose2d(64, 64,kernel_size=4,stride=2,padding=1) 

        self.to_rgb = nn.Conv2d(
            64, 3,
            kernel_size=3,
            padding=1
        )

    def forward(self, z):

        x = self.fc(z)
        x = x.view(-1, 128, 7, 7) #(batch_size, channels, height, width)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.to_rgb(x)        # (B, 3, 28, 28)
        x = F.interpolate(
            x,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )
        return x