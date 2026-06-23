import torch
import torch.nn as nn
from torchvision.models import resnet18,ResNet18_Weights

class ImageEncoder(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self.model = resnet18()
        # (512,1000) -> (512,512) -> 512 dimen output vector
        self.model.fc = nn.Identity()


    def forward(self,image):
        """ Convert the Image to 512 x 512 dimension vector"""
        output = self.model(image)
        return output



