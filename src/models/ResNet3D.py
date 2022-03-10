import torchvision.models as torchmodels
import torch.nn as nn

class ResNet3D(nn.Module):
    def __init__(self, pretrained = True):
        super(ResNet3D, self).__init__()
        self.pretrained = pretrained
        self.model = torchmodels.video.r3d_18(pretrained=pretrained)
        
    def forward(self, x):
        # take a transpose for model input
        # x = torch.transpose(x, dim0=1, dim1=2) #input 3,T,H,W
        return self.model(x)