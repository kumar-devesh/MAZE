# dummy models to verify the train framework
# must be replaced by different generators and models


import torch
from torch import nn
from torchsummary.torchsummary import summary


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class SimpleGenerator(nn.Module):
    def __init__(self, latent_vector_dim=10, start_xydim=16, start_tdim=4, out_channels=3):
        super().__init__()
        start_channels = out_channels
        self.linear = nn.Linear(latent_vector_dim, start_channels * start_tdim * start_xydim * start_xydim)
        self.change_view = View((-1, start_channels, start_tdim, start_xydim, start_xydim))
        self.batch_norm1 = nn.BatchNorm3d(start_channels)
        self.upsample = nn.Upsample(scale_factor=2)
        self.tanh = nn.Tanh()
        self.batch_norm2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.linear(x)
        x = self.change_view(x)
        x = self.batch_norm1(x)
        x = self.upsample(x)
        x_pre = self.batch_norm2(x)
        x = self.tanh(x_pre)
        return x, x_pre


class SimpleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 32, (8, 32, 32))
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv3d(32, 1, (1, 1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        return x


class SimpleCNN3D(nn.Module):
    def __init__(self, t_dim=8, img_x=32, img_y=32, num_classes=5):
        super().__init__()
        self.conv = nn.Conv3d(3, 32, (t_dim, img_x, img_y))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


if __name__ == '__main__':
    summary(SimpleCNN3D().to('cuda'), (3, 8, 32, 32), )