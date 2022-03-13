import torch
import sys
import torch.nn as nn
import os.path as osp
import torchvision.models as torchmodels
import torch.nn.functional as F
from movinets import MoViNet
from movinets.config import _C

from . import (
    conv3,
    lenet,
    wresnet,
    resnet,
    conv3_gen,
    conv3_cgen,
    conv3_dis,
    conv3_mnist,
    simple_models,
    Generator,
    Discriminator
)
from .cifar10_models import resnet18, vgg13_bn
from datasets import get_nclasses


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


model_dict = {
    "Generator_cgen": Generator.Generator,
    #"Discriminator": Discriminator.TemporalDiscriminator,
    "Discriminator": Discriminator.SpatialDiscriminator,
    "conv3_gen": conv3_gen.conv3_gen,
    "conv3_cgen": conv3_cgen.conv3_cgen,
    "conv3_dis": conv3_dis.conv3_dis,
    "lenet": lenet.lenet,
    "conv3": conv3.conv3,
    "conv3_mnist": conv3_mnist.conv3_mnist,
    "wres22": wresnet.WideResNet,
    "res20": resnet.resnet20,
    "res18_ptm": resnet18,
    "vgg13_bn": vgg13_bn,
    "simple_cnn3d": simple_models.SimpleCNN3D,
    "simple_gen": simple_models.SimpleGenerator,
    "simple_dis": simple_models.SimpleDiscriminator,
    "ResNet3d_T": torchmodels.video.r3d_18,
    "ResNet3d_S": torchmodels.video.r3d_18,
    "movinet_T": MoViNet,
    "movinet_S": MoViNet,
}

gen_channels_dict = {
    "mnist": 1,
    "cifar10": 3,
    "cifar100": 3,
    "gtsrb": 3,
    "svhn": 3,
    "fashionmnist": 1,
}

gen_dim_dict = {
    "cifar10": 8,
    "cifar100": 8,
    "gtsrb": 8,
    "svhn": 8,
    "mnist": 7,
    "fashionmnist": 7,
}

in_channel_dict = {
    "cifar10": 3,
    "cifar100": 3,
    "gtsrb": 3,
    "svhn": 3,
    "mnist": 1,
    "fashionmnist": 1,
}

class ResNet3d_wrapper(nn.Module):
    def __init__(self, pretrained = True, n_classes = 400):
        super(ResNet3d_wrapper, self).__init__()
        self.pretrained = pretrained
        self.model = torchmodels.video.r3d_18(pretrained=pretrained)
        self.linear = nn.Linear(in_features=400, out_features=n_classes)
        self.n_classes = n_classes
        self.sm = nn.Softmax(dim=-1)
        
    def forward(self, x):
        # take a transpose for model input
        # x = torch.transpose(x, dim0=1, dim1=2) #input 3,T,H,W

        #if self.n_classes == 400:
        #    return self.sm(self.model(x))
        #return self.sm(self.linear(self.model(x)))

        if self.n_classes == 400:
            return self.model(x)
        return self.linear(self.model(x))

class MoviNet_wrapper(nn.Module):
    def __init__(self, pretrained = True, n_classes = 600):
        super(MoviNet_wrapper, self).__init__()
        self.pretrained = pretrained
        self.model = MoViNet(_C.MODEL.MoViNetA2, causal = True, pretrained = pretrained)
        #self.linear = nn.Linear(in_features=400, out_features=n_classes)
        self.n_classes = n_classes
        self.sm = nn.Softmax(dim=-1)
        
    def forward(self, x, print_outputs=False):
        # take a transpose for model input
        # x = torch.transpose(x, dim0=1, dim1=2) #input 3,T,H,W
        #if self.n_classes == 600:
        #    return self.sm(self.model(x))
        #return self.linear(self.model(x))
        if print_outputs==True:
            return self.sm(self.model(x))
        return self.model(x)

def get_model(args, modelname="Generator", n_classes=400, dataset="", pretrained=None, latent_dim=10, **kwargs):
    model_fn = model_dict[modelname]
    num_classes = n_classes

    if modelname == "Generator_cgen":
        model = model_fn(in_dim=120, latent_dim=args.latent_dim, n_class=n_classes, ch=args.gen_channel, n_frames=args.n_frames, hierar_flag=False) #generator default params
    
    elif modelname == "Discriminator":
        model = model_fn() #discriminator default params
    
    elif modelname == "ResNet3d_T":
        model = ResNet3d_wrapper(pretrained=True)
    
    elif modelname == "ResNet3d_S":
        model = ResNet3d_wrapper(pretrained=False, n_classes=args.n_classes)

    elif modelname == "movinet_T":
        model = MoviNet_wrapper(pretrained = True)
    
    elif modelname == "movinet_S":
        model = MoviNet_wrapper(pretrained = False)

    elif modelname in [
        "conv3",
        "lenet",
        "res20",
        "conv3_mnist",
    ]:
        model = model_fn(num_classes)

    elif modelname == "wres22":
        if dataset in ["mnist", "fashionmnist"]:
            model = model_fn(
                depth=22,
                num_classes=num_classes,
                widen_factor=2,
                dropRate=0.0,
                upsample=True,
                in_channels=1,
            )
        else:
            model = model_fn(
                depth=22, num_classes=num_classes, widen_factor=2, dropRate=0.0
            )
    elif modelname in ["conv3_gen"]:
        model = model_fn(
            z_dim=latent_dim,
            start_dim=gen_dim_dict[dataset],
            out_channels=gen_channels_dict[dataset],
        )
    elif modelname == "conv3_gen_3d":
        model = model_fn(
            z_dim=latent_dim,
            start_tdim=4,
            start_dim=10,
            out_channels=3
        )
    elif modelname in ["conv3_cgen"]:
        model = model_fn(
            z_dim=latent_dim,
            start_dim=gen_dim_dict[dataset],
            out_channels=gen_channels_dict[dataset],
            n_classes=num_classes,
        )
    elif modelname in ["conv3_dis"]:
        model = model_fn(channels=gen_channels_dict[dataset], dataset=dataset)
    elif modelname in ["res18_ptm", "vgg13_bn"]:
        model = model_fn(pretrained=pretrained)

    elif modelname in "simple_cnn3d":
        model = model_fn(t_dim=8, img_x=32, img_y=32, num_classes=5)
    elif modelname == "simple_gen":
        model = model_fn(latent_vector_dim=args.latent_dim, start_xydim=112, start_tdim=4, out_channels=3)
    elif modelname == "simple_dis":
        model = model_fn(args.batch_size, n_classes=400)

    else:
        sys.exit("unknown model")

    return model
  