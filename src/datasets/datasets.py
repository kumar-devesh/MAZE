import torch, torchvision
import random
from torch.utils.data import Dataset
import os
import pandas as pd
from torch.utils import data
import pickle
import sys
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets.folder import ImageFolder
import random
import os.path as osp
import numpy as np
from collections import defaultdict as dd
from PIL import Image
# from ml_datasets import get_dataloaders

classes_dict = {
    "kmnist": 10,
    "mnist": 10,
    "cifar10": 10,
    "cifar10_gray": 10,
    "cifar100": 100,
    "svhn": 10,
    "gtsrb": 43,
    "fashionmnist": 10,
    "fashionmnist_32": 10,
    "mnist_32": 10,
    "randomvideoslikekinetics400": 400,
    "randomvideoslikekinetics600": 600
}


def get_nclasses(dataset: str):
    if dataset in classes_dict:
        return classes_dict[dataset]
    else:
        raise Exception("Invalid dataset")


class GTSRB(Dataset):
    base_folder = "GTSRB"

    def __init__(self, train=False, transform=None):
        """
        Args:
            train (bool): Load trainingset or test set.
            root_dir (string): Directory containing GTSRB folder.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = "./data"

        self.sub_directory = "trainingset" if train else "testset"
        self.csv_file_name = "training.csv" if train else "test.csv"

        csv_file_path = os.path.join(
            self.root_dir, self.base_folder, self.sub_directory, self.csv_file_name
        )

        self.csv_data = pd.read_csv(csv_file_path)

        self.transform = transform

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.root_dir,
            self.base_folder,
            self.sub_directory,
            self.csv_data.iloc[idx, 0],
        )
        img = Image.open(img_path)

        classId = self.csv_data.iloc[idx, 1]

        if self.transform is not None:
            img = self.transform(img)

        return img, classId


# a dummy dataset, as a replacement until we include kinetics
# does not make sense to test, but just to complete the train and test
class RandomVideosLikeKinetics(Dataset):
    def __init__(self, train=True, n_classes=400):
        super().__init__()
        self.size = 100 if train else 40
        self.X = torch.randn((self.size, 32, 3, 224, 224))
        self.y = torch.randint(low=0, high=n_classes, size=(self.size,))

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def get_dataset(args, dataset, batch_size=256, augment=False, train_and_test=False):
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    num_workers = torch.get_num_threads()//2
    if dataset in ["mnist", "kmnist", "fashionmnist"]:
        if augment:
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(28, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5,), std=(0.5,)),
                ]
            )
        else:
            transform_train = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]
            )
        transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]
        )
    elif dataset in ["mnist_32", "fashionmnist_32"]:
        transform_train = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,)),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,)),
            ]
        )
    elif dataset in ["gtsrb"]:
        transform_train = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    elif dataset in ["cifar10_gray"]:
        if augment:
            transform_train = transforms.Compose(
                [
                    transforms.Grayscale(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5,), std=(0.5,)),
                ]
            )
        else:
            transform_train = transforms.Compose(
                [
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5,), std=(0.5,)),
                ]
            )

        transform_test = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,)),
            ]
        )

    else:
        if augment:
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        else:
            transform_train = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std),]
            )

        transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

    if dataset in ["mnist", "mnist_32"]:
        trainset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform_test
        )
    elif dataset in ["kmnist"]:
        trainset = torchvision.datasets.KMNIST(
            root="./data", train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.KMNIST(
            root="./data", train=False, download=True, transform=transform_test
        )

    elif dataset in ["fashionmnist", "fashionmnist_32"]:
        trainset = torchvision.datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.FashionMNIST(
            root="./data", train=False, download=True, transform=transform_test
        )

    elif dataset in ["cifar10", "cifar10_gray"]:
        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )

    elif dataset == "cifar100":
        trainset = torchvision.datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR100(
            root="./data", train=False, download=True, transform=transform_test
        )

    elif dataset == "svhn":
        trainset = torchvision.datasets.SVHN(
            root="./data", split="train", download=True, transform=transform_train
        )
        testset = torchvision.datasets.SVHN(
            root="./data", split="test", download=True, transform=transform_test
        )

    elif dataset == "gtsrb":
        trainset = GTSRB(train=True, transform=transform_train)
        testset = GTSRB(train=False, transform=transform_test)

    elif dataset == "randomvideoslikekinetics400":
        trainset = RandomVideosLikeKinetics(train=True, n_classes=400)
        testset = RandomVideosLikeKinetics(train=False, n_classes=400)
        batch_size = args.batch_size

    elif dataset == "randomvideoslikekinetics600":
        trainset = RandomVideosLikeKinetics(train=True, n_classes=600)
        testset = RandomVideosLikeKinetics(train=False, n_classes=600)
        batch_size = args.batch_size
        
    else:
        sys.exit("Unknown dataset {}".format(dataset))

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    if train_and_test:
        return trainloader, testloader

    return trainloader
