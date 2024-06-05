import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.datasets import MNIST
import os
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F


def show_images_dataloader(out: str, data: DataLoader, cols: int = 4):
    imgs = next(iter(data))
    """ Plots some samples from the dataset """
    fig = plt.figure(figsize=(15, 15))
    for i, img in enumerate(imgs):
        plt.subplot(int(len(imgs) / cols) + 1, cols, i + 1)
        plt.imshow(np.array((img.permute(1, 2, 0) + 1) / 2 * 255, dtype=int))
        plt.axis("off")
    plt.tight_layout()
    fig.savefig(out)
    plt.close("all")


def show_images_batch(out: str, data: torch.Tensor, cols: int = 4):
    """Plots some samples from the dataset"""
    fig = plt.figure(figsize=(15, 15))
    for i in range(data.shape[0]):
        plt.subplot(int(data.shape[0] / cols) + 1, cols, i + 1)
        # img = np.array((data[i].permute(1,2,0)+1)/2*255, dtype=int)
        # img = np.clip(img, 0, 255)
        img = np.array(data[i].permute(1, 2, 0), dtype=int)
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.axis("off")
    plt.tight_layout()
    if out is not None:
        fig.savefig(out)


def cifar_data_transform(img_size=32):
    transform = [
        transforms.Grayscale(),
        transforms.Resize((img_size, img_size)),
        # transforms.ToTensor(),
        transforms.Lambda(lambda t: t / 255),  # scale to [0, 1]
        transforms.Lambda(lambda t: t + 1),  # Scale between [1, 2] for log
        # transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1]
    ]
    transform = transforms.Compose(transform)
    return transform

class SquarePad:
    def __call__(self, image):
        # print(image.size())
        _, w, h = image.size()
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, 'constant')

def celeba_data_transform(img_size=224):
    transform = [
        SquarePad(),
        transforms.Grayscale(),
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        # transforms.ToTensor(),
        transforms.Lambda(lambda t: t / 255),  # scale to [0, 1]
        transforms.Lambda(lambda t: t + 1),  # Scale between [1, 2] for log
        # transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1]
    ]
    transform = transforms.Compose(transform)
    return transform


def mnist_data_transform(img_size=32):
    transform = [
        transforms.Resize((img_size, img_size)),
        transforms.Lambda(lambda t: t / 255),  # scale to [0, 1]
        transforms.Lambda(lambda t: (t * 2) - 1),  # Scale between [-1, 1]
    ]
    transform = transforms.Compose(transform)
    return transform


class CelebADataset(Dataset):
    def __init__(self, img_dir, transform):
        self.img_dir = img_dir
        self.img_paths = []
        self.transform = transform
        for f in os.listdir(self.img_dir):
            if f.endswith("jpg"):
                self.img_paths.append(os.path.join(self.img_dir, f))

        self.len = len(self.img_paths)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        assert idx >= 0
        assert idx < self.len
        image = read_image(self.img_paths[idx])
        if self.transform is not None:
            image = self.transform(image)
        return image, 0


class CifarDataset(Dataset):
    def __init__(self, img_dir, classes, transform):
        self.img_dir = img_dir
        self.classes = classes
        if self.classes == "all":
            # default to all CIFAR classes
            self.classes = [
                "airplane",
                "automobile",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck",
            ]
        self.img_paths = []
        self.transform = transform
        for c in self.classes:
            for f in os.listdir(os.path.join(self.img_dir, c)):
                if f.endswith("jpg"):
                    self.img_paths.append(os.path.join(self.img_dir, c, f))

        self.len = len(self.img_paths)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        assert idx >= 0
        assert idx < self.len
        image = read_image(self.img_paths[idx])
        if self.transform is not None:
            image = self.transform(image)
        return image, 0


class MnistDataset(Dataset):
    def __init__(self, img_dir, classes, transform):
        self.img_dir = img_dir
        self.classes = classes
        if self.classes == "all":
            # default to all CIFAR classes
            self.classes = [str(c) for c in range(10)]  # ten digits
        self.img_paths = []
        self.transform = transform
        for c in self.classes:
            for f in os.listdir(os.path.join(self.img_dir, c)):
                if f.endswith("jpg"):
                    self.img_paths.append(os.path.join(self.img_dir, c, f))

        self.len = len(self.img_paths)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        assert idx >= 0
        assert idx < self.len
        image = read_image(self.img_paths[idx])
        if self.transform is not None:
            image = self.transform(image)
        return image


if __name__ == "__main__":
    data_transforms = cifar_data_transform(img_size=32)
    dataset = CifarDataset(
        img_dir="/home/anvuong/Desktop/datasets/CIFAR-10-images/train",
        classes="all",
        transform=data_transforms,
    )

    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    show_images_dataloader(loader)
