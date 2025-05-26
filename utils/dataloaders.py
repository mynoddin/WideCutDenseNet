
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader

class Cutout(object):
    def __init__(self, mask_size=8, p=0.5):
        self.mask_size = mask_size
        self.p = p

    def __call__(self, img):
        if np.random.rand() > self.p:
            return img
        h, w = img.size(1), img.size(2)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.mask_size // 2, 0, h)
        y2 = np.clip(y + self.mask_size // 2, 0, h)
        x1 = np.clip(x - self.mask_size // 2, 0, w)
        x2 = np.clip(x + self.mask_size // 2, 0, w)

        img[:, y1:y2, x1:x2] = 0
        return img

def get_cifar10_dataloaders(batch_size=128, num_workers=2):
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        Cutout(mask_size=8, p=0.5),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
