from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets
from torchvision import transforms

import torch
import numpy as np

def data_loader10(data_dir,
                batch_size,
                random_seed=42,
                valid_size=0.1,
                shuffle=True,
                test=False):

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # Images Regularization
    transform_original = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        normalize,
    ])

    # Image Augmentation
    transform_aug = transforms.Compose([
        transforms.RandomResizedCrop(256, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.1)
    ])

    if test:
        dataset = datasets.CIFAR10(
            root=data_dir, train=False,
            download=True, transform=transform_original
        )

        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )

    # Regular Dataset
    train_dataset_orig = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_original
    )

    # Augmented Dataset
    train_dataset_aug = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_aug
    )


    # Unify the training sets and separate their data from the validation set
    num_train = len(train_dataset_orig)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]


    train_dataset = ConcatDataset([
        torch.utils.data.Subset(train_dataset_orig, train_idx),
        torch.utils.data.Subset(train_dataset_aug, train_idx)
    ])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    valid_dataset = torch.utils.data.Subset(train_dataset_orig, valid_idx)
    valid_loader = DataLoader(
        valid_dataset, batch_size=1, shuffle=False, num_workers=4
    )

    return train_loader, valid_loader