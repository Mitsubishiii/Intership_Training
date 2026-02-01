"""
File to create PyTorch DataLoaders 
"""

import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

NUM_WORKERS = os.cpu_count()

def create_dataloaders(train_dir: str, 
                       test_dir: str,
                       transform: transforms.Compose,
                       batch_size: int,
                       num_workers: int=NUM_WORKERS,
                       dataset_type: str = None)->DataLoader:


    """
    Create for each dataset a set of iterables batches

    Arguments:
    #  - train_dir: Path to the trai data directory
    #  - test_dir: Path to the test data directory
     - transform: Torchvision transforms to perform on data
     - batch_size: Size of each batch (How many images per batch) in DataLoader
     - num_workers: Number of workers per DataLoader. Default all threads available.

    Returns:
     - A tuple of (train_dataloader, test_dataloader, class_names).
     Where class_names is a list of the target classes.
    """
    if dataset_type == "cifar10":
        train_data = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)
    else:
        # Ton code actuel pour les dossiers locaux
        train_data = datasets.ImageFolder(root=train_dir, transform=transform)
        test_data = datasets.ImageFolder(root=test_dir, transform=transform)

    # Get class names
    class_names = train_data.classes

    # Turn each dataset into a set of iterables batches (DataLoaders)
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)

    test_dataloader = DataLoader(dataset=test_data,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers)

    return (train_dataloader, test_dataloader, class_names)
