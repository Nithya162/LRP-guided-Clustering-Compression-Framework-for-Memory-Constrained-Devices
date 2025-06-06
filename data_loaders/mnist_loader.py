import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple

def get_mnist_loaders(batch_size_train: int = 128, 
                     batch_size_test: int = 1000) -> Tuple[DataLoader, DataLoader]:
    """
    Get MNIST data loaders for training and testing
    
    Args:
        batch_size_train: Batch size for training
        batch_size_test: Batch size for testing
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_ds = datasets.MNIST('./data_mnist', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size_train, shuffle=True, 
                            num_workers=2, pin_memory=torch.cuda.is_available())
    
    test_ds = datasets.MNIST('./data_mnist', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size_test, shuffle=False, 
                           num_workers=2, pin_memory=torch.cuda.is_available())
    
    return train_loader, test_loader