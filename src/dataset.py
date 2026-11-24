import random
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_fashion_mnist_loaders(batch_size=128, seed=42):
    set_seed(seed)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_raw = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_raw = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    # Gabungkan 70.000 citra lalu bagi persis 80% Train, 10% Val, 10% Test
    full_dataset = torch.utils.data.ConcatDataset([train_raw, test_raw])
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
