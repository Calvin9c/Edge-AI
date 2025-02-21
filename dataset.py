import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def _subset_indices(
    dataset_size: int,
    subset_size: int
) -> list:
    
    if subset_size is None:
        return list(range(dataset_size))

    if subset_size > dataset_size:
        raise ValueError("Subset size cannot exceed dataset size.")
    
    indices = torch.randperm(dataset_size)[:subset_size]
    return indices.tolist()

def CIFAR10_loader(
    train_size: int,
    test_size: int,
    batch_size: int = 32,
    num_workers: int = 2,
) -> dict:
    
    SPLITS = ['train', 'test']
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataflow = {}
    for split in SPLITS:
        
        dataset = datasets.CIFAR10(root='./CIFAR10', train=(split=='train'), download=True, transform=transform)
        indices = _subset_indices(len(dataset), train_size if split == 'train' else test_size)
        subset = Subset(dataset, indices)
        
        dataflow[split] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=(split=='train'),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    return dataflow