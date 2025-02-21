import torch
import torch.nn as nn
import torch.optim as optim
from optimizer import OptimizerWrapper
from mobilenetv2 import mobilenetv2
from phases import ValidationPhase, TrainPhase
from pruning import fine_grained_prune
import torch.multiprocessing as mp
from dataset import CIFAR10_loader
from pathlib import Path
from utils import num_params, save_state_dict
import pandas as pd
import argparse

def estimated_model_size(
    network: nn.Module,
    data_width: int = 32,
    only_nonzero: bool = False
) -> float:
    
    Byte = 8
    KiB = 1024 * Byte
    MiB = 1024 * KiB
    
    param_count = num_params(network, only_nonzero)
    size = param_count * data_width / MiB
    
    return size

def load_sparsity_dict(sparsity_dict: str) -> dict:
    
    sparsity_dict = Path(sparsity_dict)
    if not sparsity_dict.exists():
        raise ValueError(f"{sparsity_dict} doesn't exist.")
    
    f = pd.read_csv(sparsity_dict)

    # Ensure the necessary columns are present
    if 'name' not in f.columns or 'sparsity' not in f.columns:
        raise ValueError("The CSV file must contain 'name' and 'sparsity' columns.")

    # Generate the dictionary
    sparsity_mapping = {
        row['name']: row['sparsity']
        for _, row in f.iterrows()
    }

    return sparsity_mapping

class FineGrainedPruner:
    def __init__(self, model: nn.Module, sparsity_dict: dict):
        self.model = model
        self.masks = fine_grained_prune(model, sparsity_dict)
        self.apply()
    
    @torch.no_grad()
    def apply(self) -> None:
        for name, param in self.model.named_parameters():
            if name in self.masks:
                param.mul_(self.masks[name])

def prune(
    epochs: int,
    lr: float,
    batch_size: int,
    weights: str,
    sparsity_dict: str
):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = mobilenetv2(
        out_features=10,
        weights=weights,
        device=device
    )
    sparsity_dict = load_sparsity_dict(sparsity_dict)
    
    bef_ms = estimated_model_size(model)
    pruner = FineGrainedPruner(model=model, sparsity_dict=sparsity_dict)
    aft_ms = estimated_model_size(model, only_nonzero=True)
    print(f'Sparse model has size = {aft_ms:.2f} MiB = {aft_ms / bef_ms * 100:.2f}% of dense model size.')

    dataflow = CIFAR10_loader(train_size=None, test_size=None, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer_wrapper = OptimizerWrapper(
        optimizer_class=optim.Adam,
        device=device,
        params=list(model.parameters()),
        lr=lr,
        optimizer_args=None,
        weights=None
    )
    train_phase = TrainPhase(
        network = model,
        criterion = criterion,
        dataloader = dataflow['train'],
        device = device,
        optimizer = optimizer_wrapper,
        scheduler = None
    )
    val_phase = ValidationPhase(
        network = model,
        criterion = criterion,
        dataloader = dataflow['test'],
        device = device
    )

    best_val_acc = 0.0    
    for epoch in range(epochs):

        # The `Phase` instance will return a dictionary
        # with key `total_loss` and `accuracy`
        print(f"Epoch: {epoch}")
        train_result = train_phase()
        pruner.apply()
        
        val_result = val_phase()
        print(f"Accuracy: [Train] {train_result['accuracy']}% / [Valid] {val_result['accuracy']}%")
        print(" ---------- ---------- ---------- ")

        if best_val_acc < val_result['accuracy']:
            best_val_acc = val_result['accuracy']
            
            save_dir = Path(f'weights/epoch_{epoch}')
            save_dir.mkdir(parents=True, exist_ok=True)

            save_state_dict(model.state_dict(), save_dir / 'network.pth')
            save_state_dict(optimizer_wrapper.optimizer.state_dict(), save_dir / 'optimizer.pth')

if __name__ == '__main__':
    
    mp.set_start_method("spawn", force=True)
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--epochs', type=int, default=24, help='')
    parser.add_argument('--lr', type=float, default=1e-4, help='')
    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--weights', type=str, help='')
    parser.add_argument('--sparsity_dict', type=str, help='')
    args = parser.parse_args()
    
    prune(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        weights=args.weights,
        sparsity_dict=args.sparsity_dict
    )