import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.optim as optim
from optimizer import OptimizerWrapper
from phases import TrainPhase, ValidationPhase
from mobilenetv2 import mobilenetv2
from dataset import CIFAR10_loader
from pathlib import Path
from utils import save_state_dict
import argparse

def train(
    epochs: int,
    lr: float,
    batch_size: int
):

    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.verbose = True

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    network = mobilenetv2(
        out_features=10,
        weights=None,
        device=device
    )
    optimizer_wrapper = OptimizerWrapper(
        optimizer_class=optim.Adam,
        device=device,
        params=list(network.parameters()),
        lr=lr,
        optimizer_args=None,
        weights=None
    )
    dataflow = CIFAR10_loader(train_size=None, test_size=None, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    train_phase = TrainPhase(
        network=network,
        criterion=criterion,
        dataloader=dataflow['train'],
        device=device,
        optimizer=optimizer_wrapper,
        scheduler=None
    )
    validation_phase = ValidationPhase(
        network=network,
        criterion=criterion,
        dataloader=dataflow['test'],
        device=device
    )

    best_val_acc = 0.0    
    for epoch in range(epochs):
        
        # The `Phase` instance will return a dictionary
        # with key `total_loss` and `accuracy`
        train_result = train_phase()
        val_result = validation_phase()
        print(f"Accuracy: [Train] {train_result['accuracy']}% / [Valid] {val_result['accuracy']}%")

        if best_val_acc < val_result['accuracy']:
            best_val_acc = val_result['accuracy']
            
            save_dir = Path(f'weights/epoch_{epoch}')
            save_dir.mkdir(parents=True, exist_ok=True)

            save_state_dict(network.state_dict(), save_dir / 'network.pth')
            save_state_dict(optimizer_wrapper.optimizer.state_dict(), save_dir / 'optimizer.pth')

if __name__ == '__main__':
    
    mp.set_start_method("spawn", force=True)
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--epochs', type=int, default=24, help='')
    parser.add_argument('--lr', type=float, default=1e-4, help='')
    parser.add_argument('--batch_size', type=int, default=32, help='')
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size
    )