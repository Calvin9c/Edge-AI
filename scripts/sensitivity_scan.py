import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, repo_dir)

import torch
import torch.nn as nn
import numpy as np
from mobilenetv2 import mobilenetv2
from phases import ValidationPhase
from pruning import *
import torch.multiprocessing as mp
from dataset import CIFAR10_loader
import pandas as pd
import argparse

def _sparsities(start: float, end: float, step: float) -> np.ndarray:
    return np.arange(start=start, stop=end, step=step)

@torch.no_grad()
def sensitivity_scan(
    weights: str,
    accuracy_threshold: float
) -> None:

    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.verbose = True

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = mobilenetv2(
        out_features=10,
        weights=weights,
        device=device
    )
    dataflow = CIFAR10_loader(train_size=None, test_size=None, batch_size=32)
    
    val_phase = ValidationPhase(
        network = model,
        criterion = nn.CrossEntropyLoss(),
        dataloader = dataflow['test'],
        device = device
    )
    
    scanning_log = []

    for name, param in model.named_parameters():

        if param.dim() <= 1:
            continue

        print(f'Scanning {name}...')
        param_clone = param.detach().clone()

        recommand_sparsity = 0.0
        accuracy = []
        for sparsity in _sparsities(start=0.5, end=1.0, step=0.2):

            mask = magnitude_base_pruning(param.detach(), sparsity)
            param.mul_(mask)
            
            result = val_phase() # type(result) = dict, with key `total_loss` & `accuracy`
            accuracy.append(round(result['accuracy'], 2))
            if accuracy_threshold <= accuracy[-1]:
                recommand_sparsity = round(sparsity, 4)

            # restore
            param.copy_(param_clone)
        print(' ---------- ---------- ---------- ')

        scanning_log.append({
            'name': name,
            'accuracy': accuracy,
            'recommand_sparsity': recommand_sparsity
        })
    
    df = pd.DataFrame(scanning_log)
    df.to_csv('scanning_log.csv', index=False)

if __name__ == '__main__':
    
    mp.set_start_method("spawn", force=True)
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--weights', type=str, help='')
    parser.add_argument('--accuracy_threshold', type=float, default=75, help='')
    args = parser.parse_args()

    sensitivity_scan(
        weights=args.weights,
        accuracy_threshold=args.accuracy_threshold
    )