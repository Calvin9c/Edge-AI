import torch
import torch.nn as nn
import torch.multiprocessing as mp
from phases import ValidationPhase
from mobilenetv2 import mobilenetv2
from dataset import CIFAR10_loader
import argparse

def evaluate(
    weights: str
):
    
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.verbose = True
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    network = mobilenetv2(
        out_features=10,
        weights=weights,
        device = device
    )

    dataflow = CIFAR10_loader(train_size=None, test_size=None, batch_size=32)

    validation_phase = ValidationPhase(
        network = network,
        criterion = nn.CrossEntropyLoss(),
        dataloader = dataflow['test'],
        device = device
    )

    output_dict = validation_phase()
    print(output_dict)

if __name__ == "__main__":
    
    mp.set_start_method("spawn", force=True)
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--weights', type=str, help='')
    args = parser.parse_args()
    
    evaluate(weights=args.weights)