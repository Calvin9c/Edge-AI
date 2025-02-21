"""reference: https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html#post-training-static-quantization"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
from mobilenetv2 import mobilenetv2_quantization
from dataset import CIFAR10_loader
from pathlib import Path
import warnings
from utils import evaluate, module_size, save_torchscript_module
import argparse

warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.ao.quantization'
)

def get_model(weights: str) -> nn.Module:
    
    fp_network = mobilenetv2_quantization(10, weights)

    # We'll "fuse modules"; this can both make the model faster by saving on memory access
    # while also improving numerical accuracy. While this can be used with any model, this is
    # especially common with quantized models.

    # print('\nInverted Residual Block: Before fusion \n\n',
    #       fp_network.net.features[1].conv)
    fp_network.eval()
    fp_network.fuse_model() # Fuses modules
                            # Note fusion of Conv+BN+Relu and Conv+Relu
    # print('\nInverted Residual Block: After fusion\n\n',
    #       fp_network.net.features[1].conv)

    return fp_network

def baseline(
    fp_network: nn.Module,
    criterion: nn.Module,
    evaluate_loader: DataLoader,
    num_eval_batches: int,
    batch_size: int
) -> None:
    
    print("\n========== ========== ==========")
    print("Baseline")
    print("========== ========== ==========\n")
    
    print(f"Size of baseline model: {round(module_size(fp_network), 2)} MB")
    top1, top5 = evaluate(fp_network, criterion, evaluate_loader, neval_batches=num_eval_batches)
    print(f'Evaluation accuracy on {num_eval_batches*batch_size} images, {round(top1.avg.item(), 4)}%')
    save_torchscript_module(fp_network, "scripted_weights/fp_model.pth")

def ptq(
    fp_network: nn.Module,
    criterion: nn.Module,
    calibrate_loader: DataLoader,
    num_calibration_batches: int,
    evaluate_loader: DataLoader,
    num_eval_batches: int,
    batch_size: int
) -> None:
    
    print("\n========== ========== ==========")
    print("Post Training Quantization")
    print("========== ========== ==========\n")

    # Specify quantization configuration
    # Start with simple min/max range estimation and per-tensor quantization of weights
    fp_network.qconfig = torch.ao.quantization.default_qconfig
    # print(fp_network.qconfig)
    
    print("[Inserting Observers] ", end='')
    torch.ao.quantization.prepare(fp_network, inplace=True)
    print("done\n")
    
    # print('Inverted Residual Block: After observer insertion \n',
    #       fp_network.net.features[1].conv, "\n")
    
    # Calibrate
    print("[Calibrate with the training set] ", end='')
    evaluate(fp_network, criterion, calibrate_loader, num_calibration_batches)
    print("done\n")

    # Convert
    # You may see a user warning about needing to calibrate the model. This warning can be safely ignored.
    # This warning occurs because not all modules are run in each model runs, so some
    # modules may not be calibrated.
    print("[Convert to quantized model] ", end='')
    torch.ao.quantization.convert(fp_network, inplace=True)
    print("done\n")

    # print('Inverted Residual Block: After fusion and quantization, note fused modules: \n',
    #       fp_network.net.features[1].conv, "\n")

    print(f"Size of model after quantization: {round(module_size(fp_network), 2)} MB")
    top1, top5 = evaluate(fp_network, criterion, evaluate_loader, neval_batches=num_eval_batches)
    print(f'Evaluation accuracy on {num_eval_batches*batch_size} images, {round(top1.avg.item(), 4)}%')
    save_torchscript_module(fp_network, "scripted_weights/quant_ptq.pth")

def perchannel_ptq(
    fp_network: nn.Module,
    criterion: nn.Module,
    calibrate_loader: DataLoader,
    num_calibration_batches: int,
    evaluate_loader: DataLoader,
    num_eval_batches: int,
    batch_size: int
):
    print("\n========== ========== ==========")
    print("Per-Channel PTQ")
    print("========== ========== ==========\n")

    # The old 'fbgemm' is still available but 'x86' is the recommended default.
    fp_network.qconfig = torch.ao.quantization.get_default_qconfig('x86')
    # print(fp_network.qconfig)

    print("[Inserting Observers] ", end='')
    torch.ao.quantization.prepare(fp_network, inplace=True)
    print("done\n")

    # Calibrate
    print("[Calibrate with the training set] ", end='')
    evaluate(fp_network, criterion, calibrate_loader, num_calibration_batches)
    print("done\n")
    
    # Convert
    print("[Convert to quantized model] ", end='')
    torch.ao.quantization.convert(fp_network, inplace=True)
    print("done\n")
    
    print(f"Size of model after quantization: {round(module_size(fp_network), 2)} MB")
    top1, top5 = evaluate(fp_network, criterion, evaluate_loader, num_eval_batches)
    print(f'Evaluation accuracy on {num_eval_batches*batch_size} images, {round(top1.avg.item(), 4)}%')

    save_torchscript_module(fp_network, "scripted_weights/per_channel_quant_ptq.pth")

def main(
    weights: str,
    num_eval_batches: int
):
    
    BS = 32
    NUM_CALIBRATION_BATCHES = 32

    torch.manual_seed(191009)
    dataflow = CIFAR10_loader(None, None, BS)
    criterion = nn.CrossEntropyLoss()

    Path("scripted_weights").mkdir(exist_ok=True, parents=True)

    # ========== ========== ========== #
    # Baseline
    # ========== ========== ========== #

    fp_network = get_model(weights)
    baseline(
        fp_network,
        criterion,
        dataflow["test"],
        num_eval_batches,
        BS
    )
    
    # ========== ========== ========== #
    # ptq
    # ========== ========== ========== #

    fp_network = get_model(weights)
    ptq(
        fp_network,
        criterion,
        dataflow["test"],
        NUM_CALIBRATION_BATCHES,
        dataflow["test"],
        num_eval_batches,
        BS
    )

    # ========== ========== ========== #
    # per channel quantization
    # ========== ========== ========== #
    
    fp_network = get_model(weights)
    perchannel_ptq(
        fp_network,
        criterion,
        dataflow["test"],
        NUM_CALIBRATION_BATCHES,
        dataflow["test"],
        num_eval_batches,
        BS
    )

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--weights', type=str, help='')
    parser.add_argument('--num_eval_batches', type=int, default=2000, help='')
    args = parser.parse_args()

    main(weights=args.weights, num_eval_batches=args.num_eval_batches)