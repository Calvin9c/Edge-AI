from .arch_modified import MobileNetV2
from .arch_quantization import MobileNetV2Quantization
import torch
import torch.nn as nn
from pathlib import Path

def mobilenetv2(
    out_features: int,
    weights: str = None,
    device: torch.device = torch.device('cpu')
) -> nn.Module:
    """
    Creates and initializes a MobileNetV2 model, optionally loads pre-trained weights,
    and compiles the model for optimization if specified.

    Args:
        out_features (int): The number of output features (e.g., number of classes for classification).
        weights (str, optional): Path to the pre-trained weights file. If provided, the model's 
            state_dict will be loaded from this file. Defaults to None.
        device (torch.device, optional): The device to which the model should be moved (e.g., CPU or GPU). 
            Defaults to torch.device('cpu').

    Returns:
        nn.Module: The initialized and optionally compiled MobileNetV2 model.

    Raises:
        ValueError: If the specified weights file does not exist.
    """
    # Create a MobileNetV2 instance with the specified output features
    net = MobileNetV2(out_features)
    
    # Load pre-trained weights if a valid path is provided
    if weights is not None:
        weights_path = Path(weights)
        
        # Check if the weights file exists
        if not weights_path.exists():
            raise ValueError(f"The weights file at {weights_path} does not exist.")
        
        # Load the state_dict from the weights file
        ckpt = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=True)
        net.load_state_dict(ckpt)

    # Move the model to the specified device
    net.to(device)

    return net

def mobilenetv2_quantization(
    out_features: int,
    weights: str=None
) -> MobileNetV2Quantization:
    net = MobileNetV2Quantization(out_features)
    
    if weights:
        weight_path = Path(weights)
        if not weight_path.exists():
            raise ValueError(f"The weights file at {weights} does not exist.")
        
        ckpt = torch.load(weight_path, map_location=torch.device("cpu"), weights_only=True)
        net.load_state_dict(ckpt)

    net.to('cpu')
    return net