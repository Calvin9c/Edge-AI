import torch
import torch.nn as nn
import torch.onnx
import onnxruntime as ort
from torchprofile import profile_macs
from pathlib import Path
import os
import time

def num_params(
    network: nn.Module,
    count_nonzero_only: bool = False
) -> int:
    
    param_count = (
        sum(p.count_nonzero().item() for p in network.parameters()) if count_nonzero_only
        else sum(p.numel() for p in network.parameters())
    )
    
    return param_count

def module_size(model: nn.Module) -> float:
    temp_filename = "temp.p"

    if os.path.exists(temp_filename):
        temp_filename = f"temp_{int(time.time())}.p"

    try:
        torch.save(model.state_dict(), temp_filename)
        model_size = os.path.getsize(temp_filename) / 1e6
    except Exception as e:
        print(f"An error happens when computing the size of the model.: {e}")
        return -1
    finally:
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except Exception as e:
                print(f"Can not remove {temp_filename}: {e}")

    return model_size

def calculate_network_macs(module: nn.Module, input_shape: tuple) -> int:
    """
    Calculates the MACs (Multiply-Accumulate Operations) for a given PyTorch model.
    
    Args:
        module (torch.nn.Module): The PyTorch model to analyze.
        input_shape (tuple): The shape of the input tensor (e.g., (batch_size, channels, height, width)).
        
    Returns:
        int: The total MACs for the model.
    """
    
    if not isinstance(module, nn.Module):
        raise ValueError("Input must be an instance of nn.Module.")
    if not isinstance(input_shape, tuple) or len(input_shape) < 1:
        raise ValueError("Input shape must be a tuple describing the tensor dimensions.")

    dummy_input = torch.randn(*input_shape)

    macs = profile_macs(module, dummy_input)
    return macs

def onnx_model_io_name(
    model: str | Path,
) -> None:
    session = ort.InferenceSession(model)
    iname = session.get_inputs()[0].name
    oname = session.get_outputs()[0].name

    print(f"{str(model)}:\n - Input name: {iname}\n - Output name: {oname}")