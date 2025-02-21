import torch
import torch.nn as nn
from pathlib import Path

def save_state_dict(
    state_dict: dict,
    save_path: str | Path
) -> None:

    if isinstance(save_path, str):
        save_path = Path(save_path)
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, save_path)

def save_torchscript_module(m: nn.Module, f: str):
    torch.jit.save(torch.jit.script(m), f)

def save_onnx(
    model: nn.Module,
    dummy_input: torch.Tensor,
    save_path: str | Path,
    verbose: bool = False,
    dynamic_axes: dict = None,
    input_names: list[str] = None,
    output_names: list[str] = None
) -> None:
    
    if isinstance(save_path, str):
        save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            verbose=verbose,
            dynamic_axes=dynamic_axes,
            input_names=input_names,
            output_names=output_names
        )
        print(f"Model successfully exported to {str(save_path)}")
    except Exception as e:
        print(f"Failed to export model: {e}")