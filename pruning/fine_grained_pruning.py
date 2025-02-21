import torch
import torch.nn as nn
import numpy as np

def magnitude_base_pruning(tensor: torch.Tensor, sparsity: float) -> torch.Tensor:
    """
    Calculate a binary mask for pruning based on the given sparsity and importance metric.

    :param tensor: torch.Tensor, weight tensor of a layer.
    :param sparsity: float, pruning sparsity (0 to 1).
    :return: torch.Tensor, binary mask with the same shape as `tensor`.
    """
    sparsity = np.clip(sparsity, 0.0, 1.0)

    if sparsity == 1.0:
        return torch.zeros_like(tensor)
    elif sparsity == 0.0:
        return torch.ones_like(tensor)

    num_elements = tensor.numel()

    # Compute importance based on the selected metric
    importance = tensor.abs()

    # Determine threshold for pruning
    threshold, _ = torch.kthvalue(importance.view(-1), round(sparsity * num_elements))

    # Generate binary mask
    return (importance > threshold).to(tensor.dtype)

@torch.no_grad()
def fine_grained_prune(model: nn.Module, sparsity_dict: dict) -> dict:
    masks = dict()
    for name, param in model.named_parameters():
        # we only prune `conv`` and `fc` weights
        if 1 < param.dim() and name in sparsity_dict:
            mask = magnitude_base_pruning(param, sparsity_dict[name])
            masks[name] = mask
    return masks