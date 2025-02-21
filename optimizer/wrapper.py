import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import List, Type

class OptimizerWrapper:
    """
    A generic wrapper for PyTorch optimizers, supporting state loading and device management.

    Attributes:
        optimizer (torch.optim.Optimizer): The wrapped optimizer instance.
    """

    def __init__(
        self,
        optimizer_class: Type[optim.Optimizer],
        device: torch.device,
        params: List[nn.Parameter],
        lr: float,
        optimizer_args: dict = None,
        weights: str | Path = None
    ):
        """
        Initialize the optimizer and optionally load its state.

        Args:
            optimizer_class (Type[optim.Optimizer]): The optimizer class to instantiate.
            params (list): Model parameters to optimize.
            lr (float): Learning rate.
            optimizer_args (dict, optional): Additional arguments for the optimizer. Defaults to None.
            weights (str, optional): Path to the weights file to load optimizer state. Defaults to None.
        """
        # Initialize optimizer with parameters and arguments
        optimizer_args = optimizer_args or {}
        self.optimizer = optimizer_class(params, lr=lr, **optimizer_args)

        # Load optimizer state if weights file is provided
        if weights is not None:
            
            if isinstance(weights, str):
                weights_path = Path(weights)
            else:
                weights_path = weights
                
            if not weights_path.exists():
                raise ValueError(f"Optimizer weights file '{str(weights_path)}' does not exist.")

            checkpoint = torch.load(weights_path, map_location=device)
            self.optimizer.load_state_dict(checkpoint)

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def step(self) -> None:
        self.optimizer.step()