from .base_phase import BasePhase
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class ValidationPhase(BasePhase):
    
    def __init__(
        self,
        network: nn.Module,
        criterion: nn.Module,
        dataloader: DataLoader,
        device: torch.device
    ):
        super().__init__(network, criterion, dataloader, device)
        self._process_one_batch = torch.compile(self._process_one_batch)

    def _on_epoch_start(self, *args, **kwargs):
        self.network.eval()

    @torch.inference_mode()
    def _process_one_batch(self, inputs: torch.Tensor, targets: torch.Tensor) -> tuple:
        
        outputs = self.network(inputs)
        loss = self.criterion(outputs, targets)
        
        return outputs, loss
    
    def _process_one_epoch(self, *args, **kwargs):
        return super()._process_one_epoch(*args, **kwargs)