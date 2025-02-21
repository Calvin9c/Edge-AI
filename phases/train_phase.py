from .base_phase import BasePhase
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

class TrainPhase(BasePhase):
    
    def __init__(
        self,
        network: nn.Module,
        criterion: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        optimizer: optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
    ):
        super().__init__(network, criterion, dataloader, device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self._process_one_batch = torch.compile(self._process_one_batch)

    def _on_epoch_start(self, *args, **kwargs):
        self.network.train()
  
    def _process_one_batch(self, inputs: torch.Tensor, targets: torch.Tensor) -> tuple:
        
        outputs = self.network(inputs)
        loss = self.criterion(outputs, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return outputs, loss

    def _on_epoch_end(self, *args, **kwargs):
        if self.scheduler:
            self.scheduler.step()
    
    def _process_one_epoch(self, *args, **kwargs):
        return super()._process_one_epoch(*args, **kwargs)
        