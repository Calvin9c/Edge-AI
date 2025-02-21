import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from tqdm import tqdm

class BasePhase(ABC):
    def __init__(
        self,
        network: nn.Module,
        criterion: nn.Module,
        dataloader: DataLoader,
        device: torch.device
    ):
        self.network = network
        self.criterion = criterion
        self.dataloader = dataloader
        self.device = device

    def _on_epoch_start(self, *args, **kwargs):
        pass

    @abstractmethod
    def _process_one_batch(self, *args, **kwargs):
        raise NotImplementedError("_process_one_batch method must be implemented in subclasses")
    
    def _on_epoch_end(self, *args, **kwargs):
        pass

    def _process_one_epoch(self, *args, **kwargs):
        
        num_samples = 0
        total_loss = 0.0
        correct = 0

        for inputs, targets in tqdm(self.dataloader):
            
            batch_size = inputs.size(0)
            num_samples += batch_size
            
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs, loss = self._process_one_batch(inputs, targets)
            total_loss += loss.item() * batch_size

            _, predicted = outputs.max(1)
            correct += (predicted == targets).sum().item()

        total_loss = round(total_loss / num_samples, 2)
        accuracy = round(100.0 * correct / num_samples, 2)
        
        result = {"total_loss": total_loss,
                  "accuracy": accuracy}

        return result

    def __call__(self, *args, **kwargs):
        self._on_epoch_start(*args, **kwargs)
        res = self._process_one_epoch(*args, **kwargs)
        self._on_epoch_end(*args, **kwargs)
        return res