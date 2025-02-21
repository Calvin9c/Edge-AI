"""reference: https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html#post-training-static-quantization"""
import torch
import torch.nn as nn
import torch.optim as optim
from mobilenetv2 import mobilenetv2_quantization
from torch.utils.data import DataLoader
from dataset import CIFAR10_loader
from optimizer import OptimizerWrapper
from utils import AverageMeter, accuracy, evaluate, save_torchscript_module
from pathlib import Path
import argparse

class TrainPhase:
    def __init__(
        self,
        network: nn.Module,
        criterion: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        optimizer: optim.Optimizer,
        num_train_batches: int,
        scheduler: optim.lr_scheduler._LRScheduler=None
    ):
        self.network = network
        self.criterion = criterion
        self.dataloader = dataloader
        self.device = device
        self.optimizer = optimizer
        self.num_train_batches = num_train_batches
        self.scheduler = scheduler
        self._process_one_batch = torch.compile(self._process_one_batch)

    def _on_epoch_start(self):
        self.network.train()
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        avgloss = AverageMeter('Loss', '1.5f')
        return top1, top5, avgloss  

    def _process_one_batch(self, inputs: torch.Tensor, targets: torch.Tensor)->tuple:
        outputs = self.network(inputs)
        loss = self.criterion(outputs, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return outputs, loss

    def _process_one_epoch(self, top1: AverageMeter, top5: AverageMeter, avgloss: AverageMeter)->dict:
        cnt = 0
        for image, target in self.dataloader:
            cnt += 1
            
            batch_size = image.size(0)
            image, target = image.to(self.device), target.to(self.device)
            
            output, loss = self._process_one_batch(image, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)
            avgloss.update(loss, batch_size)
            
            if cnt >= self.num_train_batches:
                return {"total_loss": round(avgloss.avg.item(), 4),
                        "accuracy@1": round(top1.avg.item(), 4),
                        "accuracy@5": round(top5.avg.item(), 4)}

        return {"total_loss": round(avgloss.avg.item(), 4),
                "accuracy@1": round(top1.avg.item(), 4),
                "accuracy@5": round(top5.avg.item(), 4)}
        
    def _on_epoch_end(self):
        if self.scheduler:
            self.scheduler.step()

    def __call__(self):
        top1, top5, avgloss = self._on_epoch_start()
        result = self._process_one_epoch(top1, top5, avgloss)
        self._on_epoch_end()
        return result

def get_model(weights: str) -> nn.Module:
    fp_network = mobilenetv2_quantization(10, weights)
    fp_network.fuse_model(is_qat=True)
    return fp_network

def qat(
    epochs: int,
    lr: float,
    batch_size: int,
    weights: str,
    num_train_batches: int,
    num_eval_batches: int
):
    
    Path("scripted_weights").mkdir(exist_ok=True, parents=True)

    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.verbose = True

    DEVICE = torch.device('cpu')
    
    torch.manual_seed(191009)
    dataflow = CIFAR10_loader(None, None, batch_size)
    criterion = nn.CrossEntropyLoss()
    
    fp_network = get_model(weights)
    fp_network.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
    # prepare_qat performs the “fake quantization”
    # preparing the model for quantization-aware training
    torch.ao.quantization.prepare_qat(fp_network, inplace=True)
    
    optimizer_wrapper = OptimizerWrapper(
        optimizer_class=optim.Adam,
        device=DEVICE,
        params=list(fp_network.parameters()),
        lr=lr,
        optimizer_args=None,
        weights=None
    )
    
    train_phase = TrainPhase(
        fp_network,
        criterion,
        dataflow["train"],
        DEVICE,
        optimizer_wrapper,
        num_train_batches
    )
    best_acc = -float('inf')
    for epoch in range(epochs):
        
        print(f"\n[Epoch {epoch}]")

        # The `Phase` instance will return a dictionary
        # with key `total_loss`, `accuracy@1` and `accuracy@5`
        train_result = train_phase()
        print(f"Training Result")
        # print(f"    TotalLoss: {train_result['total_loss']}")
        print(f"    Accuracy@1: {train_result['accuracy@1']}% / Accuracy@5: {train_result['accuracy@5']}%")

        if epoch > 3:
            # Freeze quantizer parameters
            train_phase.network.apply(torch.ao.quantization.disable_observer)
        if epoch > 2:
            # Freeze batch norm mean and variance estimates
            train_phase.network.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        
        # Check the accuracy after each epoch
        quantized_model = torch.ao.quantization.convert(train_phase.network.eval(), inplace=False)
        quantized_model.eval()
        top1, top5 = evaluate(quantized_model, criterion, dataflow["test"], neval_batches=num_eval_batches)
        print(f"Evaluation accuracy on {num_eval_batches*batch_size} images")
        print(f"    Accuracy@1: {round(top1.avg.item(), 4)}% / Accuracy@5: {round(top5.avg.item(), 4)}%")
        print("\n---------- ---------- ----------")
        
        if top1.avg.item() > best_acc:
            best_acc = top1.avg.item()
            save_torchscript_module(quantized_model, "scripted_weights/quant_qat.pth")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--epochs', type=int, default=8, help='')
    parser.add_argument('--lr', type=float, default=1e-4, help='')
    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--weights', type=str, help='')
    parser.add_argument('--num_train_batches', type=int, default=24, help='')
    parser.add_argument('--num_eval_batches', type=int, default=2000, help='')
    args = parser.parse_args()
    
    qat(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        weights=args.weights,
        num_train_batches=args.num_train_batches,
        num_eval_batches=args.num_eval_batches
    )