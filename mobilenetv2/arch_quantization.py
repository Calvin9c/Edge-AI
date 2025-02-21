import torch
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub
from .arch_modified import MobileNetV2, ConvBNReLU, InvertedResidual

class MobileNetV2Quantization(nn.Module):
    def __init__(self, out_features: int):
        super().__init__()
        self.quant = QuantStub()
        self.net = MobileNetV2(out_features)
        self.dequant = DeQuantStub()
    
    def forward(self, x):
        x = self.quant(x)
        x = self.net(x)
        x = self.dequant(x)
        return x

    # Fuse Conv+BN and Conv+BN+Relu modules prior to quantization
    # This operation does not change the numerics
    def fuse_model(self, is_qat=False):
        fuse_modules = torch.ao.quantization.fuse_modules_qat if is_qat else torch.ao.quantization.fuse_modules
        for m in self.modules():
            if type(m) == ConvBNReLU:
                fuse_modules(m, ['0', '1', '2'], inplace=True)
            if type(m) == InvertedResidual:
                for idx in range(len(m.conv)):
                    if type(m.conv[idx]) == nn.Conv2d:
                        fuse_modules(m.conv, [str(idx), str(idx + 1)], inplace=True)