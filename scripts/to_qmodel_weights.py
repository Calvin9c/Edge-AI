import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, repo_dir)

import argparse
import torch
from mobilenetv2 import mobilenetv2, mobilenetv2_quantization

def main(weights: str):
    src = mobilenetv2(10, weights)
    target = mobilenetv2_quantization(10)
    target.net = src
    torch.save(target.state_dict(), "qmodel.pth") 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--weights', type=str, help='')
    args = parser.parse_args()
    main(args.weights)