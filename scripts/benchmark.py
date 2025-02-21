import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, repo_dir)

import torch
import time
import argparse
from dataset import CIFAR10_loader

def run_benchmark(model_file, img_loader):
    elapsed = 0
    model = torch.jit.load(model_file)
    model.eval()
    num_batches = 5
    # Run the scripted model on a few batches of images
    for i, (images, target) in enumerate(img_loader):
        if i < num_batches:
            start = time.time()
            output = model(images)
            end = time.time()
            elapsed = elapsed + (end-start)
        else:
            break
    num_images = images.size()[0] * num_batches

    print('Elapsed time: %3.0f ms' % (elapsed/num_images*1000))
    return elapsed

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_file', type=str, help='')
    args = parser.parse_args()
    
    loader = CIFAR10_loader(None, None, 32, 2)
    
    run_benchmark(args.model_file, loader["test"])