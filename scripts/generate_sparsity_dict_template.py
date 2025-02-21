import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, repo_dir)

import torch.nn as nn
import pandas as pd
from mobilenetv2 import mobilenetv2

def generate_sparsity_dict_template(model: nn.Module):
    """
    generate a csv file that describe the sparsity_dict of the input model for pruning.

    Args:
        model (nn.Module): model to analysis
    """
    
    df = pd.DataFrame([{'name': name, 'sparsity': 0.0}
                       for name, param in model.named_parameters() if param.dim() > 1])
    df.to_csv('sparsity_dict_template.csv', index=False)

if __name__ == "__main__":
    generate_sparsity_dict_template(mobilenetv2(out_features=10))