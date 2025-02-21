# Edge-AI
This repository is inspired by the Edge-AI course at NYCU and the official PyTorch tutorials. It focuses on pruning and quantization of a trained MobileNetV2 model on the CIFAR-10 dataset.

## Table of Contents
 - [MobileNetV2 Training](#mobilenetv2-training)
 - [Pruning](#pruning)
 - [Quantization](#quantization)
 - [References](#references)

## MobileNetV2 Training
We provide a pre-trained MobileNetV2 model, which can be downloaded from [Google Drive](https://drive.google.com/drive/u/1/folders/1rLWhSpRqhF9AXQfWFaOY1fv7n0adlMfP). The following are the training parameters for the provided model:
 - **Optimizer**: Adam
 - **Learning Rate**: 1e-4
 - **Batch Size**: 32
 - **Total Training Epochs**: 24
 - **Best Epoch**: 23
 - **Training Accuracy**: 96.89%
 - **Validation Accuracy**: 80.07%

Alternatively, you can train your own model using the provided training script:
```bash
# Navigate to the Edge-AI directory
python train.py --epochs <EPOCHS> --lr <LR> --batch_size <BS>
```

To evaluate the model, use `evaluate.py`:
```bash
# Navigate to the Edge-AI directory
python evaluate.py --weights <PATH_TO_WEIGHTS_FILE>
```

## Pruning
In this section, we implement a model pruning class based on the Edge-AI course materials with some modifications.
You can prune the model using `prune.py`:
```bash
python prune.py --epochs <RECOVER_EPOCHS> --lr <LR> --batch_size <BS> --weights <PATH_TO_WEIGHTS_FILE> --sparsity_dict <PATH_TO_SPARSITY_DICT>
```

The `sparsity_dict` is a CSV file that specifies the sparsity level (between 0.0 and 1.0) for each layer in the neural network. A sparsity of `1.0` means the layer is not pruned, while `0.0` means all weights in the layer are pruned. You can generate a template using:
```bash
python scripts/generate_sparsity_dict_template.py
```

### Sensitivity Scan
After [training](#mobilenetv2-training), you can use `scripts/sensitivity_scan.py` to analyze the model's accuracy on the test set after pruning different layers. The results will be saved in `scanning_log.csv`.
```bash
python scripts/sensitivity_scan.py --weights <PATH_TO_WEIGHTS_FILE> --accuracy_threshold <THRESHOLD>
```

The script prunes each layer separately, testing three different sparsity levels (default: `[0.5, 0.7, 0.9]`). If the accuracy remains above `accuracy_threshold`, the recommended sparsity value is recorded in `scanning_log.csv`.
To convert `scanning_log.csv` into a `sparsity_dict.csv` for use in [Pruning](#pruning), run:
```bash
python scripts/scanning_log_to_sparsity_dict_template.py --f <PATH_TO_scanning_log> --o <OUTPUT_FILE>
```

By applying our pruning method to the provided model with specific parameters, we can reduce the model size from approximately **8.5MB to 2.31MB**. Note that this calculation is based on the ratio of nonzero parameters, and the `.pth` file size remains 8.5MB since it does not store weights in a compressed format.

**Pruning Parameters:**
 - **Recover Epochs**: 24
 - **Learning Rate**: 1e-4
 - **Batch Size**: 32
 - **Accuracy after Recover Training**: **Training** 94.26% **Validations** 73.61%
 - **Sparse Model Size**: 2.31 MB (27.10% of dense model size)

## Quantization
We implement quantization based on the official PyTorch tutorials for training MobileNetV2 on CIFAR-10.

| Method | Script Model Size | Accuracy on Test Subset | Inference Time |
|:---:|:---:|:---:|:---:|
| Floating Point Model | 9.1 MB | 80.04% | 27 ms |
| PTQ | 2.3 MB | 77.88% | 7 ms |
| Per-Channel PTQ | 2.6 MB | 79.26% | 7 ms |
| QAT | 2.6 MB | 79.69% | 7 ms |

We introduce a `MobileNetV2Quantization` class, adapted from the PyTorch tutorial, which requires conversion of the [pre-trained model](https://drive.google.com/drive/u/1/folders/1rLWhSpRqhF9AXQfWFaOY1fv7n0adlMfP):
```bash
python scripts/to_qmodel_weights.py --weights <PATH_TO_WEIGHTS_FILE>
```
This generates `qmodel.pth`, which will be used for Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT).

### PTQ
```bash
python post_training_quantization.py --weights <PATH_TO_QMODEL> --num_eval_batches <VALUE>
```
The `num_eval_batches` parameter specifies the number of batches used for evaluation (default: 2000).

### QAT
```bash
python quantization_aware_training.py --epochs <EPOCHS> --lr <LR> --batch_size <BS> --weights <PATH_TO_QMODEL> --num_train_batches <VALUE> --num_eval_batches <VALUE>
```
 - `num_train_batches`: Number of batches for training after quantization (default: 24).
 - `num_eval_batches`: Number of batches for evaluation (default: 2000).

### Run Benchmark
After running `post_training_quantization.py` or `quantization_aware_training.py`, the model is saved as a TorchScript model using `torch.jit.save(torch.jit.script(model), "MODEL_SAVE_PATH")`. The script model is stored in the `scripted_weights` directory. You can evaluate the inference time of the model using the `scripts/benchmarks.py` script:
```bash
python scripts/benchmarks.py --model_file <MODEL_SAVE_PATH>
```

## References
1. [Course Material from Edge-AI at NYCU](https://timetable.nycu.edu.tw/?r=main/crsoutline&Acy=113&Sem=2&CrsNo=535525&lang=zh-tw)
2. [PyTorch Quantization Tutorials](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html#post-training-static-quantization)