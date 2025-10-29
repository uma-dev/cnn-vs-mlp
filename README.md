## Project Overview

Compare the performance, efficiency, and characteristics of Convolutional Neural Networks (CNNs) versus Multi-Layer Perceptrons (MLPs) on image classification tasks.

## Objectives

1. Implement both CNN and MLP architectures from using PyTorch.
2. Compare models across three key metrics: **precision/accuracy**, **parameter count**, and **training time**
3. Understand why CNNs are preferred for image data despite MLPs being universal function approximators

## Dataset

**Primary**: CIFAR-10 (32×32 color images, 10 classes)

- Small enough for reasonable training times
- Complex enough to show meaningful differences
- Standard benchmark dataset

## Proposed Architectures

### MLP Architecture

- **Input layer**: Flatten 32×32×3 = 3,072 input features
- **Hidden layers**: 2-3 fully connected layers (e.g., 512 → 256 → 128 neurons)
- **Output layer**: 10 classes with softmax
- **Activation**: ReLU
- **Regularization**: Dropout (0.2-0.5) (optional)

### CNN Architecture

- **Convolutional blocks**: 3-4 blocks of Conv → ReLU → MaxPool
  - Block 1: 32 filters, 3×3 kernel
  - Block 2: 64 filters, 3×3 kernel
  - Block 3: 128 filters, 3×3 kernel
- **Fully connected layers**: 1-2 dense layers (128 → 10)
- **Regularization**: Dropout, Batch Normalization

## Experimental Design

### 1. Fair Comparison Setup

- Same optimizer (Adam with learning rate 0.001)
- Same batch size (64 or 128)
- Same number of epochs (50)
- Same data augmentation (random flips, crops)
- Same train/validation/test split (80/10/10)

### 2. Metrics to Track

**Precision/Accuracy**:

- Test accuracy
- Per-class precision, recall, F1-score
- Confusion matrices
- Learning curves (training vs validation accuracy over epochs)

**Parameter Count**:

- Total trainable parameters

**Training Time**:

- Time per epoch
- Total training time
- Inference time (predictions per second)
- GPU vs CPU performance (optional)

**Parameter Matching Experiment**:

- Design an MLP and CNN with approximately the same number of parameters
- Compare their performance to isolate architectural benefits

## Deliverables

1. **Code Repository**:
   - Well-documented Python notebooks/scripts
   - Reproducible experiments with random seeds
   - Requirements.txt for dependencies

2. **Technical Report** (8-10 pages):
   - Introduction and motivation
   - Methodology and architecture details
   - Results with tables and visualizations
   - Discussion of trade-offs
   - Conclusion and insights
