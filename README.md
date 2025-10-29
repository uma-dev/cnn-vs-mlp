## Project Overview

Compare the performance, efficiency, and characteristics of Convolutional Neural Networks (CNNs) versus Multi-Layer Perceptrons (MLPs) on **image classification** tasks.

## Objectives

- [x] 1. Implement both CNN and MLP architectures from using PyTorch.
- [x] 2. Compare models across three key metrics: **precision/accuracy**, **parameter count**, and **training time**
- [x] 3. Understand why CNNs are preferred for image data despite MLPs being universal function approximators

## Dataset

**Primary**: CIFAR-10 (60,000 32×32 colour images, 10 classes, balanced) -> downloaded via `torchvision.datasets.CIFAR10`

![CIFAR Dataset](https://docs.pytorch.org/tutorials/_images/cifar10.png "CIFAR-10")

- Small enough for reasonable training times
- Complex enough to show meaningful differences
- Standard benchmark dataset

## Proposed Architectures

### MLP

- **Input layer**: Flatten 32×32×3 = 3,072 input features -> images of 32px by 32px of three  color channels (R,G,B)
- **Hidden layers**: 2-3 fully connected layers (e.g., 512 → 256 → 128 neurons)
- **Output layer**: 10 classes with softmax
- **Activation**: ReLU
- **Regularization**: Dropout (0.2-0.5) (optional)

### CNN

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
<table>
  <thead>
    <tr>
      <th>CNN</th>
      <th>MLP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">
        <img width="600" height="500" alt="CNN confusion matrix" src="https://github.com/user-attachments/assets/89872e62-673a-4684-be0c-b662ab2485fa" />
      </td>
      <td align="center">
        <img width="600" height="500" alt="MLP confusion matrix" src="https://github.com/user-attachments/assets/e673fe28-930f-40ca-ac8b-3b58bb9e8110" />
      </td>
    </tr>
  </tbody>
</table>
 
- Learning curves (training vs validation accuracy over epochs)
<table>
  <thead>
    <tr>
      <th>CNN</th>
      <th>MLP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">
        <img width="640" height="480" alt="CNN learning_curves" src="https://github.com/user-attachments/assets/31ba3a2a-9f10-46fc-a25e-eb8dcda6d586" />
      </td>
      <td align="center">
        <img width="640" height="480" alt="MLP learning_curves" src="https://github.com/user-attachments/assets/892ce7b4-0992-4938-9cc8-609077aa71a3" />
      </td>
    </tr>
  </tbody>
</table>

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
