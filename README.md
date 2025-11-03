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

- **Input layer**: Flatten 32×32×3 = 3,072 input features -> images of 32px by 32px of three color channels (R,G,B)
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

# CNN vs MLP — CIFAR-10 Comparison

## Key Findings

| Model   | Params        | Test Accuracy | Total Train Time          | Avg Epoch Time |
| ------- | ------------- | ------------- | ------------------------- | -------------- |
| **CNN** | **357,258**   | **81.89%**    | **5724.77 s (~95.4 min)** | ~114–121 s     |
| **MLP** | **1,738,890** | **46.22%**    | **4105.73 s (~68.4 min)** | ~80–86 s       |

- CNN reaches **~82% accuracy** with **5× fewer parameters**.
- MLP lags at **~46%**, even with **5× more parameters**.
- CNN converges best around **epoch ~47**; MLP plateaus ~epoch 16–25.

---

## Parameter Count

- CNN: **357,258 trainable parameters**
- MLP: **1,738,890 trainable parameters**

CNN is dramatically more parameter-efficient.

---

## Training Time

### Time per epoch

- CNN: ~114–121 s
- MLP: ~80–86 s

MLP is faster per epoch, but delivers weak accuracy.

### Total training time

- CNN: **5724.77 s**
- MLP: **4105.73 s**

### Inference time

> **Not measured — see Future Work.**

### CPU/GPU comparison

> **Not measured — see Future Work.**

---

## Per-Class Performance (F1)

### CNN — best to worst

- automobile: **0.913**
- ship: **0.887**
- truck: **0.884**
- deer: **0.810**
- airplane: **0.826**
- horse: **0.862**
- frog: **0.860**
- dog: **0.749**
- bird: **0.737**
- cat: **0.651**

### MLP — best to worst

- airplane: **0.560**
- ship: **0.598**
- truck: **0.526**
- horse: **0.537**
- frog: **0.501**
- cat: **0.379**
- bird: **0.279**
- dog: **0.277**

CNN dominates every class. MLP mostly guesses.

---

## Training Dynamics (Summary)

**CNN**

- Slow, consistent improvement
- Best val ~0.8166 near epoch 47
- Late epochs still give small gains

**MLP**

- Early improvement, then plateau
- Best val ~0.50
- Little meaningful progress after epoch 20

---

## Parameter Matching Experiment

> **Not completed — see Future Work.**

Intent:

- Build an MLP and CNN with **similar parameter counts (~0.36M)**
- Compare accuracy + timing to isolate architectural benefit

---

## Visual Comparisons (recommended)

1. Learning curves (val_acc)
   - CNN → ~0.81
   - MLP → ~0.50

2. Per-class F1 bar chart
   - CNN wins everywhere

3. Confusion matrices
   - CNN structured
   - MLP chaotic

4. Param vs accuracy scatter
   - CNN: **0.36M → 81.9%**
   - MLP: **1.74M → 46.2%**

---

## Discussion of Trade-offs

### Parameter efficiency

CNN: **0.36M params → 81.9%**  
MLP: **1.74M params → 46.2%**  
MLP wastes ~1.3M extra parameters for much worse accuracy.

### Accuracy

CNN is dramatically stronger, especially on animal classes where MLP crumbles.

### Training dynamics

CNN learns meaningful features across depth; MLP flatlines early.

### Time

MLP is faster per epoch, but that “advantage” is functionally pointless since accuracy is poor.

---

## Why CNNs Beat MLPs on Images

- **Images have spatial structure** — nearby pixels relate.
- **MLPs ignore structure**, treating all pixels as unrelated.
- **CNNs share weights** across the image, reducing parameter count.
- **Local receptive fields** capture edges → textures → objects.
- **Translation invariance**: a feature is recognized anywhere in the image.
- MLPs must learn these things manually. Spoiler: they don’t.

**Analogy:**  
MLP = trying to understand a city by memorizing every intersection individually.  
CNN = realizing city blocks repeat, so once you know one, you know many.

Universal approximation is a math trophy, not a practical training strategy.

---

## Conclusion & Insights

- CNNs deliver **massively superior accuracy** with **far fewer parameters**.
- MLP simply lacks the inductive biases needed for vision.
- Even with 5× more parameters, MLP fails to compete.
- Outcome is straightforward:  
  **CNNs are the correct tool for image tasks.**

---

## Future Work

- Measure **inference time** (images/sec)
- Compare **CPU vs GPU performance**
- Complete **parameter-matching experiment**
  - Build ~0.36M-param MLP
  - Compare to ~0.36M-param CNN
- Add aggregated final tables + plots for report

---

## Why MLP is Faster per Epoch than CNN?

| Aspect                   | MLP (Fully-Connected)                                    | CNN (Convolutional)                                              | Why This Makes MLP Faster                                                        |
| ------------------------ | -------------------------------------------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| Computation type         | Dense matrix multiplications (GEMM)                      | Convolutions + pooling + BN                                      | GEMM kernels are heavily optimized and run ridiculously fast on modern hardware. |
| Data access pattern      | Simple, contiguous memory loads                          | Sliding windows over spatial grid                                | Convolutions require more complex memory access, increasing latency.             |
| Operations per parameter | Straightforward multiply-accumulate per weight           | Reuse filters spatially, but each conv op touches many positions | Convs need more work per weight; MLP does simpler math per param.                |
| Feature shape changes    | Constant vector dimension                                | Spatial reshaping (H×W), pooling, channel expansion              | CNN constantly reshapes tensors; MLP just stays flat → less overhead.            |
| Kernel re-use            | None — one-shot matmul                                   | Repeated kernel application across spatial grid                  | Convs generate way more intermediate ops → slower.                               |
| Framework optimization   | Dense ops are the most mature & optimized in BLAS/cuBLAS | Conv ops are fast but still more involved                        | MLP benefits more from decades of matmul optimization.                           |
| Parallelism              | Uniform workload → easy to parallelize                   | Uneven spatial ops → more scheduling overhead                    | Uniform GEMM makes hardware utilization simpler & more efficient.                |
| Forward complexity       | O(N × D)                                                 | O(N × k² × C × H × W)                                            | Conv includes kernel area + spatial dims → more computation.                     |
| Backward pass            | Simple gradient of GEMM                                  | Gradient wrt filters & spatial positions                         | Conv backprop is more expensive than dense backprop.                             |

**Short version:**  
MLP = big matrix multiply → very fast  
CNN = many sliding-window convolutions → more compute + more memory juggling → slower
