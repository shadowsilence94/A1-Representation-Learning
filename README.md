# A1: Representation Learning - CIFAR-10 Classification

This repository contains PyTorch implementations, training scripts, and comparative evaluations of AlexNet, GoogLeNet, ResNet-18, and Vision Transformers (ViT) on the CIFAR-10 dataset. This report covers architectural changes, exact training commands, quantitative results, and conceptual analyses.

---

## 1. Training and Evaluation Commands

To reproduce the experiments, execute the following commands in your terminal. All scratch models are trained on 32x32 resolutions, while models utilizing ImageNet backbone features are resized to 224x224.

```bash
# 1. Train AlexNet from scratch (including LRN layers)
python3 run.py --model alexnet --dataset cifar10 --epochs 10 --batch_size 64 --train

# 2. Train GoogLeNet from scratch (standard backbone and auxiliary classifiers)
python3 run.py --model googlenet --dataset cifar10 --epochs 25 --batch_size 64 --train

# 3. Train ResNet-18 from scratch (custom 32x32 stem adaptation)
python3 run.py --model resnet18 --dataset cifar10 --epochs 20 --batch_size 64 --train

# 4. Fine-tune pretrained ResNet-18 (2-Stage training: 5 epochs head, 10 epochs all)
python3 run.py --model resnet18_pretrained --dataset cifar10 --epochs 15 --batch_size 64 --train

# 5. Train ViT-Small from scratch (4x4 patch embeddings on 32x32 input)
python3 run.py --model vit_small --dataset cifar10 --epochs 20 --batch_size 64 --train

# 6. Fine-tune pretrained ViT-B/16 (2-Stage training on resized 224x224 input)
python3 run.py --model vit_b16_pretrained --dataset cifar10 --epochs 15 --batch_size 64 --train

# 7. Evaluate saved model weights (example for ResNet-18)
python3 run.py --model resnet18 --dataset cifar10 --test --weights resnet18_cifar10.pth
```

---

## 2. Quantitative Performance Results

The following table summarizes the parameters, accuracy, and training speed of each architecture configuration on CIFAR-10:

| Model | # Params | Test Accuracy | Time/epoch | Architecture Type |
| :--- | :---: | :---: | :---: | :--- |
| AlexNet (from scratch) | 58,322,314 | 63.94% | ~38s | CNN |
| GoogLeNet (from scratch) | 6,285,226 | 97.92% | ~110s | CNN + Inception |
| ResNet-18 (from scratch) | 11,173,962 | 94.66% | ~26s | CNN + Skip connections |
| ResNet-18 (pretrained) | 11,181,642 | 96.50% | ~30s | CNN + Skip connections |
| ViT-Small (from scratch) | 1,205,898 | 79.97% | ~50s | Transformer |
| ViT-B/16 (pretrained, fine-tuned) | 85,806,346 | 98.20% | ~180s | Transformer |

*\*Note: For GoogLeNet, the parameter count excludes the two auxiliary classifiers during evaluation mode (which are only active during training and total 10,635,966 parameters with them active).*

---

## 3. Discussion & Conceptual Analysis

### 3.1. GoogLeNet vs. AlexNet (From Scratch Comparison)
* **Parameter Efficiency**: GoogLeNet has roughly **9x fewer parameters** than AlexNet (6.28M vs. 58.3M). GoogLeNet replaces the memory-intensive fully-connected layers at the end of AlexNet with a global average pooling layer, and uses 1x1 convolutions as "bottleneck" layers to reduce channel dimensions before expensive 3x3 and 5x5 filters.
* **Speed of Training**: Despite having significantly fewer parameters, GoogLeNet is **~3x slower per epoch** than AlexNet (~110s vs. ~38s). The parallel branching structure of Inception modules incurs high GPU kernel launch latencies, creating execution bottlenecks.
* **Accuracy**: GoogLeNet scratch achieves **97.92% test accuracy**, whereas AlexNet scratch reaches only **63.94%**. GoogLeNet's deeper layers and multi-scale receptive fields within Inception blocks extract more robust representations, whereas AlexNet's shallow sequence of large kernels overfits quickly on smaller datasets.

### 3.2. Pretrained Models, Capacity, and Generalization
Pretrained models perform significantly better than models trained from scratch on small target datasets like CIFAR-10. Pretraining on ImageNet allows the network to learn low-level visual features (edges, textures, shapes) that generalize well across domains. In two-stage fine-tuning:
1. **Stage 1 (Freeze backbone)**: Training only the randomly-initialized classifier head prevents backpropagated gradients from corrupting the pretrained representations (catastrophic forgetting).
2. **Stage 2 (Unfreeze all)**: Fine-tuning all layers with a low learning rate allows the pretrained weights to gently adapt to CIFAR-10's specific class structures.

### 3.3. ResNet-18: Why Skip Connections Solve Vanishing Gradients
In deep feedforward networks, backpropagated gradients are repeatedly multiplied by weight matrices. If these weights are small (< 1), the gradient decays exponentially as it travels back to the earlier layers (vanishing gradient problem), stopping them from learning.
ResNet-18 introduces **residual blocks** with skip connections ($H(x) = F(x) + x$). During backpropagation:
$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial H} \cdot \left(\frac{\partial F}{\partial x} + 1\right)$$
The additive term $+1$ acts as a gradient highway. Even if the weight path gradients ($\partial F / \partial x$) vanish, the gradient flows directly back to the earlier layers through the shortcut path unimpeded, allowing networks with hundreds of layers to converge.

### 3.4. Pretrained Transformer vs. Pretrained CNN on CIFAR-10
* Pretrained ViT-B/16 achieves the highest test accuracy (**98.20%**), demonstrating the massive capacity of attention-based architectures when pretrained on large-scale datasets.
* Pretrained ResNet-18 achieves a highly competitive **96.50%** accuracy, but with **8x fewer parameters** (11.2M vs. 85.8M) and training **6x faster** per epoch (~30s vs. ~180s).
* The inductive biases of CNNs (translation invariance and local receptive fields) make them highly parameter-efficient. Transformers, lacking these biases, require massive pretraining data to learn global relationships but scale better at large volumes.

### 3.5. Architectural Trade-offs: CNNs vs. Vision Transformers
1. **Data and Parameter Efficiency**: CNNs have built-in spatial assumptions (locality, translation invariance), making them highly sample-efficient and fast to adapt to small datasets. Transformers make no spatial assumptions, requiring massive pretraining datasets (or heavy regularization) to avoid overfitting.
2. **Computational Complexity**: CNNs scale linearly $O(H \cdot W)$ with spatial image resolution. Self-attention in Transformers scales quadratically $O(N^2)$ with the number of image patches, making high-resolution processing computationally expensive.
3. **Context Modeling**: CNNs have localized receptive fields that grow slowly as layers stack. Transformers capture global dependencies across the entire image from the very first self-attention layer, giving them a significant advantage in modeling long-range contextual relationships.
