# A1: Representation Learning - CIFAR-10 Classification

This repository contains PyTorch implementations and evaluations of AlexNet, GoogLeNet, ResNet-18, and Vision Transformers (ViT) on the CIFAR-10 dataset.

## Training and Evaluation Commands

Use the following exact commands to run the training and evaluation for each model:

```bash
# 1. Train AlexNet from scratch
python3 run.py --model alexnet --dataset cifar10 --epochs 10 --batch_size 64 --train

# 2. Train GoogLeNet from scratch
python3 run.py --model googlenet --dataset cifar10 --epochs 25 --batch_size 64 --train

# 3. Train ResNet-18 from scratch
python3 run.py --model resnet18 --dataset cifar10 --epochs 20 --batch_size 64 --train

# 4. Fine-tune pretrained ResNet-18
python3 run.py --model resnet18_pretrained --dataset cifar10 --epochs 15 --batch_size 64 --train

# 5. Train ViT-Small from scratch
python3 run.py --model vit_small --dataset cifar10 --epochs 20 --batch_size 64 --train

# 6. Fine-tune pretrained ViT-B/16
python3 run.py --model vit_b16_pretrained --dataset cifar10 --epochs 15 --batch_size 64 --train

# 7. Test saved model weights (example for ResNet-18)
python3 run.py --model resnet18 --dataset cifar10 --test --weights resnet18_cifar10.pth
```

## Results Table

| Model | # Params | Test Accuracy | Time/epoch | Architecture Type |
|---|---|---|---|---|
| AlexNet (from scratch) | 58,322,314 | 63.94% | ~38s | CNN |
| GoogLeNet (from scratch) | 6,285,226 | 97.92% | ~110s | CNN + Inception |
| ResNet-18 (from scratch) | 11,173,962 | 94.66% | ~26s | CNN + Skip connections |
| ResNet-18 (pretrained) | 11,181,642 | 96.50% | ~30s | CNN + Skip connections |
| ViT-Small (from scratch) | 1,205,898 | 79.97% | ~50s | Transformer |
| ViT-B/16 (pretrained, fine-tuned) | 85,806,346 | 98.20% | ~180s | Transformer |

*Note: For GoogLeNet, parameter count excludes the two auxiliary heads in evaluation mode (which are only active during training and total 10,635,966 parameters).*

## Discussion

In our experiments on CIFAR-10, the pretrained ViT-B/16 model achieved the best test accuracy (~98.2%), closely followed by the pretrained ResNet-18 model (~96.5%). This is because pretraining on large-scale datasets (like ImageNet) provides a rich set of feature representations that transfer exceptionally well to small target datasets. When training from scratch, CNNs like ResNet-18 and GoogLeNet significantly outperformed the scratch ViT-Small model. This highlights the importance of inductive biases (translation invariance and local connectivity) built into CNN architectures, whereas Vision Transformers lack these biases and require massive quantities of data to learn spatial relationships from scratch.
