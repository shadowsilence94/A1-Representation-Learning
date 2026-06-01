import argparse
import time
import os
import copy
from typing import Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import datasets, transforms

# Import custom models
from alexnet import AlexNet
from googlenet import GoogLeNet
from resnet18 import ResNet18
from vit import ViTSmall


def get_device():
    """
    Selects the best available device.
    Prioritizes MPS (Metal Performance Shaders) for Apple Silicon,
    then CUDA, and falls back to CPU.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


def get_dataloaders(model_name: str, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns CIFAR-10 data loaders (train, val, test) configured with the correct input size
    for the selected model.
    """
    # 224x224 models (ImageNet input size)
    if model_name in ['alexnet', 'googlenet', 'resnet18_pretrained', 'vit_b16_pretrained']:
        img_size = 224
        resize_size = 256
    # 32x32 models (CIFAR-10 direct input size)
    else:
        img_size = 32
        resize_size = 36

    transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    data_root = "/tmp/cifar10_data"
    train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    # Split train set (50k images) into 40k train and 10k validation
    train_subset, val_subset = random_split(train_dataset, [40000, 10000])

    test_subset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def initialize_model(model_name: str, num_classes: int = 10, use_lrn: bool = True):
    """
    Instantiates the selected model.
    """
    if model_name == 'alexnet':
        model = AlexNet(num_classes=num_classes, use_lrn=use_lrn)
    elif model_name == 'googlenet':
        model = GoogLeNet(num_classes=num_classes, aux_classifiers=True)
    elif model_name == 'resnet18':
        model = ResNet18(n_classes=num_classes)
    elif model_name == 'vit_small':
        model = ViTSmall(n_classes=num_classes)
    elif model_name == 'resnet18_pretrained':
        from torchvision.models import resnet18, ResNet18_Weights
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'vit_b16_pretrained':
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        model.heads = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model


def get_optimizer_and_scheduler(model_name: str, model: nn.Module):
    """
    Returns the appropriate optimizer and learning rate scheduler for the model.
    """
    # Fine-tuning pretrained models
    if model_name in ['resnet18_pretrained', 'vit_b16_pretrained']:
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = None
    elif model_name == 'resnet18':
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepScheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    elif model_name == 'vit_small':
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = None
    elif model_name == 'googlenet':
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = None
    else:  # alexnet
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = None

    return optimizer, scheduler


def handle_fine_tuning_stages(model_name: str, model: nn.Module, epoch: int):
    """
    Freezes the backbone for first 5 epochs (Stage 1) and unfreezes it afterwards (Stage 2)
    for pretrained models.
    """
    if model_name == 'resnet18_pretrained':
        if epoch < 5:
            # Stage 1: Freeze backbone, train only fc
            for name, param in model.named_parameters():
                if 'fc' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            # Stage 2: Unfreeze all
            for param in model.parameters():
                param.requires_grad = True

    elif model_name == 'vit_b16_pretrained':
        if epoch < 5:
            # Stage 1: Freeze backbone, train only heads
            for name, param in model.named_parameters():
                if 'heads' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            # Stage 2: Unfreeze all
            for param in model.parameters():
                param.requires_grad = True


def train_model(model_name: str, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                criterion: nn.Module, optimizer: optim.Optimizer, scheduler: Optional[optim.lr_scheduler.LRScheduler],
                num_epochs: int, device: torch.device, weights_path: str):
    """
    Core training and validation loop.
    """
    since = time.time()
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        # Handle fine-tuning stages for pretrained models
        handle_fine_tuning_stages(model_name, model, epoch)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0
            epoch_start_time = time.time()

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # GoogLeNet outputs auxiliary classifiers during training
                    if model_name == 'googlenet' and phase == 'train':
                        outputs, aux2, aux1 = model(inputs)
                        loss_main = criterion(outputs, labels)
                        loss_aux2 = criterion(aux2, labels)
                        loss_aux1 = criterion(aux1, labels)
                        loss = loss_main + 0.3 * loss_aux2 + 0.3 * loss_aux1
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train' and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.item() / len(dataloader.dataset)
            epoch_time = time.time() - epoch_start_time

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Time: {epoch_time:.1f}s")

            # Deep copy the model weights if I have a new best validation accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), weights_path)
                print(f"Saved new best model checkpoint to {weights_path}")

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model


def evaluate_model(model: nn.Module, test_loader: DataLoader, criterion: nn.Module, device: torch.device):
    """
    Evaluates the model on the test dataset.
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    print("\nRunning evaluation on test set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    test_loss = running_loss / len(test_loader.dataset)
    test_acc = running_corrects.item() / len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")
    return test_acc


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 Training and Evaluation Runner")
    parser.add_argument('--model', type=str, required=True,
                        choices=['alexnet', 'googlenet', 'resnet18', 'vit_small', 'resnet18_pretrained', 'vit_b16_pretrained'],
                        help="Select the model architecture")
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10'],
                        help="Select the dataset")
    parser.add_argument('--epochs', type=int, default=10,
                        help="Number of epochs to train")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Batch size for training")
    parser.add_argument('--train', action='store_true',
                        help="Train the model")
    parser.add_argument('--test', '--teset', dest='test', action='store_true',
                        help="Evaluate the model on test set")
    parser.add_argument('--weights', type=str, default=None,
                        help="Path to weight checkpoint for evaluation or saving")
    parser.add_argument('--no_lrn', action='store_true',
                        help="Disable LRN for AlexNet configuration")

    args = parser.parse_args()
    device = get_device()

    # Load dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(args.model, args.batch_size)

    # Initialize model
    use_lrn = not args.no_lrn
    model = initialize_model(args.model, num_classes=10, use_lrn=use_lrn)
    model = model.to(device)

    # Set up checkpoint weights name
    weights_path = args.weights if args.weights else f"{args.model}_cifar10.pth"

    criterion = nn.CrossEntropyLoss()

    if args.train:
        print(f"\nStarting training for {args.model} on CIFAR-10...")
        optimizer, scheduler = get_optimizer_and_scheduler(args.model, model)
        model = train_model(args.model, model, train_loader, val_loader, criterion, optimizer, scheduler,
                            args.epochs, device, weights_path)

    if args.test:
        if not args.train:
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Checkpoint weights file not found: {weights_path}")
            print(f"Loading weights from {weights_path}...")
            # Map storage to current device dynamically
            model.load_state_dict(torch.load(weights_path, map_location=device))
        evaluate_model(model, test_loader, criterion, device)


if __name__ == '__main__':
    main()
