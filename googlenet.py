import torch
import torch.nn as nn
from typing import Tuple, Optional, Union

class Inception(nn.Module):
    """
    Inception block with parallel 1x1, 3x3, 5x5 convolutions and 3x3 max pooling.
    """
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True)
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True)
        )

        # 1x1 conv -> 5x5 conv branch (implemented as two 3x3 convolutions for computational efficiency)
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True)
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1)


class InceptionAux(nn.Module):
    """
    Auxiliary classifier to inject gradients at intermediate layers of GoogLeNet.
    """
    def __init__(self, in_planes: int, num_classes: int):
        super(InceptionAux, self).__init__()
        self.avgpool = nn.AvgPool2d(5, stride=3)
        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Dropout(0.7)
        )
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class GoogLeNet(nn.Module):
    """
    Standard GoogLeNet (Inception v1) for 224x224 inputs,
    complete with auxiliary classifiers for training.
    """
    def __init__(self, num_classes: int = 10, aux_classifiers: bool = True):
        super(GoogLeNet, self).__init__()
        self.aux_classifiers = aux_classifiers

        # Standard Backbone
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(True)
        )
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)

        # Inception 3a, 3b
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)

        # Inception 4a, 4b, 4c, 4d, 4e
        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)

        # Inception 5a, 5b
        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.linear = nn.Linear(1024, num_classes)

        # Auxiliary classifiers
        if aux_classifiers:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        else:
            self.aux1 = None
            self.aux2 = None

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        out = self.conv1(x)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.maxpool2(out)

        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool3(out)

        # Inception 4a
        out = self.a4(out)
        aux1_out: Optional[torch.Tensor] = None
        if self.training and self.aux_classifiers and self.aux1 is not None:
            aux1_out = self.aux1(out)

        out = self.b4(out)
        out = self.c4(out)

        # Inception 4d
        out = self.d4(out)
        aux2_out: Optional[torch.Tensor] = None
        if self.training and self.aux_classifiers and self.aux2 is not None:
            aux2_out = self.aux2(out)

        out = self.e4(out)
        out = self.maxpool4(out)

        out = self.a5(out)
        out = self.b5(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        final_out = self.linear(out)

        if self.training and self.aux_classifiers and aux1_out is not None and aux2_out is not None:
            return final_out, aux2_out, aux1_out
        return final_out
