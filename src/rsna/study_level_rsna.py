import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNeXtBottleneck2D(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        cardinality=32,
        base_width=4,
        downsample=None,
    ):
        super(ResNeXtBottleneck2D, self).__init__()
        D = int(math.floor(out_channels * (base_width / 64.0)))
        C = cardinality

        self.conv1 = nn.Conv2d(in_channels, D * C, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(D * C)
        self.conv2 = nn.Conv2d(
            D * C, D * C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False
        )
        self.bn2 = nn.BatchNorm2d(D * C)
        self.conv3 = nn.Conv2d(
            D * C, out_channels * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.LeakyReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXtBottleneck3D(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        cardinality=32,
        base_width=4,
        downsample=None,
    ):
        super(ResNeXtBottleneck3D, self).__init__()
        D = int(math.floor(out_channels * (base_width / 64.0)))
        C = cardinality

        self.conv1 = nn.Conv3d(in_channels, D * C, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(D * C)
        self.conv2 = nn.Conv3d(
            D * C, D * C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False
        )
        self.bn2 = nn.BatchNorm3d(D * C)
        self.conv3 = nn.Conv3d(
            D * C, out_channels * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm3d(out_channels * self.expansion)
        self.relu = nn.LeakyReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt2D(nn.Module):
    def __init__(
        self, block, layers, cardinality=32, num_classes=1000, input_channels=8
    ):
        super(ResNeXt2D, self).__init__()
        self.in_channels = 64
        self.cardinality = cardinality
        self.base_width = 4

        # Adjusted first convolution layer for input_channels
        self.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )  # 7x7x8 filters
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.layer1 = self._make_layer(block, 64, layers[0])  # Conv2
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # Conv3
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # Conv4
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # Conv5

        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.in_channels,
                out_channels,
                stride,
                self.cardinality,
                self.base_width,
                downsample,
            )
        )
        self.in_channels = out_channels * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.in_channels,
                    out_channels,
                    cardinality=self.cardinality,
                    base_width=self.base_width,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        # x is axial input
        x = self.conv1(x)  # 7x7x8 filters
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)  # Conv2
        x = self.layer2(x)  # Conv3
        x = self.layer3(x)  # Conv4
        x = self.layer4(x)  # Conv5

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


class ResNeXt3D(nn.Module):
    def __init__(
        self, block, layers, cardinality=32, num_classes=1000, input_channels=1
    ):
        super(ResNeXt3D, self).__init__()
        self.in_channels = 64
        self.cardinality = cardinality
        self.base_width = 4

        # First convolution layer for sagittal input
        self.conv1 = nn.Conv3d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )  # 7x7x7 filters
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.layer1 = self._make_layer(block, 64, layers[0])  # Conv2
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # Conv3
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # Conv4
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # Conv5

        self.avgpool = nn.AdaptiveAvgPool3d(1)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(out_channels * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.in_channels,
                out_channels,
                stride,
                self.cardinality,
                self.base_width,
                downsample,
            )
        )
        self.in_channels = out_channels * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.in_channels,
                    out_channels,
                    cardinality=self.cardinality,
                    base_width=self.base_width,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        # x is sagittal input
        x = self.conv1(x)  # 7x7x7 filters
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)  # Conv2
        x = self.layer2(x)  # Conv3
        x = self.layer3(x)  # Conv4
        x = self.layer4(x)  # Conv5

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x
