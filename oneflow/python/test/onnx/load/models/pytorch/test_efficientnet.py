"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import torchvision

from oneflow.python.test.onnx.load.util import load_pytorch_module_and_check

import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/kuangliu/pytorch-cifar/blob/master/models/efficientnet.py

def swish(x):
    return x * x.sigmoid()

def drop_connect(x, drop_ratio):
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x


class SE(nn.Module):
    """Squeeze-and-Excitation block with Swish."""

    def __init__(self, in_channels, se_channels):
        super(SE, self).__init__()
        self.se1 = nn.Conv2d(in_channels, se_channels, kernel_size=1, bias=True)
        self.se2 = nn.Conv2d(se_channels, in_channels, kernel_size=1, bias=True)

    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, (1, 1))
        out = swish(self.se1(out))
        out = self.se2(out).sigmoid()
        out = x * out
        return out


class Block(nn.Module):
    """expansion + depthwise + pointwise + squeeze-excitation"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        expand_ratio=1,
        se_ratio=0.0,
        drop_rate=0.0,
    ):
        super(Block, self).__init__()
        self.stride = stride
        self.drop_rate = drop_rate
        self.expand_ratio = expand_ratio

        # Expansion
        channels = expand_ratio * in_channels
        self.conv1 = nn.Conv2d(
            in_channels, channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)

        # Depthwise conv
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(1 if kernel_size == 3 else 2),
            groups=channels,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels)

        # SE layers
        se_channels = int(in_channels * se_ratio)
        self.se = SE(channels, se_channels)

        # Output
        self.conv3 = nn.Conv2d(
            channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Skip connection if in and out shapes are the same (MV-V2 style)
        self.has_skip = (stride == 1) and (in_channels == out_channels)

    def forward(self, x):
        out = x if self.expand_ratio == 1 else swish(self.bn1(self.conv1(x)))
        out = swish(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))
        if self.has_skip:
            if self.training and self.drop_rate > 0:
                out = drop_connect(out, self.drop_rate)
            out = out + x
        return out


class EfficientNet(nn.Module):
    def __init__(self, cfg, num_classes=10):
        super(EfficientNet, self).__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_channels=32)
        self.linear = nn.Linear(cfg["out_channels"][-1], num_classes)

    def _make_layers(self, in_channels):
        layers = []
        cfg = [
            self.cfg[k]
            for k in [
                "expansion",
                "out_channels",
                "num_blocks",
                "kernel_size",
                "stride",
            ]
        ]
        b = 0
        blocks = sum(self.cfg["num_blocks"])
        for expansion, out_channels, num_blocks, kernel_size, stride in zip(*cfg):
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                drop_rate = self.cfg["drop_connect_rate"] * b / blocks
                layers.append(
                    Block(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride,
                        expansion,
                        se_ratio=0.25,
                        drop_rate=drop_rate,
                    )
                )
                in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = swish(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        dropout_rate = self.cfg["dropout_rate"]
        if self.training and dropout_rate > 0:
            out = F.dropout(out, p=dropout_rate)
        out = self.linear(out)
        return out


def EfficientNetB0():
    cfg = {
        "num_blocks": [1, 2, 2, 3, 3, 4, 1],
        "expansion": [1, 6, 6, 6, 6, 6, 6],
        "out_channels": [16, 24, 40, 80, 112, 192, 320],
        "kernel_size": [3, 3, 5, 3, 5, 5, 3],
        "stride": [1, 2, 2, 2, 1, 2, 1],
        "dropout_rate": 0.2,
        "drop_connect_rate": 0.2,
    }
    return EfficientNet(cfg)


def test_efficientNetB0(test_case):
    load_pytorch_module_and_check(
        test_case, EfficientNetB0, input_size=(1, 3, 32, 32), train_flag=False,
    )
