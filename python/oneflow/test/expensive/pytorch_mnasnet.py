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
import torch
import torch.nn as nn
from torch import Tensor

import warnings
from typing import Any, Dict, List

__all__ = [
    "MNASNet",
    "mnasnet1_0",
]


# Paper suggests 0.9997 momentum, for TensorFlow. Equivalent PyTorch momentum is
# 1.0 - tensorflow.
_BN_MOMENTUM = 1 - 0.9997


class _InvertedResidual(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int,
        expansion_factor: int,
        bn_momentum: float = 0.1,
    ) -> None:
        super().__init__()
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")
        if kernel_size not in [3, 5]:
            raise ValueError(f"kernel_size should be 3 or 5 instead of {kernel_size}")
        mid_ch = in_ch * expansion_factor
        self.apply_residual = in_ch == out_ch and stride == 1
        self.layers = nn.Sequential(
            # Pointwise
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # Depthwise
            nn.Conv2d(
                mid_ch,
                mid_ch,
                kernel_size,
                padding=kernel_size // 2,
                stride=stride,
                groups=mid_ch,
                bias=False,
            ),
            nn.BatchNorm2d(mid_ch, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # Linear pointwise. Note that there's no activation.
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch, momentum=bn_momentum),
        )

    def forward(self, input: Tensor) -> Tensor:
        if self.apply_residual:
            return self.layers(input) + input
        else:
            return self.layers(input)


def _stack(
    in_ch: int,
    out_ch: int,
    kernel_size: int,
    stride: int,
    exp_factor: int,
    repeats: int,
    bn_momentum: float,
) -> nn.Sequential:
    """Creates a stack of inverted residuals."""
    if repeats < 1:
        raise ValueError(f"repeats should be >= 1, instead got {repeats}")
    # First one has no skip, because feature map size changes.
    first = _InvertedResidual(
        in_ch, out_ch, kernel_size, stride, exp_factor, bn_momentum=bn_momentum
    )
    remaining = []
    for _ in range(1, repeats):
        remaining.append(
            _InvertedResidual(
                out_ch, out_ch, kernel_size, 1, exp_factor, bn_momentum=bn_momentum
            )
        )
    return nn.Sequential(first, *remaining)


def _round_to_multiple_of(val: float, divisor: int, round_up_bias: float = 0.9) -> int:
    """Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, unless the number is no more than 10% greater than the
    smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88."""
    if not 0.0 < round_up_bias < 1.0:
        raise ValueError(
            f"round_up_bias should be greater than 0.0 and smaller than 1.0 instead of {round_up_bias}"
        )
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor


def _get_depths(alpha: float) -> List[int]:
    """Scales tensor depths as in reference MobileNet code, prefers rouding up
    rather than down."""
    depths = [32, 16, 24, 40, 80, 96, 192, 320]
    return [_round_to_multiple_of(depth * alpha, 8) for depth in depths]


class MNASNet(torch.nn.Module):
    """MNASNet, as described in https://arxiv.org/pdf/1807.11626.pdf. This
    implements the B1 variant of the model.
    >>> model = MNASNet(1.0, num_classes=1000)
    >>> x = torch.rand(1, 3, 224, 224)
    >>> y = model(x)
    >>> y.dim()
    2
    >>> y.nelement()
    1000
    """

    # Version 2 adds depth scaling in the initial stages of the network.
    _version = 2

    def __init__(
        self, alpha: float, num_classes: int = 1000, dropout: float = 0.2
    ) -> None:
        super().__init__()
        if alpha <= 0.0:
            raise ValueError(f"alpha should be greater than 0.0 instead of {alpha}")
        self.alpha = alpha
        self.num_classes = num_classes
        depths = _get_depths(alpha)
        layers = [
            # First layer: regular conv.
            nn.Conv2d(3, depths[0], 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(depths[0], momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
            # Depthwise separable, no skip.
            nn.Conv2d(
                depths[0],
                depths[0],
                3,
                padding=1,
                stride=1,
                groups=depths[0],
                bias=False,
            ),
            nn.BatchNorm2d(depths[0], momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(depths[0], depths[1], 1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(depths[1], momentum=_BN_MOMENTUM),
            # MNASNet blocks: stacks of inverted residuals.
            _stack(depths[1], depths[2], 3, 2, 3, 3, _BN_MOMENTUM),
            _stack(depths[2], depths[3], 5, 2, 3, 3, _BN_MOMENTUM),
            _stack(depths[3], depths[4], 5, 2, 6, 3, _BN_MOMENTUM),
            _stack(depths[4], depths[5], 3, 1, 6, 2, _BN_MOMENTUM),
            _stack(depths[5], depths[6], 5, 2, 6, 4, _BN_MOMENTUM),
            _stack(depths[6], depths[7], 3, 1, 6, 1, _BN_MOMENTUM),
            # Final mapping to classifier input.
            nn.Conv2d(depths[7], 1280, 1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(1280, momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
        ]
        self.layers = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True), nn.Linear(1280, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(
                    m.weight, mode="fan_out", nonlinearity="sigmoid"
                )
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        # Equivalent to global avgpool and removing H and W dimensions.
        x = x.mean([2, 3])
        return self.classifier(x)


def _mnasnet(alpha: float, progress: bool, **kwargs: Any) -> MNASNet:
    model = MNASNet(alpha, **kwargs)
    return model


def mnasnet1_0(progress: bool = True, **kwargs: Any) -> MNASNet:
    r"""MNASNet with depth multiplier of 1.0 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.
    Args:
        weights (MNASNet1_0_Weights, optional): The pretrained weights for the model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _mnasnet(1.0, progress, **kwargs)
