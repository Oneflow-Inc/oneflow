"""
Modified from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
"""
from typing import Type, Any, Callable, Union, List, Optional

import oneflow as flow
import oneflow.nn as nn
from oneflow import Tensor

# from .registry import ModelCreator

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


model_urls = {
    "resnet18": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/ResNet/resnet18.zip",
    "resnet34": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/ResNet/resnet34.zip",
    "resnet50": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/ResNet/resnet50.zip",
    "resnet101": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/ResNet/resnet101.zip",
    "resnet152": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/ResNet/resnet152.zip",
    "resnext50_32x4d": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/ResNet/resnext50_32x4d.zip",
    "resnext101_32x8d": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/ResNet/resnext101_32x8d.zip",
    "wide_resnet50_2": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/ResNet/wide_resnet50_2.zip",
    "wide_resnet101_2": "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/ResNet/wide_resnet101_2.zip",
}


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = flow.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = flow.hub.load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


# @ModelCreator.register_model
def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """
    Constructs the ResNet-18 model.
    .. note::
        `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``
    For example:
    .. code-block:: python
        >>> import flowvision
        >>> resnet18 = flowvision.models.resnet18(pretrained=False, progress=True)
    """
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


# @ModelCreator.register_model
def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """
    Constructs the ResNet-34 model.
    .. note::
        `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``
    For example:
    .. code-block:: python
        >>> import flowvision
        >>> resnet34 = flowvision.models.resnet34(pretrained=False, progress=True)
    """
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


# @ModelCreator.register_model
def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """
    Constructs the ResNet-50 model.
    .. note::
        `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``
    For example:
    .. code-block:: python
        >>> import flowvision
        >>> resnet50 = flowvision.models.resnet50(pretrained=False, progress=True)
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


# @ModelCreator.register_model
def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """
    Constructs the ResNet-101 model.
    .. note::
        `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``
    For example:
    .. code-block:: python
        >>> import flowvision
        >>> resnet101 = flowvision.models.resnet101(pretrained=False, progress=True)
    """
    return _resnet(
        "resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


# @ModelCreator.register_model
def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """
    Constructs the ResNet-152 model.
    .. note::
        `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``
    For example:
    .. code-block:: python
        >>> import flowvision
        >>> resnet152 = flowvision.models.resnet152(pretrained=False, progress=True)
    """
    return _resnet(
        "resnet152", Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs
    )


# @ModelCreator.register_model
def resnext50_32x4d(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> ResNet:
    """
    Constructs the ResNeXt-50 32x4d model.
    .. note::
        `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``
    For example:
    .. code-block:: python
        >>> import flowvision
        >>> resnext50_32x4d = flowvision.models.resnext50_32x4d(pretrained=False, progress=True)
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet(
        "resnext50_32x4d", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


# @ModelCreator.register_model
def resnext101_32x8d(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> ResNet:
    """
    Constructs the ResNeXt-101 32x8d model.
    .. note::
        `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``
    For example:
    .. code-block:: python
        >>> import flowvision
        >>> resnext101_32x8d = flowvision.models.resnext101_32x8d(pretrained=False, progress=True)
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet(
        "resnext101_32x8d", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


# @ModelCreator.register_model
def wide_resnet50_2(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> ResNet:
    """
    Constructs the Wide ResNet-50-2 model.
    .. note::
        `Wide Residual Networks <https://arxiv.org/pdf/1605.07146.pdf>`_.
        The model is the same as ResNet except for the bottleneck number of channels
        which is twice larger in every block. The number of channels in outer 1x1
        convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
        channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``
    For example:
    .. code-block:: python
        >>> import flowvision
        >>> wide_resnet50_2 = flowvision.models.wide_resnet50_2(pretrained=False, progress=True)
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet(
        "wide_resnet50_2", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


# @ModelCreator.register_model
def wide_resnet101_2(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> ResNet:
    """
    Constructs the Wide ResNet-101-2 model.
    .. note::
        `Wide Residual Networks <https://arxiv.org/pdf/1605.07146.pdf>`_.
        The model is the same as ResNet except for the bottleneck number of channels
        which is twice larger in every block. The number of channels in outer 1x1
        convolutions is the same, e.g. 
    Args:
        pretrained (bool): Whether to download the pre-trained model on ImageNet. Default: ``False``
        progress (bool): If True, displays a progress bar of the download to stderr. Default: ``True``
    For example:
    .. code-block:: python
        >>> import flowvision
        >>> wide_resnet101_2 = flowvision.models.wide_resnet101_2(pretrained=False, progress=True)
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet(
        "wide_resnet101_2", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )