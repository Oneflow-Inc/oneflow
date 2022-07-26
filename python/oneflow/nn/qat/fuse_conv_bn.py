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
import oneflow as flow
from oneflow.nn.module import Module
from oneflow.nn.qat.utils import init_qat_helper_modules


class QatFuseConvBN(Module):
    r"""This module provides an implementation of paper: `https://arxiv.org/pdf/1806.08342.pdf` for quantization aware training,
    which merges the parameters of batchnorm into conv.

    The QatFuseConvBN takes conv, batchnorm module and some extra parameters for fake quantization as inputs,
    see :class:`~oneflow.nn.MinMaxObserver`, :class:`~oneflow.nn.MovingAverageMinMaxObserver` and :class:`~oneflow.nn.FakeQuantization` for more details.

    Args:
        conv_module (oneflow.nn.ConvNd): Convolution module, support 1d, 2d and 3d.
        bn_module (oneflow.nn.BatchNormNd): BatchNorm module, support 1d, 2d and 3d.
        quantization_formula (str): Support "google" or "cambricon".
        quantization_bit (int): Quantize input to uintX / intX, X can be in range [2, 8]. Defaults to 8.
        quantization_scheme (str): "symmetric" or "affine", quantize to signed / unsigned integer. Defaults to "symmetric".
        weight_per_layer_quantization (bool): True or False, means per-layer / per-channel for weight quantization. Defaults to True.
        input_quant_momentum (float): Smoothing parameter for exponential moving average operation for input quantization. Defaults to 0.95.

    For example: 

    .. code-block:: python

        >>> import oneflow as flow
        >>> import oneflow.nn as nn

        >>> input = flow.rand(1, 2, 5, 5)
        >>> conv = nn.Conv2d(2, 4, kernel_size=3, stride=1)
        >>> bn = nn.BatchNorm2d(4)
        >>> qat = nn.QatFuseConvBN(conv, bn, quantization_formula="google", quantization_bit=8, quantization_scheme="symmetric")
        >>> output = qat(input)

        >>> qat.eval()
        >>> flow.nn.qat.freeze_all_qat_submodules(qat) # after training, freeze module before exporting to onnx for deployment.

    """

    def __init__(
        self,
        conv_module,
        bn_module,
        quantization_formula: str = "google",
        quantization_bit: int = 8,
        quantization_scheme: str = "symmetric",
        weight_per_layer_quantization: bool = True,
        input_quant_momentum: float = 0.95,
    ):
        super().__init__()
        self.conv_module = conv_module
        assert (
            self.conv_module.channel_pos == "channels_first"
        ), "quantization aware training only support nchw for now."
        self.bn_module = bn_module
        init_qat_helper_modules(
            self,
            quantization_formula=quantization_formula,
            quantization_bit=quantization_bit,
            quantization_scheme=quantization_scheme,
            weight_per_layer_quantization=weight_per_layer_quantization,
            input_quant_momentum=input_quant_momentum,
        )
        # indicates whether the bn parameters is truly fused into the conv parameters.
        self.is_freezed = False

    def _truly_fuse_bn_into_conv(self):
        with flow.no_grad():
            weight, bias = self._fold_bn(
                self.bn_module.running_mean,
                flow.sqrt(self.bn_module.running_var + self.bn_module.eps),
            )
            self.conv_module.weight.data = weight
            self.conv_module.bias.data = bias
            self.is_freezed = True

    def _fold_bn(self, mean, std):
        # support conv1d, 2d, and 3d
        view_param = [1] * (len(self.conv_module.kernel_size) + 1)
        view_param = [self.conv_module.out_channels] + view_param

        if self.bn_module.affine:
            gamma_ = self.bn_module.weight / std
            weight = self.conv_module.weight * gamma_.view(*view_param)
            if self.conv_module.bias is not None:
                bias = gamma_ * (self.conv_module.bias - mean) + self.bn_module.bias
            else:
                bias = self.bn_module.bias - gamma_ * mean
        else:
            gamma_ = 1 / std
            weight = self.conv_module.weight * gamma_.view(*view_param)
            if self.conv_module.bias is not None:
                bias = gamma_ * (self.conv_module.bias - mean)
            else:
                bias = -gamma_ * mean

        return weight, bias

    def forward(self, x):
        input_scale, input_zero_point = self.input_min_max_observer(
            x, self.current_train_step,
        )
        x = self.fake_quantizer(x, input_scale, input_zero_point)

        if not self.is_freezed:
            if self.training:
                y = self.conv_module(x)
                reduce_list = list(range(0, len(y.shape)))
                reduce_list.remove(1)
                mean = y.mean(reduce_list)
                var = y.var(reduce_list)
                with flow.no_grad():
                    self.bn_module.running_mean = (
                        self.bn_module.momentum * self.bn_module.running_mean
                        + (1 - self.bn_module.momentum) * mean
                    )
                    self.bn_module.running_var = (
                        self.bn_module.momentum * self.bn_module.running_var
                        + (1 - self.bn_module.momentum) * var
                    )
            else:
                mean = self.bn_module.running_mean
                var = self.bn_module.running_var

            std = flow.sqrt(var + self.bn_module.eps)

            weight, bias = self._fold_bn(mean, std)
        else:
            weight = self.conv_module.weight
            bias = self.conv_module.bias

        weight_scale, weight_zero_point = self.weight_min_max_observer(weight)
        weight = self.fake_quantizer(weight, weight_scale, weight_zero_point)

        return self.conv_module._conv_forward(x, weight, bias)


def _fuse_bn_into_conv(m):
    if type(m) == QatFuseConvBN:
        m._truly_fuse_bn_into_conv()


def freeze_all_qat_submodules(module):
    module.apply(_fuse_bn_into_conv)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
