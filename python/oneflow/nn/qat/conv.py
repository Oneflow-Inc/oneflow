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
from oneflow import nn as nn
from oneflow.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Union


def get_conv_fake_quantized(
    input, input_observer, current_train_step, weight, weight_observer, fake_quantizer
):
    in_scale, in_zero_point = input_observer(input, current_train_step)
    input_fake_quanted = fake_quantizer(input, in_scale, in_zero_point)
    w_scale, w_zero_point = weight_observer(weight)
    weight_fake_quanted = fake_quantizer(weight, w_scale, w_zero_point)
    return input_fake_quanted, weight_fake_quanted


def init_conv_fake_quants(
    self,
    quantization_formula: str = "google",
    quantization_bit: int = 8,
    quantization_scheme: str = "symmetric",
    weight_quant_per_layer: bool = True,
    input_quant_momentum: float = 0.95,
):
    self.input_min_max_observer = nn.MovingAverageMinMaxObserver(
        stop_update_after_iters=1,
        quantization_formula=quantization_formula,
        quantization_bit=quantization_bit,
        quantization_scheme=quantization_scheme,
        momentum=input_quant_momentum,
    )
    self.register_buffer("current_train_step", flow.zeros(1, dtype=flow.int64,))
    self.weight_min_max_observer = nn.MinMaxObserver(
        quantization_formula=quantization_formula,
        quantization_bit=quantization_bit,
        quantization_scheme=quantization_scheme,
        per_layer_quantization=weight_quant_per_layer,
    )
    self.fake_quantizer = nn.FakeQuantization(
        quantization_formula=quantization_formula,
        quantization_bit=quantization_bit,
        quantization_scheme=quantization_scheme,
    )


class QatConv1d(nn.Conv1d):
    r"""A Conv1d module attached with `nn.MinMaxObserver`, `nn.MovingAverageMinMaxObserver` and `nn.FakeQuantization` modules for weight and input,
    used for quantization aware training.

    The parameters of QatConv1d are the same as :class:`~oneflow.nn.Conv1d` with some extra parameters for fake quantization,
    see :class:`~oneflow.nn.MinMaxObserver`, :class:`~oneflow.nn.MovingAverageMinMaxObserver` and :class:`~oneflow.nn.FakeQuantization` for more details.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to both sides of
            the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
        padding_mode (string, optional): ``'zeros'``. Default: ``'zeros'``
        quantization_formula (str): Support "google" or "cambricon".
        quantization_bit (int): Quantize input to uintX / intX, X can be in range [2, 8]. Defaults to 8.
        quantization_scheme (str): "symmetric" or "affine", quantize to signed / unsigned integer. Defaults to "symmetric".
        weight_quant_per_layer (bool): True or False, means per-layer / per-channel for weight quantization. Defaults to True.
        input_quant_momentum (float): Smoothing parameter for exponential moving average operation for input quantization. Defaults to 0.95.

    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`(N, C_{out}, L_{out})` where

          .. math::
              L_{out} = \\left\\lfloor\\frac{L_{in} + 2 \\times \\text{padding} - \\text{dilation}
                        \\times (\\text{kernel\\_size} - 1) - 1}{\\text{stride}} + 1\\right\\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            :math:`(\\text{out\\_channels},
            \\frac{\\text{in\\_channels}}{\\text{groups}}, \\text{kernel\\_size})`.
            The values of these weights are sampled from
            :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})` where
            :math:`k = \\frac{groups}{C_\\text{in} * \\text{kernel\\_size}}`
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels). If :attr:`bias` is ``True``, then the values of these weights are
            sampled from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})` where
            :math:`k = \\frac{groups}{C_\\text{in} * \\text{kernel\\_size}}`

    For example: 

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        >>> import oneflow.nn as nn
        
        >>> arr = np.random.randn(20, 16, 50)
        >>> input = flow.Tensor(arr)
        >>> m = nn.QatConv1d(16, 33, 3, stride=2, quantization_formula="google", quantization_bit=8, quantization_scheme="symmetric")
        >>> output = m(input)

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: Union[str, _size_1_t] = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        quantization_formula: str = "google",
        quantization_bit: int = 8,
        quantization_scheme: str = "symmetric",
        weight_quant_per_layer: bool = True,
        input_quant_momentum: float = 0.95,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.channel_pos = "channels_first"
        init_conv_fake_quants(
            self,
            quantization_formula=quantization_formula,
            quantization_bit=quantization_bit,
            quantization_scheme=quantization_scheme,
            weight_quant_per_layer=weight_quant_per_layer,
            input_quant_momentum=input_quant_momentum,
        )

    def forward(self, x):
        fake_quan_input, fake_quan_weight = get_conv_fake_quantized(
            x,
            self.input_min_max_observer,
            self.current_train_step,
            self.weight,
            self.weight_min_max_observer,
            self.fake_quantizer,
        )
        return self._conv_forward(fake_quan_input, fake_quan_weight, self.bias)


class QatConv2d(nn.Conv2d):
    r"""A Conv2d module attached with `nn.MinMaxObserver`, `nn.MovingAverageMinMaxObserver` and `nn.FakeQuantization` modules for weight and input,
    used for quantization aware training.

    The parameters of QatConv2d are the same as :class:`~oneflow.nn.Conv2d` with some extra parameters for fake quantization,
    see :class:`~oneflow.nn.MinMaxObserver`, :class:`~oneflow.nn.MovingAverageMinMaxObserver` and :class:`~oneflow.nn.FakeQuantization` for more details.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
        padding_mode (string, optional): ``'zeros'``. Default: ``'zeros'``
        quantization_formula (str): Support "google" or "cambricon".
        quantization_bit (int): Quantize input to uintX / intX, X can be in range [2, 8]. Defaults to 8.
        quantization_scheme (str): "symmetric" or "affine", quantize to signed / unsigned integer. Defaults to "symmetric".
        weight_quant_per_layer (bool): True or False, means per-layer / per-channel for weight quantization. Defaults to True.
        input_quant_momentum (float): Smoothing parameter for exponential moving average operation for input quantization. Defaults to 0.95.


    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where

          .. math::
              H_{out} = \\left\\lfloor\\frac{H_{in}  + 2 \\times \\text{padding}[0] - \\text{dilation}[0]
                        \\times (\\text{kernel_size}[0] - 1) - 1}{\\text{stride}[0]} + 1\\right\\rfloor

          .. math::
              W_{out} = \\left\\lfloor\\frac{W_{in}  + 2 \\times \\text{padding}[1] - \\text{dilation}[1]
                        \\times (\\text{kernel_size}[1] - 1) - 1}{\\text{stride}[1]} + 1\\right\\rfloor

    Attr:
        - weight (Tensor): the learnable weights of the module of shape
            :math:`(\\text{out_channels}, \\frac{\\text{in_channels}}{\\text{groups}},`
            :math:`\\text{kernel_size[0]}, \\text{kernel_size[1]})`.
            The values of these weights are sampled from
            :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})` where
            :math:`k = \\frac{groups}{C_\\text{in} * \\prod_{i=0}^{1}\\text{kernel_size}[i]}`

        - bias (Tensor):   the learnable bias of the module of shape
            (out_channels). If :attr:`bias` is ``True``,
            then the values of these weights are
            sampled from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})` where
            :math:`k = \\frac{groups}{C_\\text{in} * \\prod_{i=0}^{1}\\text{kernel_size}[i]}`

    For example: 

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        >>> import oneflow.nn as nn
        
        >>> arr = np.random.randn(20, 16, 50, 100)
        >>> input = flow.Tensor(arr)
        >>> m = nn.QatConv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1), quantization_formula="google", quantization_bit=8, quantization_scheme="symmetric")
        >>> output = m(input)

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        quantization_formula: str = "google",
        quantization_bit: int = 8,
        quantization_scheme: str = "symmetric",
        weight_quant_per_layer: bool = True,
        input_quant_momentum: float = 0.95,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.channel_pos = "channels_first"
        init_conv_fake_quants(
            self,
            quantization_formula=quantization_formula,
            quantization_bit=quantization_bit,
            quantization_scheme=quantization_scheme,
            weight_quant_per_layer=weight_quant_per_layer,
            input_quant_momentum=input_quant_momentum,
        )

    def forward(self, x):
        fake_quan_input, fake_quan_weight = get_conv_fake_quantized(
            x,
            self.input_min_max_observer,
            self.current_train_step,
            self.weight,
            self.weight_min_max_observer,
            self.fake_quantizer,
        )
        return self._conv_forward(fake_quan_input, fake_quan_weight, self.bias)


class QatConv3d(nn.Conv3d):
    r"""A Conv3d module attached with `nn.MinMaxObserver`, `nn.MovingAverageMinMaxObserver` and `nn.FakeQuantization` modules for weight and input,
    used for quantization aware training.

    The parameters of QatConv3d are the same as :class:`~oneflow.nn.Conv3d` with some extra parameters for fake quantization,
    see :class:`~oneflow.nn.MinMaxObserver`, :class:`~oneflow.nn.MovingAverageMinMaxObserver` and :class:`~oneflow.nn.FakeQuantization` for more details.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all six sides of
            the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        padding_mode (string, optional): ``'zeros'``. Default: ``'zeros'``
        quantization_formula (str): Support "google" or "cambricon".
        quantization_bit (int): Quantize input to uintX / intX, X can be in range [2, 8]. Defaults to 8.
        quantization_scheme (str): "symmetric" or "affine", quantize to signed / unsigned integer. Defaults to "symmetric".
        weight_quant_per_layer (bool): True or False, means per-layer / per-channel for weight quantization. Defaults to True.
        input_quant_momentum (float): Smoothing parameter for exponential moving average operation for input quantization. Defaults to 0.95.


    Shape:
        - Input: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` where

          .. math::
              D_{out} = \left\lfloor\frac{D_{in} + 2 \times \text{padding}[0] - \text{dilation}[0]
                    \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding}[1] - \text{dilation}[1]
                    \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 \times \text{padding}[2] - \text{dilation}[2]
                    \times (\text{kernel\_size}[2] - 1) - 1}{\text{stride}[2]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
                         :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]}, \text{kernel\_size[2]})`.
                         The values of these weights are sampled from
                         :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{2}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape (out_channels). If :attr:`bias` is ``True``,
                         then the values of these weights are
                         sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{2}\text{kernel\_size}[i]}`

    For example: 

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        >>> import oneflow.nn as nn

        >>> arr = np.random.randn(1, 2, 5, 5, 5)
        >>> input = flow.Tensor(arr)
        >>> m = nn.QatConv3d(2, 4, kernel_size=3, stride=1, quantization_formula="google", quantization_bit=8, quantization_scheme="symmetric")
        >>> output = m(input)

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: Union[str, _size_3_t] = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        quantization_formula: str = "google",
        quantization_bit: int = 8,
        quantization_scheme: str = "symmetric",
        weight_quant_per_layer: bool = True,
        input_quant_momentum: float = 0.95,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.channel_pos = "channels_first"
        init_conv_fake_quants(
            self,
            quantization_formula=quantization_formula,
            quantization_bit=quantization_bit,
            quantization_scheme=quantization_scheme,
            weight_quant_per_layer=weight_quant_per_layer,
            input_quant_momentum=input_quant_momentum,
        )

    def forward(self, x):
        fake_quan_input, fake_quan_weight = get_conv_fake_quantized(
            x,
            self.input_min_max_observer,
            self.current_train_step,
            self.weight,
            self.weight_min_max_observer,
            self.fake_quantizer,
        )
        return self._conv_forward(fake_quan_input, fake_quan_weight, self.bias)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
