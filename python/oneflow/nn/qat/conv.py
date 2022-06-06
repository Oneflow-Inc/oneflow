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
import oneflow.nn as nn
from oneflow.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Union


def get_fake_quantized(
    input, input_observer, current_train_step, weight, weight_observer, fake_quantizer
):
    in_scale, in_zero_point = input_observer(input, current_train_step)
    input_fake_quanted = fake_quantizer(input, in_scale, in_zero_point)
    w_scale, w_zero_point = weight_observer(weight)
    weight_fake_quanted = fake_quantizer(weight, w_scale, w_zero_point)
    return input_fake_quanted, weight_fake_quanted


def init_fake_quants(
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
    r"""A Conv1d module attached with MinMaxObserver, MovingAverageMinMaxObserver and FakeQuantize modules for weight and input,
    used for quantization aware training.

    The parameters of QatConv1d are the same as :class:`~oneflow.nn.Conv1d` with some extra parameters for fake quantization.
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
        init_fake_quants(
            self,
            quantization_formula=quantization_formula,
            quantization_bit=quantization_bit,
            quantization_scheme=quantization_scheme,
            weight_quant_per_layer=weight_quant_per_layer,
            input_quant_momentum=input_quant_momentum,
        )

    def forward(self, x):
        fake_quan_weight, fake_quan_input = get_fake_quantized(
            x,
            self.input_min_max_observer,
            self.current_train_step,
            self.weight,
            self.weight_min_max_observer,
            self.fake_quantizer,
        )
        return self._conv_forward(fake_quan_input, fake_quan_weight, self.bias)


class QatConv2d(nn.Conv2d):
    r"""A Conv2d module attached with MinMaxObserver, MovingAverageMinMaxObserver and FakeQuantize modules for weight and input,
    used for quantization aware training.

    The parameters of QatConv2d are the same as :class:`~oneflow.nn.Conv2d` with some extra parameters for fake quantization.
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
        init_fake_quants(
            self,
            quantization_formula=quantization_formula,
            quantization_bit=quantization_bit,
            quantization_scheme=quantization_scheme,
            weight_quant_per_layer=weight_quant_per_layer,
            input_quant_momentum=input_quant_momentum,
        )

    def forward(self, x):
        fake_quan_weight, fake_quan_input = get_fake_quantized(
            x,
            self.input_min_max_observer,
            self.current_train_step,
            self.weight,
            self.weight_min_max_observer,
            self.fake_quantizer,
        )
        return self._conv_forward(fake_quan_input, fake_quan_weight, self.bias)


class QatConv3d(nn.Conv3d):
    r"""A Conv3d module attached with MinMaxObserver, MovingAverageMinMaxObserver and FakeQuantize modules for weight and input,
    used for quantization aware training.

    The parameters of QatConv3d are the same as :class:`~oneflow.nn.Conv3d` with some extra parameters for fake quantization.
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
        init_fake_quants(
            self,
            quantization_formula=quantization_formula,
            quantization_bit=quantization_bit,
            quantization_scheme=quantization_scheme,
            weight_quant_per_layer=weight_quant_per_layer,
            input_quant_momentum=input_quant_momentum,
        )

    def forward(self, x):
        fake_quan_weight, fake_quan_input = get_fake_quantized(
            x,
            self.input_min_max_observer,
            self.current_train_step,
            self.weight,
            self.weight_min_max_observer,
            self.fake_quantizer,
        )
        return self._conv_forward(fake_quan_input, fake_quan_weight, self.bias)
