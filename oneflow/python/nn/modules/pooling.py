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

from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.nn.module import Module
from oneflow.python.nn.modules.utils import (
    _single,
    _pair,
    _triple,
    _reverse_repeat_tuple,
)
from oneflow.python.nn.common_types import _size_1_t, _size_2_t, _size_3_t, _size_any_t
from typing import Optional, List, Tuple
from oneflow.python.ops.nn_ops import calc_pool_padding, get_dhw_offset


@oneflow_export("nn.AvgPool2d")
class AvgPool2d(Module):
    r"""Applies a 2D average pooling over an input.
    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
    output :math:`(N, C, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kH, kW)`
    can be precisely described as:
    .. math::
        out(N_i, C_j, h, w)  = \frac{1}{kH * kW} \sum_{m=0}^{kH-1} \sum_{n=0}^{kW-1}
                               input(N_i, C_j, stride[0] \times h + m, stride[1] \times w + n)
    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
    for :attr:`padding` number of points.
    Note:
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
        or the input. Sliding windows that would start in the right padded region are ignored.
    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding` can either be:
        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension
    Args:
        kernel_size: the size of the window
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: padding(value 0) to be added on both sides
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape
    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})`, where
          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] -
                \text{kernel\_size}[0]}{\text{stride}[0]} + 1\right\rfloor
          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] -
                \text{kernel\_size}[1]}{\text{stride}[1]} + 1\right\rfloor
    Examples::
        input = flow.Tensor(np.random.randn(20, 16, 50, 32))
        # pool of square window of size=3, stride=2
        m1 = flow.nn.AvgPool2d(kernel_size=3, stride=2)
        output = m1(input) # output.shape: [20, 16, 24, 15]
        # pool of non-square window
        m2 = flow.nn.AvgPool2d(kernel_size=(3, 2), stride=(2, 1))
        output = m2(input) # output.shape: [20, 16, 24, 31]
    """

    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: Optional[_size_2_t] = None,
        padding: _size_2_t = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = None,
        divisor_override: Optional[int] = None,
    ):
        super().__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride) if (stride is not None) else kernel_size
        if isinstance(padding, int):
            padding = [0, 0, padding, padding]
        elif isinstance(padding, tuple):
            padding = [0, 0, *padding]
        else:
            raise ValueError("padding should only be a int or a tuple of 2 ints")

        ceil_mode = ceil_mode

        assert count_include_pad is None, "count_include_pad not supported yet"
        assert divisor_override is None, "divisor_override not supported yet"
        count_include_pad = count_include_pad
        divisor_override = divisor_override

        _channel_pos = "channels_first"

        _padding_type, _pads_list = calc_pool_padding(
            padding, get_dhw_offset(_channel_pos), 2
        )
        _padding_before = [pad[0] for pad in _pads_list]
        _padding_after = [pad[1] for pad in _pads_list]

        self._op = (
            flow.builtin_op("avg_pool_2d")
            .Attr("data_format", _channel_pos)
            .Attr("pool_size", kernel_size)
            .Attr("strides", stride)
            .Attr("ceil_mode", ceil_mode)
            .Attr("padding", _padding_type)
            .Attr("padding_before", _padding_before)
            .Attr("padding_after", _padding_after)
            .Input("x")
            .Output("y")
            .Build()
        )

    def forward(self, x):
        res = self._op(x)[0]
        return res


@oneflow_export("nn.MaxPool2d")
class MaxPool2d(Module):
    r"""Applies a 2D max pooling over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
    output :math:`(N, C, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kH, kW)`
    can be precisely described as:

    .. math::
        \begin{aligned}
            out(N_i, C_j, h, w) ={} & \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
                                    & \text{input}(N_i, C_j, \text{stride[0]} \times h + m,
                                                   \text{stride[1]} \times w + n)
        \end{aligned}

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
    for :attr:`padding` number of points. :attr:`dilation` controls the spacing between the kernel points.
    It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    Note:
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
        or the input. Sliding windows that would start in the right padded region are ignored.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Args:
        kernel_size: the size of the window to take a max over
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on both sides
        dilation: a parameter that controls the stride of elements in the window
        return_indices: if ``True``, will return the max indices along with the outputs.
                        Useful for :class:`torch.nn.MaxUnpool2d` later
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 * \text{padding[0]} - \text{dilation[0]}
                    \times (\text{kernel\_size[0]} - 1) - 1}{\text{stride[0]}} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 * \text{padding[1]} - \text{dilation[1]}
                    \times (\text{kernel\_size[1]} - 1) - 1}{\text{stride[1]}} + 1\right\rfloor
    
    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np

        kernel_size, stride, padding = (4, 4), (1, 1), (1, 2)
        m = flow.nn.MaxPool2d(kernel_size, stride, padding)
        x = flow.Tensor(6, 4, 7, 9)
        output = m(x)

    """

    def __init__(
        self,
        kernel_size: _size_any_t,
        stride: Optional[_size_any_t] = None,
        padding: _size_any_t = 0,
        dilation: _size_any_t = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ):
        super().__init__()
        _kernel_size = _pair(kernel_size)
        _strides = _pair(stride) if (stride is not None) else kernel_size
        _padding = padding
        _dilation = dilation
        _return_indices = return_indices
        _ceil_mode = ceil_mode
        _data_format = "NCHW"
        _channel_pos = "channels_last" if _data_format == "NHWC" else "channels_first"

        assert return_indices == False, "Only support return_indices==False for now!"
        assert dilation == 1, "Only support dilation==1 for now!"

        if isinstance(padding, int):
            padding = [padding, padding]
        if len(padding) == 2:
            if _data_format == "NCHW":
                padding = (0, 0, padding[0], padding[1])
            elif _data_format == "NHWC":
                padding = (0, padding[0], padding[1], 0)
            else:
                raise ValueError("error padding param!")

        _padding_type, _pads_list = calc_pool_padding(
            padding, get_dhw_offset(_channel_pos), 2
        )
        _padding_before = [pad[0] for pad in _pads_list]
        _padding_after = [pad[1] for pad in _pads_list]

        self._op = (
            flow.builtin_op("max_pool_2d")
            .Attr("data_format", _channel_pos)
            .Attr("pool_size", _kernel_size)
            .Attr("strides", _strides)
            .Attr("ceil_mode", _ceil_mode)
            .Attr("padding", _padding_type)
            .Attr("padding_before", _padding_before)
            .Attr("padding_after", _padding_after)
            .Input("x")
            .Output("y")
            .Build()
        )

    def forward(self, x):
        res = self._op(x)[0]
        return res
