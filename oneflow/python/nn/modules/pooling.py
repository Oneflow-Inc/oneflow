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
from typing import Optional, List, Tuple

import oneflow as flow
from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.nn.module import Module
from oneflow.python.nn.modules.utils import _pair
from oneflow.python.nn.common_types import _size_2_t
from oneflow.python.ops.nn_ops import calc_pool_padding, get_dhw_offset


@oneflow_export("nn.AvgPool2d")
class AvgPool2d(Module):
    r"""Performs the 2d-average pooling on the input.

    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
    output :math:`(N, C, H_{out}, W_{out})` and `kernel_size` :math:`(kH, kW)`
    can be precisely described as:
        
    .. math::

        out(N_i, C_j, h, w)  = \frac{1}{kH * kW} \sum_{m=0}^{kH-1} \sum_{n=0}^{kW-1}
                               input(N_i, C_j, stride[0] \times h + m, stride[1] \times w + n)

    Args:
        kernel_size (Union[int, Tuple[int, int]]):  An int or list of ints that has length 1, 2. The size of the window for each dimension of the input Tensor.
        strides (Union[int, Tuple[int, int]]): An int or list of ints that has length 1, 2. The stride of the sliding window for each dimension of the input Tensor.
        padding (Tuple[int, int]): An int or list of ints that has length 1, 2. Implicit zero padding to be added on both sides.
        ceil_mode (bool, default to False): When True, will use ceil instead of floor to compute the output shape.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np


        of_avgpool2d = flow.nn.AvgPool2d(
            kernel_size=(3, 2),
            padding=0,
            stride=(2, 1),
        )
        x = flow.Tensor(shape=(1, 1, 10, 10))
        of_y = of_avgpool2d(x)

    """

    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: Optional[_size_2_t] = None,
        padding: _size_2_t = 0,
        ceil_mode: bool = False,
        count_include_pad: Optional[bool] = None,
        divisor_override: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super().__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride) if (stride is not None) else kernel_size

        assert isinstance(padding, int) or isinstance(
            padding, tuple
        ), "padding can only int int or tuple of 2 ints."
        padding = _pair(padding)
        padding = [0, 0, *padding]

        assert count_include_pad is None, "count_include_pad not supported yet"
        assert divisor_override is None, "divisor_override not supported yet"

        _channel_pos = "channels_first"
        # TODO(yaochi): align with pytorch when padding is asymmetric
        _padding_type, _pads_list = calc_pool_padding(
            padding, get_dhw_offset(_channel_pos), 2
        )
        _padding_before = [pad[0] for pad in _pads_list]
        _padding_after = [pad[1] for pad in _pads_list]

        self._op = (
            flow.builtin_op("avg_pool_2d", name)
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
    If :attr:`padding` is non-zero, then the input is implicitly minimum value padded on both sides
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
        padding: implicit minimum value padding to be added on both sides
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
        x = flow.Tensor(np.random.rand(6, 4, 7, 9))
        y = m(x)
    """

    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: Optional[_size_2_t] = None,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ):
        super().__init__()
        kernel_size = _pair(kernel_size)
        strides = _pair(stride) if (stride is not None) else kernel_size
        data_format = "NCHW"
        channel_pos = "channels_last" if data_format == "NHWC" else "channels_first"

        assert return_indices == False, "Only support return_indices==False for now!"
        assert dilation == 1, "Only support dilation==1 for now!"

        padding = _pair(padding)
        if len(padding) == 2:
            if data_format == "NCHW":
                padding = (0, 0, padding[0], padding[1])
            else:
                raise ValueError("error padding param!")
        else:
            raise ValueError("error padding param!")

        padding_type, pads_list = calc_pool_padding(
            padding, get_dhw_offset(channel_pos), 2
        )
        padding_before = [pad[0] for pad in pads_list]
        padding_after = [pad[1] for pad in pads_list]

        self._op = (
            flow.builtin_op("max_pool_2d")
            .Attr("data_format", channel_pos)
            .Attr("pool_size", kernel_size)
            .Attr("strides", strides)
            .Attr("ceil_mode", ceil_mode)
            .Attr("padding", padding_type)
            .Attr("padding_before", padding_before)
            .Attr("padding_after", padding_after)
            .Input("x")
            .Output("y")
            .Build()
        )

    def forward(self, x):
        return self._op(x)[0]
