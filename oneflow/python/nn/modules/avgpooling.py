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
from typing import Optional

import oneflow as flow
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.nn.module import Module
from oneflow.python.nn.modules.utils import _single, _pair, _triple
from oneflow.python.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from oneflow.python.ops.nn_ops import calc_pool_padding, get_dhw_offset


@oneflow_export("nn.AvgPool1d")
@experimental_api
class AvgPool1d(Module):
    r"""Applies a 1D average pooling over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
    output :math:`(N, C, H_{out}, W_{out})` and `kernel_size` :math:`k`
    can be precisely described as:

    .. math::

        out(N_i, C_j, l)  = \frac{1}{k} \sum_{m=0}^{k-1}
                               input(N_i, C_j, stride[0] \times h + m, stride*l + m)

    If padding is non-zero, then the input is implicitly zero-padded on both sides for padding number of points.
    The parameters kernel_size, stride, padding can each be an int or a one-element tuple.

    Note:
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding or the
        input. Sliding windows that would start in the right padded region are ignored.
    
    Args:
        kernel_size: the size of the window.
        strides: the stride of the window. Default value is kernel_size.
        padding: implicit zero padding to be added on both sides.
        ceil_mode: when True, will use ceil instead of floor to compute the output shape.
        count_include_pad: when True, will include the zero-padding in the averaging calculation.


    # TODO: fix cuDNN bugs in pooling_1d
    
    """

    def __init__(
        self,
        kernel_size: _size_1_t,
        stride: Optional[_size_1_t] = None,
        padding: _size_1_t = 0,
        ceil_mode: bool = False,
        count_include_pad: Optional[bool] = None,
        name: Optional[str] = None,
    ):
        # TODO: fix cuDNN bugs in pooling_1d
        raise NotImplementedError


@oneflow_export("nn.AvgPool3d")
@experimental_api
class AvgPool3d(Module):
    r"""Applies a 3D average pooling over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, D, H, W)`,
    output :math:`(N, C, D_{out}, H_{out}, W_{out})` and `kernel_size` :math:`(kD, kH, kW)`
    can be precisely described as:

    .. math::

        out(N_i, C_j, d, h, w)  = \frac{1}{kD * kH * kW } \sum_{k=0}^{kD-1} \sum_{m=0}^{kH-1} \sum_{n=0}^{kW-1}
                               input(N_i, C_j, stride[0] \times d + k, stride[1] \times h + m, stride[2] \times w + n)
    
    If padding is non-zero, then the input is implicitly zero-padded on all three sides for padding number of points.

    Note:
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding or the
        input. Sliding windows that would start in the right padded region are ignored.

    Args:
        kernel_size: the size of the window.
        strides:  the stride of the window. Default value is kernel_size.
        padding:  implicit zero padding to be added on all three sides.
        ceil_mode:  when True, will use ceil instead of floor to compute the output shape.
        count_include_pad: when True, will include the zero-padding in the averaging calculation.
        divisor_override: if specified, it will be used as divisor, otherwise kernel_size will be used.

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})`, where

          .. math::
              D_{out} = \left\lfloor\frac{D_{in} + 2 \times \text{padding}[0] - \text{kernel_size}[0]}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding}[1] - \text{kernel_size}[1]}{\text{stride}[1]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 \times \text{padding}[2] - \text{kernel_size}[2]}{\text{stride}[2]} + 1\right\rfloor

    For example:

    .. code-block:: python

        >>> import oneflow.experimental as flow
        >>> import numpy as np

        >>> flow.enable_eager_execution()
        >>> inputarr = np.random.randn(9, 7, 11, 32, 20)
        >>> of_avgpool3d = flow.nn.AvgPool3d(kernel_size=(2,2,2),padding=(0,0,0),stride=(1,1,1),)
        >>> x = flow.Tensor(inputarr)
        >>> y = of_avgpool3d(x)

    """

    def __init__(
        self,
        kernel_size: _size_3_t,
        stride: Optional[_size_3_t] = None,
        padding: _size_3_t = 0,
        ceil_mode: bool = False,
        count_include_pad: Optional[bool] = None,
        divisor_override: Optional[int] = None,
    ):
        super().__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride) if (stride is not None) else kernel_size

        assert padding == (0, 0, 0), "padding>0 not supported yet"
        assert isinstance(padding, int) or isinstance(
            padding, tuple
        ), "padding can only int int or tuple of 3 ints."
        padding = _pair(padding)
        padding = [0, 0, *padding]

        assert count_include_pad is None, "count_include_pad not supported yet"
        assert divisor_override is None, "divisor_override not supported yet"

        _channel_pos = "channels_first"
        # TODO(yaochi): align with pytorch when padding is asymmetric
        _padding_type, _pads_list = calc_pool_padding(
            padding, get_dhw_offset(_channel_pos), 3
        )
        _padding_before = [pad[0] for pad in _pads_list]
        _padding_after = [pad[1] for pad in _pads_list]

        self._op = (
            flow.builtin_op("avg_pool_3d")
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


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
