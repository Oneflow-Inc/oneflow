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
from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.nn.module import Module
from oneflow.python.nn.modules.utils import _getint, _single, _pair, _triple
from oneflow.python.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from oneflow.python.ops.nn_ops import calc_pool_padding, get_dhw_offset, _GetSequence


@oneflow_export("nn.AvgPool1d")
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
    ):
        # TODO: fix cuDNN bugs in pooling_1d
        raise NotImplementedError


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
    ):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride) if (stride is not None) else _pair(kernel_size)

        assert isinstance(padding, int) or isinstance(
            padding, tuple
        ), "padding can only int int or tuple of 2 ints."
        padding = _pair(padding)
        self.padding = padding
        padding = [0, 0, *padding]

        assert count_include_pad is None, "count_include_pad not supported yet"
        assert divisor_override is None, "divisor_override not supported yet"

        self._channel_pos = "channels_first"
        # TODO(yaochi): align with pytorch when padding is asymmetric
        self._padding_type, _pads_list = calc_pool_padding(
            padding, get_dhw_offset(self._channel_pos), 2
        )
        self._padding_before = [pad[0] for pad in _pads_list]
        self._padding_after = [pad[1] for pad in _pads_list]
        self.ceil_mode = ceil_mode

    def forward(self, x):
        return flow.F.avg_pool_2d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self._padding_type,
            padding_before=self._padding_before,
            padding_after=self._padding_after,
            ceil_mode=self.ceil_mode,
            data_format=self._channel_pos,
        )

    def extra_repr(self) -> str:
        return (
            "kernel_size={kernel_size}, stride={stride}, padding={padding}"
            ", ceil_mode={ceil_mode}".format(**self.__dict__)
        )


@oneflow_export("nn.AvgPool3d")
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

        >>> import oneflow as flow
        >>> import numpy as np
        
        >>> m = flow.nn.AvgPool3d(kernel_size=(2,2,2),padding=(0,0,0),stride=(1,1,1))
        >>> x = flow.Tensor(np.random.randn(9, 7, 11, 32, 20))
        >>> y = m(x)
        >>> y.shape
        flow.Size([9, 7, 10, 31, 19])

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
        kernel_size = _triple(kernel_size)
        stride = _triple(stride) if (stride is not None) else _triple(kernel_size)

        assert padding == (0, 0, 0), "padding>0 not supported yet"
        assert isinstance(padding, int) or isinstance(
            padding, tuple
        ), "padding can only int int or tuple of 3 ints."
        padding = _triple(padding)
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
        return self._op(x)[0]


@oneflow_export("nn.MaxPool2d")
class MaxPool2d(Module):
    r"""The interface is consistent with PyTorch.
    The documentation is referenced from: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d

    Applies a 2D max pooling over an input signal composed of several input planes.

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
                    \times (\text{kernel_size[0]} - 1) - 1}{\text{stride[0]}} + 1\right\rfloor
          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 * \text{padding[1]} - \text{dilation[1]}
                    \times (\text{kernel_size[1]} - 1) - 1}{\text{stride[1]}} + 1\right\rfloor

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> kernel_size, stride, padding = (3, 4), (1, 1), (1, 2)
        >>> m = flow.experimental.nn.MaxPool2d(kernel_size, stride, padding)
        >>> np.random.seed(0)
        >>> x = flow.Tensor(np.random.rand(1, 1, 5, 3))
        >>> y = m(x)
        >>> y #doctest: +ELLIPSIS
        tensor([[[[0.7152, 0.7152, 0.7152, 0.7152],
                  ...
                  [0.9256, 0.9256, 0.9256, 0.9256]]]], dtype=oneflow.float32)

        >>> kernel_size, stride, padding = (2, 4), (4, 5), (1, 2)
        >>> m = flow.experimental.nn.MaxPool2d(kernel_size, stride, padding)
        >>> x = flow.Tensor(np.random.randn(9, 7, 32, 20))
        >>> y = m(x)
        >>> y.shape
        flow.Size([9, 7, 9, 5])

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
        self.kernel_size = _pair(kernel_size)
        data_format = "NCHW"  # only support "NCHW" for now !
        self.channel_pos = (
            "channels_first" if data_format == "NCHW" else "channels_last"
        )
        self.stride = _pair(stride) if (stride is not None) else _pair(kernel_size)
        self.dilation = _GetSequence(dilation, 2, "dilation")
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        self.padding = _pair(padding)

    def forward(self, x):
        y, indice = flow.F.maxpool_2d(
            x,
            data_format=self.channel_pos,
            padding=self.padding,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            return_indices=True,
            ceil_mode=self.ceil_mode,
        )
        if self.return_indices:
            return y, indice
        else:
            return y

    def extra_repr(self) -> str:
        return "kernel_size={}, stride={}, padding={}, dilation={}".format(
            self.kernel_size, self.stride, self.padding, self.dilation
        )


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
