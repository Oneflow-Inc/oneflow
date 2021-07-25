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

from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.nn.common_types import (
    _size_1_t,
    _size_2_t,
    _size_3_t,
)
from oneflow.compatible.single_client.nn.module import Module
from oneflow.compatible.single_client.nn.modules.utils import _pair, _single, _triple
from oneflow.compatible.single_client.ops.nn_ops import (
    calc_pool_padding,
    get_dhw_offset,
)


class AvgPool1d(Module):
    """Applies a 1D average pooling over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
    output :math:`(N, C, H_{out}, W_{out})` and `kernel_size` :math:`k`
    can be precisely described as:

    .. math::

        out(N_i, C_j, l)  = \\frac{1}{k} \\sum_{m=0}^{k-1}
                               input(N_i, C_j, stride[0] \\times h + m, stride*l + m)

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
        raise NotImplementedError


class AvgPool2d(Module):
    """Performs the 2d-average pooling on the input.

    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
    output :math:`(N, C, H_{out}, W_{out})` and `kernel_size` :math:`(kH, kW)`
    can be precisely described as:

    .. math::

        out(N_i, C_j, h, w)  = \\frac{1}{kH * kW} \\sum_{m=0}^{kH-1} \\sum_{n=0}^{kW-1}
                               input(N_i, C_j, stride[0] \\times h + m, stride[1] \\times w + n)

    Args:
        kernel_size (Union[int, Tuple[int, int]]):  An int or list of ints that has length 1, 2. The size of the window for each dimension of the input Tensor.
        strides (Union[int, Tuple[int, int]]): An int or list of ints that has length 1, 2. The stride of the sliding window for each dimension of the input Tensor.
        padding (Tuple[int, int]): An int or list of ints that has length 1, 2. Implicit zero padding to be added on both sides.
        ceil_mode (bool, default to False): When True, will use ceil instead of floor to compute the output shape.

    For example:

    .. code-block:: python

        import oneflow.compatible.single_client.experimental as flow
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
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride) if stride is not None else kernel_size
        assert isinstance(padding, int) or isinstance(
            padding, tuple
        ), "padding can only int int or tuple of 2 ints."
        padding = _pair(padding)
        padding = [0, 0, *padding]
        assert count_include_pad is None, "count_include_pad not supported yet"
        assert divisor_override is None, "divisor_override not supported yet"
        self._channel_pos = "channels_first"
        (self._padding_type, _pads_list) = calc_pool_padding(
            padding, get_dhw_offset(self._channel_pos), 2
        )
        self._padding_before = [pad[0] for pad in _pads_list]
        self._padding_after = [pad[1] for pad in _pads_list]
        self.ceil_mode = ceil_mode

    def forward(self, x):
        res = flow.F.avg_pool_2d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self._padding_type,
            padding_before=self._padding_before,
            padding_after=self._padding_after,
            ceil_mode=self.ceil_mode,
            data_format=self._channel_pos,
        )
        return res


class AvgPool3d(Module):
    """Applies a 3D average pooling over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, D, H, W)`,
    output :math:`(N, C, D_{out}, H_{out}, W_{out})` and `kernel_size` :math:`(kD, kH, kW)`
    can be precisely described as:

    .. math::

        out(N_i, C_j, d, h, w)  = \\frac{1}{kD * kH * kW } \\sum_{k=0}^{kD-1} \\sum_{m=0}^{kH-1} \\sum_{n=0}^{kW-1}
                               input(N_i, C_j, stride[0] \\times d + k, stride[1] \\times h + m, stride[2] \\times w + n)
    
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
              D_{out} = \\left\\lfloor\\frac{D_{in} + 2 \\times \\text{padding}[0] - \\text{kernel_size}[0]}{\\text{stride}[0]} + 1\\right\\rfloor

          .. math::
              H_{out} = \\left\\lfloor\\frac{H_{in} + 2 \\times \\text{padding}[1] - \\text{kernel_size}[1]}{\\text{stride}[1]} + 1\\right\\rfloor

          .. math::
              W_{out} = \\left\\lfloor\\frac{W_{in} + 2 \\times \\text{padding}[2] - \\text{kernel_size}[2]}{\\text{stride}[2]} + 1\\right\\rfloor

    For example:

    .. code-block:: python

        >>> import oneflow.compatible.single_client.experimental as flow
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
        stride = _pair(stride) if stride is not None else kernel_size
        assert padding == (0, 0, 0), "padding>0 not supported yet"
        assert isinstance(padding, int) or isinstance(
            padding, tuple
        ), "padding can only int int or tuple of 3 ints."
        padding = _pair(padding)
        padding = [0, 0, *padding]
        assert count_include_pad is None, "count_include_pad not supported yet"
        assert divisor_override is None, "divisor_override not supported yet"
        _channel_pos = "channels_first"
        (_padding_type, _pads_list) = calc_pool_padding(
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


class MaxPool1d(Module):
    """The interface is consistent with PyTorch.
    The documentation is referenced from: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html#torch.nn.MaxPool1d

    Applies a 1D max pooling over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, L)`
    and output :math:`(N, C, L_{out})` can be precisely described as:

    .. math::
        out(N_i, C_j, k) = \\max_{m=0, \\ldots, \\text{kernel\\_size} - 1}
                input(N_i, C_j, stride \\times k + m)

    If :attr:`padding` is non-zero, then the input is implicitly padded with minimum value on both sides
    for :attr:`padding` number of points. :attr:`dilation` is the stride between the elements within the
    sliding window. This `link`_ has a nice visualization of the pooling parameters.

    Note:
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
        or the input. Sliding windows that would start in the right padded region are ignored.

    Args:
        kernel_size: The size of the sliding window, must be > 0.
        stride: The stride of the sliding window, must be > 0. Default value is :attr:`kernel_size`.
        padding: Implicit negative infinity padding to be added on both sides, must be >= 0 and <= kernel_size / 2.
        dilation: The stride between elements within a sliding window, must be > 0.
        return_indices: If ``True``, will return the argmax along with the max values.
                        Useful for :class:`torch.nn.MaxUnpool1d` later
        ceil_mode: If ``True``, will use `ceil` instead of `floor` to compute the output shape. This
                   ensures that every element in the input tensor is covered by a sliding window.

    Shape:
        - Input: :math:`(N, C, L_{in})`
        - Output: :math:`(N, C, L_{out})`, where

          .. math::
              L_{out} = \\left\\lfloor \\frac{L_{in} + 2 \\times \\text{padding} - \\text{dilation}
                    \\times (\\text{kernel_size} - 1) - 1}{\\text{stride}} + 1\\right\\rfloor

    """

    def __init__(
        self,
        kernel_size: _size_1_t,
        stride: Optional[_size_1_t] = None,
        padding: _size_1_t = 0,
        dilation: _size_1_t = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ):
        raise NotImplementedError


class MaxPool2d(Module):
    """The interface is consistent with PyTorch.
    The documentation is referenced from: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d

    Applies a 2D max pooling over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
    output :math:`(N, C, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kH, kW)`
    can be precisely described as:

    .. math::
        \\begin{aligned}
            out(N_i, C_j, h, w) ={} & \\max_{m=0, \\ldots, kH-1} \\max_{n=0, \\ldots, kW-1} \\\\
                                    & \\text{input}(N_i, C_j, \\text{stride[0]} \\times h + m,
                                                   \\text{stride[1]} \\times w + n)
        \\end{aligned}

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
              H_{out} = \\left\\lfloor\\frac{H_{in} + 2 * \\text{padding[0]} - \\text{dilation[0]}
                    \\times (\\text{kernel_size[0]} - 1) - 1}{\\text{stride[0]}} + 1\\right\\rfloor
          .. math::
              W_{out} = \\left\\lfloor\\frac{W_{in} + 2 * \\text{padding[1]} - \\text{dilation[1]}
                    \\times (\\text{kernel_size[1]} - 1) - 1}{\\text{stride[1]}} + 1\\right\\rfloor

    For example:

    .. code-block:: python

        >>> import oneflow.compatible.single_client.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()

        >>> kernel_size, stride, padding = (3, 3), (1, 1), (1, 2)
        >>> m = flow.nn.MaxPool2d(kernel_size, stride, padding)
        >>> np.random.seed(0)
        >>> x = flow.Tensor(np.random.rand(1, 1, 5, 3))
        >>> y = m(x)
        >>> y #doctest: +ELLIPSIS
        tensor([[[[0.5488, 0.7152, 0.7152, 0.7152, 0.6459],
                  ...
                  [0.568 , 0.9256, 0.9256, 0.9256, 0.5289]]]], dtype=oneflow.float32)

        >>> kernel_size, stride, padding = (2, 3), (4, 5), (1, 2)
        >>> m = flow.nn.MaxPool2d(kernel_size, stride, padding)
        >>> x = flow.Tensor(np.random.randn(9, 7, 32, 20))
        >>> y = m(x)
        >>> y.size()
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
        self.strides = _pair(stride) if stride is not None else kernel_size
        data_format = "NCHW"
        self.channel_pos = (
            "channels_last" if data_format == "NHWC" else "channels_first"
        )
        assert return_indices is False, "Only support return_indices==False for now!"
        assert dilation == 1 or dilation == (1, 1), "Only support dilation==1 for now!"
        padding = _pair(padding)
        if len(padding) == 2:
            if data_format == "NCHW":
                padding = (0, 0, padding[0], padding[1])
            else:
                raise ValueError("error padding param!")
        else:
            raise ValueError("error padding param!")
        (self.padding_type, pads_list) = calc_pool_padding(
            padding, get_dhw_offset(self.channel_pos), 2
        )
        self.padding_before = [pad[0] for pad in pads_list]
        self.padding_after = [pad[1] for pad in pads_list]
        self.ceil_mode = ceil_mode

    def forward(self, x):
        return flow.F.max_pool_2d(
            x,
            kernel_size=self.kernel_size,
            stride=self.strides,
            padding=self.padding_type,
            padding_before=self.padding_before,
            padding_after=self.padding_after,
            ceil_mode=self.ceil_mode,
            data_format=self.channel_pos,
        )


class MaxPool3d(Module):
    """The interface is consistent with PyTorch.
    The documentation is referenced from: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool3d.html#torch.nn.MaxPool3d

    Applies a 3D max pooling over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, D, H, W)`,
    output :math:`(N, C, D_{out}, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kD, kH, kW)`
    can be precisely described as:

    .. math::
        \\begin{aligned}
            \\text{out}(N_i, C_j, d, h, w) ={} & \\max_{k=0, \\ldots, kD-1} \\max_{m=0, \\ldots, kH-1} \\max_{n=0, \\ldots, kW-1} \\\\
                                              & \\text{input}(N_i, C_j, \\text{stride[0]} \\times d + k,
                                                             \\text{stride[1]} \\times h + m, \\text{stride[2]} \\times w + n)
        \\end{aligned}

    If :attr:`padding` is non-zero, then the input is implicitly minimum value on both sides
    for :attr:`padding` number of points. :attr:`dilation` controls the spacing between the kernel points.
    It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    Note:
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
        or the input. Sliding windows that would start in the right padded region are ignored.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the depth, height and width dimension
        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
          the second `int` for the height dimension and the third `int` for the width dimension

    Args:
        kernel_size: the size of the window to take a max over
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit minimum value padding to be added on all three sides
        dilation: a parameter that controls the stride of elements in the window
        return_indices: if ``True``, will return the max indices along with the outputs.
                        Useful for :class:`torch.nn.MaxUnpool3d` later
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})`, where

          .. math::
              D_{out} = \\left\\lfloor\\frac{D_{in} + 2 \\times \\text{padding}[0] - \\text{dilation}[0] \\times
                (\\text{kernel_size}[0] - 1) - 1}{\\text{stride}[0]} + 1\\right\\rfloor

          .. math::
              H_{out} = \\left\\lfloor\\frac{H_{in} + 2 \\times \\text{padding}[1] - \\text{dilation}[1] \\times
                (\\text{kernel_size}[1] - 1) - 1}{\\text{stride}[1]} + 1\\right\\rfloor

          .. math::
              W_{out} = \\left\\lfloor\\frac{W_{in} + 2 \\times \\text{padding}[2] - \\text{dilation}[2] \\times
                (\\text{kernel_size}[2] - 1) - 1}{\\text{stride}[2]} + 1\\right\\rfloor

    For example:

    .. code-block:: python

        >>> import oneflow.compatible.single_client.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()

        >>> kernel_size, stride, padding = (3, 3, 3), (1, 1, 1), (1, 1, 2)
        >>> m = flow.nn.MaxPool3d(kernel_size, stride, padding)
        >>> np.random.seed(0)
        >>> x = flow.Tensor(np.random.rand(1, 1, 3, 5, 3))
        >>> y = m(x)
        >>> y #doctest: +ELLIPSIS
        tensor([[[[[0.7782, 0.87  , 0.9786, 0.9786, 0.9786],
                   ...
                   [0.9447, 0.9447, 0.9447, 0.6668, 0.6668]]]]], dtype=oneflow.float32)
        >>> kernel_size, stride, padding = (2, 2, 3), (3, 4, 5), (2, 1, 2)
        >>> m = flow.nn.MaxPool3d(kernel_size, stride, padding)
        >>> x = flow.Tensor(np.random.randn(9, 7, 11, 32, 20))
        >>> y = m(x)
        >>> y.size()
        flow.Size([9, 7, 5, 9, 5])

    """

    def __init__(
        self,
        kernel_size: _size_3_t,
        stride: Optional[_size_3_t] = None,
        padding: _size_3_t = 0,
        dilation: _size_3_t = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ):
        super().__init__()
        kernel_size = _triple(kernel_size)
        strides = _triple(stride) if stride is not None else kernel_size
        data_format = "NCDHW"
        channel_pos = "channels_last" if data_format == "NDHWC" else "channels_first"
        assert return_indices is False, "Only support return_indices==False for now!"
        assert dilation == 1 or dilation == (
            1,
            1,
            1,
        ), "Only support dilation==1 for now!"
        padding = _triple(padding)
        if len(padding) == 3:
            if data_format == "NCDHW":
                padding = (0, 0, padding[0], padding[1], padding[2])
            else:
                raise ValueError("error padding param!")
        else:
            raise ValueError("error padding param!")
        (padding_type, pads_list) = calc_pool_padding(
            padding, get_dhw_offset(channel_pos), 3
        )
        padding_before = [pad[0] for pad in pads_list]
        padding_after = [pad[1] for pad in pads_list]
        self._op = (
            flow.builtin_op("max_pool_3d")
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


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
