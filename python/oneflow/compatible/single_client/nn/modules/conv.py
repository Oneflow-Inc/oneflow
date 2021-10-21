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
import math

from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.nn import init
from oneflow.compatible.single_client.nn.common_types import _size_1_t, _size_2_t
from oneflow.compatible.single_client.nn.module import Module
from oneflow.compatible.single_client.nn.modules.utils import _pair, _single


def slice(x, begin, size):
    ndim = len(x.shape)
    if not isinstance(begin, (list, tuple)) or len(begin) != ndim:
        raise ValueError(
            "begin must be a list/tuple with the same length as input tensor's number of dimensions"
        )
    if not all((isinstance(b, int) or b is None for b in begin)):
        raise ValueError("element of begin must be a int or None")
    if not isinstance(size, (list, tuple)) or len(size) != ndim:
        raise ValueError(
            "size must be a list/tuple with the same length as input tensor's number of dimensions."
        )
    if not all((isinstance(s, int) or s is None for s in size)):
        raise ValueError("element of size must be a int or None")
    slice_tup_list = []
    for (b, s, dim_size) in zip(begin, size, x.shape):
        (start, stop, step) = (None, None, 1)
        if b is not None:
            if b < -dim_size or b >= dim_size:
                raise ValueError("element of begin is out of range")
            start = b
        if s is not None:
            if s == -1:
                stop = dim_size
            else:
                if s <= 0 or s > dim_size:
                    raise ValueError("element of size is invalid")
                if b + s < dim_size:
                    stop = b + s
        slice_tup_list.append((start, stop, step))
    return flow.experimental.slice(x, slice_tup_list)


class ConvUtil(object):
    @classmethod
    def split(cls, x, axis, split_num):
        split_len = x.shape[axis] // split_num
        result_list = []
        slice_begin = [0] * len(x.shape)
        slice_size = [-1] * len(x.shape)
        slice_size[axis] = split_len
        for i in range(split_num):
            slice_begin[axis] = i * split_len
            result = slice(x, slice_begin, slice_size)
            result_list.append(result)
        return result_list


class Conv1d(Module):
    """The interface is consistent with PyTorch.    
    The documentation is referenced from: https://pytorch.org/docs/master/generated/torch.nn.Conv1d.html#conv1d
    
    Applies a 1D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\\text{in}}, L)` and output :math:`(N, C_{\\text{out}}, L_{\\text{out}})` can be
    precisely described as:

    .. math::
        \\text{out}(N_i, C_{\\text{out}_j}) = \\text{bias}(C_{\\text{out}_j}) +
        \\sum_{k = 0}^{C_{in} - 1} \\text{weight}(C_{\\text{out}_j}, k)
        \\star \\text{input}(N_i, k)

    where :math:`\\star` is the valid `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`L` is a length of signal sequence.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a one-element tuple.

    * :attr:`padding` controls the amount of padding applied to the input. It
      can be either a string {{'valid', 'same'}} or a tuple of ints giving the
      amount of implicit padding applied on both sides.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    Note:
        ``padding='valid'`` is the same as no padding. ``padding='same'`` pads
        the input so the output has the shape as the input. However, this mode
        doesn't support any stride values other than 1.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to both sides of
            the input. Default: 0
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``

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
        >>> import oneflow.compatible.single_client.experimental as flow
        >>> import oneflow.compatible.single_client.experimental.nn as nn
        >>> flow.enable_eager_execution()

        >>> arr = np.random.randn(20, 16, 50)
        >>> input = flow.Tensor(arr)
        >>> m = nn.Conv1d(16, 33, 3, stride=2)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        assert padding_mode == "zeros"
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride)
        self.padding = _single(padding)
        self.dilation = _single(dilation)
        self.groups = groups
        assert in_channels % groups == 0
        assert out_channels % groups == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = flow.nn.Parameter(
            flow.Tensor(out_channels, in_channels // groups, *self.kernel_size)
        )
        self.out_channel_groups = out_channels // groups
        self.bias = None
        if bias:
            self.bias = flow.nn.Parameter(flow.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            (fan_in, _) = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if x.device.type == "cpu" and self.groups > 1:
            in_channel_axis = 1
            weight_channel_axis = 0
            bias_channel_axis = 0
            in_split_list = ConvUtil.split(
                x, axis=in_channel_axis, split_num=self.groups
            )
            out_list = []
            for i in range(len(in_split_list)):
                out_list.append(
                    flow.F.conv1d(
                        in_split_list[i],
                        self.weight[
                            i
                            * self.out_channel_groups : (i + 1)
                            * self.out_channel_groups,
                            :,
                            :,
                        ],
                        self.bias[
                            i
                            * self.out_channel_groups : (i + 1)
                            * self.out_channel_groups
                        ]
                        if self.bias
                        else None,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=1,
                    )
                )
            res = flow.experimental.cat(out_list, dim=in_channel_axis)
        else:
            res = flow.F.conv1d(
                x,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        return res


class Conv2d(Module):
    """The interface is consistent with PyTorch.    
    The documentation is referenced from: https://pytorch.org/docs/master/generated/torch.nn.Conv2d.html#conv2d
    
    Applies a 2D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\\text{in}}, H, W)` and output :math:`(N, C_{\\text{out}}, H_{\\text{out}}, W_{\\text{out}})`
    can be precisely described as:

    .. math::
        \\text{out}(N_i, C_{\\text{out}_j}) = \\text{bias}(C_{\\text{out}_j}) +
        \\sum_{k = 0}^{C_{\\text{in}} - 1} \\text{weight}(C_{\\text{out}_j}, k) \\star \\text{input}(N_i, k)


    where :math:`\\star` is the valid 2D `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels.


    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a tuple.
    * :attr:`padding` controls the amount of implicit padding on both
      sides for :attr:`padding` number of points for each dimension.
    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.
    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\\frac{\\text{out_channels}}{\\text{in_channels}}`).,

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Note:
        When `groups == in_channels` and `out_channels == K * in_channels`,
        where `K` is a positive integer, this operation is also known as a "depthwise convolution".

        In other words, for an input of size :math:`(N, C_{in}, L_{in})`,
        a depthwise convolution with a depthwise multiplier `K` can be performed with the arguments
        :math:`(C_\\text{in}=C_\\text{in}, C_\\text{out}=C_\\text{in} \\times \\text{K}, ..., \\text{groups}=C_\\text{in})`.


    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``

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
        >>> import oneflow.compatible.single_client.experimental as flow
        >>> import oneflow.compatible.single_client.experimental.nn as nn
        >>> flow.enable_eager_execution()

        >>> arr = np.random.randn(20, 16, 50, 100)
        >>> input = flow.Tensor(arr)
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        assert padding_mode == "zeros"
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        assert in_channels % groups == 0
        assert out_channels % groups == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = flow.nn.Parameter(
            flow.Tensor(out_channels, in_channels // groups, *self.kernel_size)
        )
        self.out_channel_groups = out_channels // groups
        self.bias = None
        if bias:
            self.bias = flow.nn.Parameter(flow.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            (fan_in, _) = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if x.shape[1] != self.in_channels:
            raise ValueError("The input channels should be equal to self.in_channels")
        if x.device.type == "cpu" and self.groups > 1:
            in_channel_axis = 1
            in_split_list = ConvUtil.split(
                x, axis=in_channel_axis, split_num=self.groups
            )
            out_list = []
            for i in range(len(in_split_list)):
                out_list.append(
                    flow.F.conv2d(
                        in_split_list[i],
                        self.weight[
                            i
                            * self.out_channel_groups : (i + 1)
                            * self.out_channel_groups,
                            :,
                            :,
                            :,
                        ],
                        self.bias[
                            i
                            * self.out_channel_groups : (i + 1)
                            * self.out_channel_groups
                        ]
                        if self.bias
                        else None,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=1,
                    )
                )
            res = flow.experimental.cat(out_list, dim=in_channel_axis)
        else:
            res = flow.F.conv2d(
                x,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        return res


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
