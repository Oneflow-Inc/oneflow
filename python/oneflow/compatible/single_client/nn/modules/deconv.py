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
from oneflow.compatible.single_client.nn.common_types import _size_2_t
from oneflow.compatible.single_client.nn.module import Module
from oneflow.compatible.single_client.nn.modules.utils import _pair


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


class ConvTranspose2d(Module):
    """
    
    Applies a 2D transposed convolution operator over an input image composed of several input planes.

    This module can be seen as the gradient of Conv2d with respect to its input.
    It is also known as a fractionally-strided convolution or
    a deconvolution (although it is not an actual deconvolution operation).

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): ``dilation * (kernel_size - 1) - padding`` zero-padding
            will be added to both sides of each dimension in the input. Default: 0
        output_padding (int or tuple, optional): Additional size added to one side
            of each dimension in the output shape. Default: 0
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where

        .. math::
              H_{out} = (H_{in} - 1) \\times \\text{stride}[0] - 2 \\times \\text{padding}[0] + \\text{dilation}[0] 

                        \\times (\\text{kernel_size}[0] - 1) + \\text{output_padding}[0] + 1
        .. math::
              W_{out} = (W_{in} - 1) \\times \\text{stride}[1] - 2 \\times \\text{padding}[1] + \\text{dilation}[1]
              
                        \\times (\\text{kernel_size}[1] - 1) + \\text{output_padding}[1] + 1

    Attributes:
        ConvTranspose2d.weight (Tensor): the learnable weights of the module of shape
                         :math:`(\\text{in_channels}, \\frac{\\text{out_channels}}{\\text{groups}},`
                         :math:`\\text{kernel_size[0]}, \\text{kernel_size[1]})`.
                         The values of these weights are sampled from
                         :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})` where
                         :math:`k = \\frac{groups}{C_\\text{out} * \\prod_{i=0}^{1}\\text{kernel_size}[i]}`
        ConvTranspose2d.bias (Tensor): the learnable bias of the module of shape (out_channels)
                         If :attr:`bias` is ``True``, then the values of these weights are
                         sampled from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})` where
                         :math:`k = \\frac{groups}{C_\\text{out} * \\prod_{i=0}^{1}\\text{kernel_size}[i]}`

    Examples::

        >>> import numpy as np
        >>> import oneflow.compatible.single_client.experimental as flow
        >>> import oneflow.compatible.single_client.experimental.nn as nn
        >>> flow.enable_eager_execution()

        >>> m = nn.ConvTranspose2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> m = m.to("cuda")
        >>> input = flow.Tensor(np.random.randn(20, 16, 50, 100), device=flow.device("cuda"))
        >>> output = m(input)
        >>> output.size()
        flow.Size([20, 33, 93, 100])

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
        output_padding: _size_2_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        assert padding_mode == "zeros"
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        output_padding = _pair(output_padding)
        dilation = _pair(dilation)
        self.groups = groups
        assert in_channels % groups == 0
        assert out_channels % groups == 0
        self.weight = flow.nn.Parameter(
            flow.Tensor(in_channels, out_channels // groups, *kernel_size)
        )
        self.in_channel_groups = in_channels // groups
        self.bias = None
        self._bias_add_op = None
        if bias:
            self.bias = flow.nn.Parameter(flow.Tensor(out_channels))
            self._bias_add_op = (
                flow.builtin_op("bias_add")
                .Input("a")
                .Input("b")
                .Output("out")
                .Attr("axis", 1)
                .Build()
            )
        self._op = (
            flow.builtin_op("deconv2d")
            .Input("in")
            .Input("weight")
            .Attr("filters", out_channels // groups)
            .Attr("padding_before", padding)
            .Attr("data_format", "channels_first")
            .Attr("kernel_size", kernel_size)
            .Attr("strides", stride)
            .Attr("dilation_rate", dilation)
            .Attr("output_padding", output_padding)
            .Attr("groups", 1)
            .Output("out")
            .Build()
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            (fan_in, _) = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if self.groups > 1:
            in_channel_axis = 1
            in_split_list = ConvUtil.split(
                x, axis=in_channel_axis, split_num=self.groups
            )
            out_list = []
            for i in range(len(in_split_list)):
                out_list.append(
                    self._op(
                        in_split_list[i],
                        self.weight[
                            i
                            * self.in_channel_groups : (i + 1)
                            * self.in_channel_groups,
                            :,
                            :,
                            :,
                        ],
                    )[0]
                )
            res = flow.experimental.cat(out_list, dim=in_channel_axis)
        else:
            res = self._op(x, self.weight)[0]
        if self._bias_add_op is not None:
            res = self._bias_add_op(res, self.bias)[0]
        return res


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
