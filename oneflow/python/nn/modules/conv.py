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
import oneflow as flow
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.nn.module import Module
from oneflow.python.nn.modules.utils import _single, _pair
from oneflow.python.nn.common_types import _size_1_t, _size_2_t
from oneflow.python.nn import init


def slice(x, begin, size):
    ndim = len(x.shape)
    if not isinstance(begin, (list, tuple)) or len(begin) != ndim:
        raise ValueError(
            "begin must be a list/tuple with the same length as input tensor's number of dimensions"
        )

    if not all(isinstance(b, int) or b is None for b in begin):
        raise ValueError("element of begin must be a int or None")

    if not isinstance(size, (list, tuple)) or len(size) != ndim:
        raise ValueError(
            "size must be a list/tuple with the same length as input tensor's number of dimensions."
        )

    if not all(isinstance(s, int) or s is None for s in size):
        raise ValueError("element of size must be a int or None")

    slice_tup_list = []
    for b, s, dim_size in zip(begin, size, x.shape):
        start, stop, step = (None, None, 1)
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

@oneflow_export("nn.Conv1d")
@experimental_api
class Conv1d(Module):
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
        padding_mode: str = "zeros",  # TODO: refine this type
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
        self.bias = None
        if bias:
            self.bias = flow.nn.Parameter(flow.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if x.device.type == "cpu" and self.groups > 1:
            in_channel_axis = 1
            in_split_list = ConvUtil.split(
                x, axis=in_channel_axis, split_num=self.groups
            )
            out_list = []
            for i in range(len(in_split_list)):
                out_list.append(
                    flow.F.conv1d(
                        in_split_list[i],
                        self.weight[i : i + 1, :, :],
                        self.bias[i : i + 1],
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


@oneflow_export("nn.Conv2d")
@experimental_api
class Conv2d(Module):
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
        padding_mode: str = "zeros",  # TODO: refine this type
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
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
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
                        self.weight[i : i + 1, :, :, :],
                        self.bias[i : i + 1, :, :, :],
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
