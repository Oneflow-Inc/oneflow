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
from oneflow.python.nn.modules.utils import _pair
from oneflow.python.nn.common_types import _size_2_t
from oneflow.python.nn import init


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
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.weight = flow.nn.Parameter(
            flow.Tensor(out_channels, in_channels // groups, *kernel_size)
        )
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
            flow.builtin_op("conv2d")
            .Input("in")
            .Input("weight")
            .Attr("filters", out_channels)
            .Attr("padding_before", padding)
            .Attr("strides", stride)
            .Attr("kernel_size", kernel_size)
            .Attr("dilation_rate", dilation)
            .Attr("groups", groups)
            .Attr("data_format", "channels_first")
            .Output("out")
            .Build()
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        res = self._op(x, self.weight)[0]
        if self._bias_add_op is not None:
            res = self._bias_add_op(res, self.bias)[0]
        return res
