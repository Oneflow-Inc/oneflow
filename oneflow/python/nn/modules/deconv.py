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
from oneflow.python.nn.module import Module
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.nn.modules.utils import _pair
from oneflow.python.nn.common_types import _size_2_t
from oneflow.python.nn import init

class ConvTranspose2d(Module):
    def __init__(self, 
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        output_padding: _size_2_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        padding_mode: str = 'zeros'
    ) -> None:
        super().__init__()
        if dilation is None:
            dilation = [1, 1]
        else:
            if isinstance(dilation, (list, tuple)):
                assert len(dilation) == 2, ValueError(
                    "dilation length must be 2 when passed as a list."
                )
            elif isinstance(dilation, int):
                dilation = [dilation, dilation]
            else:
                raise ValueError("dilation must be an int or a list.")

        assert padding_mode == "zeros"
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.groups = groups
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

        

    def forward(self, x):
        return self._op(x)[0]


if __name__ == "__main__":
    import doctest

    doctest.testmod()
