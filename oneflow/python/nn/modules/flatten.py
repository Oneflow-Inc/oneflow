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
from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.framework.tensor import register_tensor_op


@oneflow_export("nn.Flatten")
class Flatten(Module):
    """Flattens a contiguous range of dims into a tensor. For use with: nn.Sequential.

    Args:
        start_dim: first dim to flatten (default = 1).
        end_dim: last dim to flatten (default = -1).
    

    For example: 

    .. code-block:: python 

        import oneflow as flow
        input = flow.Tensor(32, 1, 5, 5)
        m = flow.nn.Flatten()
        output = m(input)
        output.size()
        # out flow.Size([32, 25])

    """

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.op_ = (
            flow.builtin_op("flatten")
            .Input("in")
            .Output("out")
            .Attr("start_dim", start_dim)
            .Attr("end_dim", end_dim)
            .Build()
        )

    def forward(self, input):
        return self.op_(input)[0]


@oneflow_export("tmp.flatten")
@register_tensor_op("flatten")
def _flow_flatten(input, start_dim: int = 1, end_dim: int = -1):
    return Flatten(start_dim=start_dim, end_dim=end_dim)(input)
