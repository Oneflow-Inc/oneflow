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


class Cast(Module):
    def __init__(self, dtype: flow.dtype) -> None:
        super().__init__()
        self._op = (
            flow.builtin_op("cast")
            .Input("in")
            .Output("out")
            .Attr("dtype", dtype)
            .Build()
        )

    def forward(self, x):
        return self._op(x)[0]


@oneflow_export("tmp.cast")
@register_tensor_op("cast")
def cast_op(x, dtype):
    r"""The operation takes input tensor `x` and casts it to the output with `dtype`

    Args:
        x (oneflow.Tensor): A Tensor
        dtype (flow.dtype): Data type of the output tensor

    Returns:
        oneflow.Tensor: A Tensor with specific dtype.

    For example:

    .. code-block:: python

        import oneflow.experimental as flow
        import numpy as np

        np_arr = np.random.randn(2, 3, 4, 5).astype(np.float32)
        input = flow.Tensor(np_arr, dtype=flow.float32)
        output = flow.cast(input, flow.int8)
        
        # equal to np_arr.astype(np.int8)

    """
    return Cast(dtype)(x)
