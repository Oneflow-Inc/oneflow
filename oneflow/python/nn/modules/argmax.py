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
from oneflow.python.ops.transpose_util import (
    get_perm_when_transpose_axis_to_last_dim,
    get_inversed_perm,
)


class Argmax(Module):
    def __init__(self, dim: int = None, keepdim: bool = False) -> None:
        super().__init__()
        self._op_softmax_last_dim = (
            flow.builtin_op("argmax").Input("in").Output("out").Build()
        )
        self._expand_op = (
            flow.builtin_op("expand_dims")
            .Input("in")
            .Output("out")
            .Attr("axis", -1)
            .Build()
        )
        self._flatten = (
            flow.builtin_op("flatten")
            .Input("in")
            .Output("out")
            .Attr("start_dim", 0)
            .Attr("end_dim", -1)
            .Build()
        )

        self.dim = dim
        self.keepdim = keepdim

    def forward(self, input):
        if self.dim == None:
            input = self._flatten(input)[0]
            self.dim = 0

        num_axes = len(input.shape)
        axis = self.dim if self.dim >= 0 else self.dim + num_axes
        assert 0 <= axis < num_axes, "axis out of range"
        if axis == num_axes - 1:
            x = self._op_softmax_last_dim(input)[0]
            if self.keepdim == True:
                x = self._expand_op(x)
            return x
        else:
            perm = get_perm_when_transpose_axis_to_last_dim(num_axes, axis)
            x = flow.tmp.transpose(input, perm=perm)
            x = self._op_softmax_last_dim(x)[0]
            x = self._expand_op(x)[0]
            x = flow.tmp.transpose(x, perm=get_inversed_perm(perm))
            if self.keepdim == False:
                x = flow.tmp.squeeze(x, axis=[axis])
            return x


@oneflow_export("argmax")
@register_tensor_op("argmax")
def argmax_op(input, dim: int = None, keepdim: bool = False):
    """The op computes the index with the largest value of a Tensor at specified axis.

    Args:
        input (oneflow.Tensor): Input Tensor
        dim (int, optional): dimension to be calculated. Defaults to the last dim (-1)
        keepdim (bool optional):  whether the output tensor has dim retained or not. Ignored if dim=None.

    Returns:
        oneflow.Tensor: A Tensor(dtype=int32) contains the index with the largest value of `input`

    For example:

    .. code-block:: python 

        import oneflow as flow
        import numpy as np

        x = np.array([[1, 3, 8, 7, 2],
                    [1, 9, 4, 3, 2]], dtype=np.float32)

        out = flow.argmax(flow.Tensor(x))

        # out [2 1]

    """
    return Argmax(dim=dim, keepdim=keepdim)(input)
