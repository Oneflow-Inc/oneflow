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
from oneflow.compatible import single_client as flow
from oneflow.compatible_single_client_python.nn.module import Module
from oneflow.compatible_single_client_python.oneflow_export import (
    oneflow_export,
    experimental_api,
)
from oneflow.compatible_single_client_python.framework.tensor import register_tensor_op
from oneflow.compatible_single_client_python.ops.transpose_util import (
    get_perm_when_transpose_axis_to_last_dim,
    get_inversed_perm,
)


class Argmax(Module):
    def __init__(self, dim: int = None, keepdim: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, input):
        if self.dim == None:
            input = flow.F.flatten(input)
            self.dim = 0

        num_axes = len(input.shape)
        axis = self.dim if self.dim >= 0 else self.dim + num_axes
        assert 0 <= axis < num_axes, "axis out of range"
        if axis == num_axes - 1:
            x = flow.F.argmax(input)
            if self.keepdim == True:
                x = flow.experimental.unsqueeze(x, -1)
            return x
        else:
            perm = get_perm_when_transpose_axis_to_last_dim(num_axes, axis)
            x = flow.F.transpose(input, perm=perm)
            x = flow.F.argmax(x)
            x = flow.experimental.unsqueeze(x, -1)
            x = flow.F.transpose(x, perm=get_inversed_perm(perm))
            if self.keepdim == False:
                x = x.squeeze(dim=[axis])
            return x


@oneflow_export("argmax")
@register_tensor_op("argmax")
@experimental_api
def argmax_op(input, dim: int = None, keepdim: bool = False):
    """The op computes the index with the largest value of a Tensor at specified axis.

    Args:
        input (oneflow.compatible.single_client.Tensor): Input Tensor
        dim (int, optional): dimension to be calculated. Defaults to the last dim (-1)
        keepdim (bool optional):  whether the output tensor has dim retained or not. Ignored if dim=None.

    Returns:
        oneflow.compatible.single_client.Tensor: A Tensor(dtype=int32) contains the index with the largest value of `input`

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow.compatible.single_client.experimental as flow
        >>> flow.enable_eager_execution()

        >>> x = np.array([[1, 3, 8, 7, 2],
        ...            [1, 9, 4, 3, 2]], dtype=np.float32)

        >>> out = flow.argmax(flow.Tensor(x))
        >>> out
        tensor([6], dtype=oneflow.int32)
        >>> out = flow.argmax(flow.Tensor(x), dim=1)
        >>> out
        tensor([2, 1], dtype=oneflow.int32)

    """
    return Argmax(dim=dim, keepdim=keepdim)(input)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
