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
from oneflow.framework.tensor import register_tensor_op
from oneflow.nn.module import Module
from oneflow.ops.transpose_util import (
    get_inversed_perm,
    get_perm_when_transpose_axis_to_last_dim,
)


class Argmin(Module):
    def __init__(self, dim: int = None, keepdim: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, input):
        if self.dim == None:
            input = flow._C.flatten(input)
            self.dim = 0
        num_axes = len(input.shape)
        axis = self.dim if self.dim >= 0 else self.dim + num_axes
        assert 0 <= axis < num_axes, "axis out of range"
        if axis == num_axes - 1:
            x = flow._C.argmin(input)
            if self.keepdim == True:
                x = flow.unsqueeze(x, -1)
            return x
        else:
            perm = get_perm_when_transpose_axis_to_last_dim(num_axes, axis)
            x = flow._C.transpose(input, perm=perm)
            x = flow._C.argmin(x)
            x = flow.unsqueeze(x, -1)
            x = flow._C.transpose(x, perm=get_inversed_perm(perm))
            if self.keepdim == False:
                x = x.squeeze(dim=[axis])
            return x


@register_tensor_op("argmin")
def argmin_op(input, dim: int = None, keepdim: bool = False):
    """The op computes the index with the largest value of a Tensor at specified axis.

    Args:
        input (oneflow.Tensor): Input Tensor
        dim (int, optional): dimension to be calculated. Defaults to the last dim (-1)
        keepdim (bool optional):  whether the output tensor has dim retained or not. Ignored if dim=None.

    Returns:
        oneflow.Tensor: A Tensor(dtype=int64) contains the index with the largest value of `input`

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.tensor([[4, 3, 1, 0, 2],
        ...            [5, 9, 7, 6, 8]], dtype=flow.float32)
        >>> output = flow.argmin(input)
        >>> output
        tensor(3, dtype=oneflow.int64)
        >>> output = flow.argmin(input, dim=1)
        >>> output
        tensor([3, 0], dtype=oneflow.int64)

    """
    return Argmin(dim=dim, keepdim=keepdim)(input)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
