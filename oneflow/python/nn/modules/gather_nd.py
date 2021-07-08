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

from oneflow.python.framework.tensor import Tensor
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.framework.tensor import register_tensor_op
from oneflow.python.nn.module import Module

from typing import Optional, List, Tuple


class Gather_nd(Module):
    def __init__(self) -> None:
        super().__init__()
        self.gather_nd_op = (
            flow.builtin_op("gather_nd").Input("params").Input("indices").Output("out").Build()
        )

    def forward(self, input, indices):

        return self.gather_nd_op(input,indices)[0]


@oneflow_export("gather_nd")
@experimental_api
def gather_nd_op(input, indices):
    r"""This operator is a high-dimensional extension of `gather`, `indices` is a K-dimensional
    tensor, which is regarded as a index of input Blob `input`.

    Each element defines a slice of `input`:

    .. math::

        output[(i_0,i_1,...,i_{K-2})] = input[indices(i_{0},i_{1},...,i_{K-2})]


    Args:
        input: The input Blob.
        indices: The slice indices.

    For example:

    .. code-block:: python

        >>> import oneflow.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()

        >>> input = flow.Tensor(np.array([[1, 2,3], [4, 5,6],[7,8,9]]), dtype=flow.float)
        >>> indices_1 = flow.Tensor(np.array([[0], [2]]), dtype=flow.int)
        >>> out_1 = flow.gather_nd(input,indices_1)
        >>> print(out_1.shape)
        flow.Size([2, 3])
        >>> print(out_1)
        tensor([[1., 2., 3.],
                [7., 8., 9.]], dtype=oneflow.float32)
        >>> indices_2 = flow.Tensor(np.array([[0,2], [2,1]]), dtype=flow.int)
        >>> out_2 = flow.gather_nd(input,indices_2)
        >>> print(out_2)
        tensor([3., 8.], dtype=oneflow.float32)

    """
    return Gather_nd()(input,indices)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
