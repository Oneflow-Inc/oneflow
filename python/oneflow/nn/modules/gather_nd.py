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
from oneflow.framework.tensor import Tensor


def gather_nd_op(input, index):
    """This operator is a high-dimensional extension of `gather`, `index` is a K-dimensional
    tensor, which is regarded as a index of input Tensor `input`.

    Each element defines a slice of `input`:

    .. math::

        output[i_{0},i_{1},...,i_{K-2}] = input[index(i_{0},i_{1},...,i_{K-2})]


    Args:
        input: The input Tensor.
        index: The slice indices.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> input = flow.Tensor(np.array([[1, 2,3], [4, 5,6],[7,8,9]]), dtype=flow.float)
        >>> index_1 = flow.Tensor(np.array([[0], [2]]), dtype=flow.int)
        >>> out_1 = flow.gather_nd(input,index_1)
        >>> print(out_1.shape)
        flow.Size([2, 3])
        >>> out_1
        tensor([[1., 2., 3.],
                [7., 8., 9.]], dtype=oneflow.float32)
        >>> index_2 = flow.Tensor(np.array([[0,2], [2,1]]), dtype=flow.int)
        >>> out_2 = flow.gather_nd(input,index_2)
        >>> out_2
        tensor([3., 8.], dtype=oneflow.float32)

    """
    return flow.F.gather_nd(input, index)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
