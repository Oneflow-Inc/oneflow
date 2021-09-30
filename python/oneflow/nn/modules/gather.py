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


@register_tensor_op("gather")
def gather_op(input, dim, index, sparse_grad=False):
    """Gathers values along an axis specified by `dim`.

    For a 3-D tensor the output is specified by::

        out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

    :attr:`input` and :attr:`index` must have the same number of dimensions.
    It is also required that ``index.size(d) <= input.size(d)`` for all
    dimensions ``d != dim``.  :attr:`out` will have the same shape as :attr:`index`.
    Note that ``input`` and ``index`` do not broadcast against each other.

    Args:
        input (Tensor): the source tensor
        dim (int): the axis along which to index
        index (LongTensor): the indices of elements to gather

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> input = np.random.randn(3, 4, 3, 5)
        >>> index = np.random.choice(np.arange(3), size=180, replace=True).reshape((3, 4, 3, 5))
        >>> output = flow.gather(flow.Tensor(input), 1, flow.tensor(index, dtype=flow.int))
        >>> output.shape
        oneflow.Size([3, 4, 3, 5])

    """

    assert sparse_grad is False, "Only support bool = False for now!"
    assert dim < len(
        index.shape
    ), "Value of dim is out of range(dim should be less than len(index.shape))"
    assert len(input.shape) == len(
        index.shape
    ), "dimensions of input and index should equal"
    for i in range(0, len(input.shape)):
        if i != dim:
            assert (
                index.shape[i] <= input.shape[i]
            ), "index.size(d) <= input.size(d) for all dimensions d != dim"
    return flow._C.dim_gather(input, index, dim=dim)


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
        >>> input = flow.tensor(np.array([[1, 2,3], [4, 5,6],[7,8,9]]), dtype=flow.float)
        >>> index_1 = flow.tensor(np.array([[0], [2]]), dtype=flow.int)
        >>> out_1 = flow.gather_nd(input,index_1)
        >>> print(out_1.shape)
        oneflow.Size([2, 3])
        >>> out_1
        tensor([[1., 2., 3.],
                [7., 8., 9.]], dtype=oneflow.float32)
        >>> index_2 = flow.tensor(np.array([[0,2], [2,1]]), dtype=flow.int)
        >>> out_2 = flow.gather_nd(input,index_2)
        >>> out_2
        tensor([3., 8.], dtype=oneflow.float32)

    """
    return flow._C.gather_nd(input, index)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
