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
import oneflow
from oneflow.framework.docstr.utils import add_docstr

add_docstr(
    oneflow.nonzero,
    """nonzero(input, *, out=None, as_tuple=False) -> Tensor or tuple of Tensors

    .. note::
        When :attr:`as_tuple` is ``False`` (default):  returns a
        2-D tensor where each row is the index for a nonzero value.

        When :attr:`as_tuple` is ``True``: returns a tuple of 1-D
        index tensors, allowing for advanced indexing, so ``x[x.nonzero(as_tuple=True)]``
        gives all nonzero values of tensor ``x``. Of the returned tuple, each index tensor
        contains nonzero indices for a certain dimension.

        See below for more details on the two behaviors.

    **When** :attr:`as_tuple` **is** ``False`` **(default)**:

    Returns a tensor containing the indices of all non-zero elements of
    :attr:`input`.  Each row in the result contains the indices of a non-zero
    element in :attr:`input`. The result is sorted lexicographically, with
    the last index changing the fastest (C-style).

    If :attr:`input` has :math:`n` dimensions, then the resulting indices tensor
    :attr:`out` is of size :math:`(z \\times n)`, where :math:`z` is the total number of
    non-zero elements in the :attr:`input` tensor.

    **When** :attr:`as_tuple` **is** ``True``:

    Returns a tuple of 1-D tensors, one for each dimension in :attr:`input`,
    each containing the indices (in that dimension) of all non-zero elements of
    :attr:`input` .

    If :attr:`input` has :math:`n` dimensions, then the resulting tuple contains :math:`n`
    tensors of size :math:`z`, where :math:`z` is the total number of
    non-zero elements in the :attr:`input` tensor.

    As a special case, when :attr:`input` has zero dimensions and a nonzero scalar
    value, it is treated as a one-dimensional tensor with one element.

    Args:
        input(Tensor): the input tensor.

    Keyword args:
        out (Tensor, optional): the output tensor containing indices

    Returns:
        Tensor or tuple of Tensors: If :attr:`as_tuple` is ``False``, the output
        tensor containing indices. If :attr:`as_tuple` is ``True``, one 1-D tensor for
        each dimension, containing the indices of each nonzero element along that
        dimension.

    Example::

        >>> import oneflow as flow
        >>> flow.nonzero(flow.tensor([1, 1, 1, 0, 1]))
        tensor([[0],
                [1],
                [2],
                [4]], dtype=oneflow.int64)
        >>> flow.nonzero(flow.tensor([[0.6, 0.0, 0.0, 0.0],
        ...                             [0.0, 0.4, 0.0, 0.0],
        ...                             [0.0, 0.0, 1.2, 0.0],
        ...                             [0.0, 0.0, 0.0,-0.4]]))
        tensor([[0, 0],
                [1, 1],
                [2, 2],
                [3, 3]], dtype=oneflow.int64)
        >>> flow.nonzero(flow.tensor([1, 1, 1, 0, 1]), as_tuple=True)
        (tensor([0, 1, 2, 4], dtype=oneflow.int64),)
        >>> flow.nonzero(flow.tensor([[0.6, 0.0, 0.0, 0.0],
        ...                             [0.0, 0.4, 0.0, 0.0],
        ...                             [0.0, 0.0, 1.2, 0.0],
        ...                             [0.0, 0.0, 0.0,-0.4]]), as_tuple=True)
        (tensor([0, 1, 2, 3], dtype=oneflow.int64), tensor([0, 1, 2, 3], dtype=oneflow.int64))
        >>> flow.nonzero(flow.tensor(5), as_tuple=True)
        (tensor([0], dtype=oneflow.int64),)

    """,
)
