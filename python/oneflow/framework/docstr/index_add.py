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
    oneflow.Tensor.index_add_,
    r"""
    index_add_(dim, index, source, *, alpha=1) -> Tensor

    The interface is consistent with PyTorch.    

    Accumulate the elements of :attr:`alpha` times ``source`` into the :attr:`self`
    tensor by adding to the indices in the order given in :attr:`index`. For example,
    if ``dim == 0``, ``index[i] == j``, and ``alpha=-1``, then the ``i``\ th row of
    ``source`` is subtracted from the ``j``\ th row of :attr:`self`.

    The :attr:`dim`\ th dimension of ``source`` must have the same size as the
    length of :attr:`index` (which must be a vector), and all other dimensions must
    match :attr:`self`, or an error will be raised.

    For a 3-D tensor the output is given as::

        self[index[i], :, :] += alpha * src[i, :, :]  # if dim == 0
        self[:, index[i], :] += alpha * src[:, i, :]  # if dim == 1
        self[:, :, index[i]] += alpha * src[:, :, i]  # if dim == 2

    Args:
        dim (int): dimension along which to index
        index (Tensor): indices of ``source`` to select from,
                should have dtype either `oneflow.int64` or `oneflow.int32`
        source (Tensor): the tensor containing values to add

    Keyword args:
        alpha (Number): the scalar multiplier for ``source``

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.ones(5, 3)
        >>> t = flow.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=flow.float)
        >>> index = flow.tensor([0, 4, 2])
        >>> x.index_add_(0, index, t)
        tensor([[ 2.,  3.,  4.],
                [ 1.,  1.,  1.],
                [ 8.,  9., 10.],
                [ 1.,  1.,  1.],
                [ 5.,  6.,  7.]], dtype=oneflow.float32)
        >>> x.index_add_(0, index, t, alpha=-1)
        tensor([[1., 1., 1.],
                [1., 1., 1.],
                [1., 1., 1.],
                [1., 1., 1.],
                [1., 1., 1.]], dtype=oneflow.float32)
    """,
)

add_docstr(
    oneflow._C.index_add,
    r"""
    index_add(input, dim, index, source, *, alpha=1, out=None) -> Tensor

    See :meth:`oneflow.Tensor.index_add_` for function description.
    """,
)

add_docstr(
    oneflow._C.index_add_,
    r"""
    index_add_(dim, index, source, *, alpha=1) -> Tensor

    Out-of-place version of :meth:`oneflow.Tensor.index_add_`.
    """,
)
