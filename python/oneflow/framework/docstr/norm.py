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
    oneflow._C.vector_norm,
    """
    linalg.vector_norm(input, ord=2, dim=None, keepdim=False, *, dtype=None, out=None) -> Tensor

    Computes a vector norm.

    Supports input of float, double dtypes.

    This function does not necessarily treat multidimensonal attr:`input` as a batch of
    vectors, instead:

    - If :attr:`dim`\\ `= None`, :attr:`input` will be flattened before the norm is computed.
    - If :attr:`dim` is an `int` or a `tuple`, the norm will be computed over these dimensions and the other dimensions will be treated as batch dimensions.

    This behavior is for consistency with :func:`flow.linalg.norm`.

    :attr:`ord` defines the vector norm that is computed. The following norms are supported:

    ======================   ========================================================
    :attr:`ord`              vector norm
    ======================   ========================================================
    `2` (default)            `2`-norm (see below)
    `inf`                    `max(abs(x))`
    `-inf`                   `min(abs(x))`
    `0`                      `sum(x != 0)`
    other `int` or `float`   `sum(abs(x)^{ord})^{(1 / ord)}`
    ======================   ========================================================

    where `inf` refers to `float('inf')`, NumPy's `inf` object, or any equivalent object.


    Args:
        input (Tensor): tensor, flattened by default, but this behavior can be
            controlled using :attr:`dim`.
        ord (int, float, inf, -inf, 'fro', 'nuc', optional): order of norm. Default: `2`
        dim (int, Tuple[int], optional): dimensions over which to compute
            the norm. See above for the behavior when :attr:`dim`\\ `= None`.
            Default: `None`
        keepdim (bool, optional): If set to `True`, the reduced dimensions are retained
            in the result as dimensions with size one. Default: `False`

    Returns:
        A real-valued tensor.

    Examples::

        >>> import oneflow as flow
        >>> import numpy as np
        >>> a = flow.tensor(np.arange(9, dtype=np.float32) - 4)
        >>> a
        tensor([-4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.], dtype=oneflow.float32)
        >>> b = a.reshape(3, 3)
        >>> b
        tensor([[-4., -3., -2.],
                [-1.,  0.,  1.],
                [ 2.,  3.,  4.]], dtype=oneflow.float32)
        >>> flow._C.vector_norm(a, ord=3.5)
        tensor(5.4345, dtype=oneflow.float32)
        >>> flow._C.vector_norm(b, ord=3.5)
        tensor(5.4345, dtype=oneflow.float32)
    """,
)

add_docstr(
    oneflow._C.matrix_norm,
    """
    linalg.matrix_norm(input, ord='fro', dim=(-2, -1), keepdim=False, *, dtype=None, out=None) -> Tensor

    Computes a matrix norm.

    Support input of float, double, cfloat and cdouble dtypes.
    Also supports batches of matrices: the norm will be computed over the
    dimensions specified by the 2-tuple :attr:`dim` and the other dimensions will
    be treated as batch dimensions. The output will have the same batch dimensions.

    :attr:`ord` defines the matrix norm that is computed. The following norms are supported:

    ======================   ========================================================
    :attr:`ord`              matrix norm
    ======================   ========================================================
    `'fro'` (default)        Frobenius norm
    `'nuc'`                  -- not supported yet --
    `inf`                    `max(sum(abs(x), dim=1))`
    `-inf`                   `min(sum(abs(x), dim=1))`
    `1`                      `max(sum(abs(x), dim=0))`
    `-1`                     `min(sum(abs(x), dim=0))`
    `2`                      -- not supported yet --
    `-2`                     -- not supported yet --
    ======================   ========================================================

    where `inf` refers to `float('inf')`, NumPy's `inf` object, or any equivalent object.

    Args:
        input (Tensor): tensor with two or more dimensions. By default its
            shape is interpreted as `(*, m, n)` where `*` is zero or more
            batch dimensions, but this behavior can be controlled using :attr:`dim`.
        ord (int, inf, -inf, 'fro', 'nuc', optional): order of norm. Default: `'fro'`
        dim (Tuple[int, int], optional): dimensions over which to compute the norm. Default: `(-2, -1)`
        keepdim (bool, optional): If set to `True`, the reduced dimensions are retained
            in the result as dimensions with size one. Default: `False`


    Returns:
        A real-valued tensor.

    Examples::

        >>> import oneflow as flow
        >>> import numpy as np
        >>> a = flow.tensor(np.arange(9, dtype=np.float32)).reshape(3,3)
        >>> a
        tensor([[0., 1., 2.],
                [3., 4., 5.],
                [6., 7., 8.]], dtype=oneflow.float32)
        >>> flow._C.matrix_norm(a)
        tensor(14.2829, dtype=oneflow.float32)
        >>> flow._C.matrix_norm(a, ord=-1)
        tensor(9., dtype=oneflow.float32)
        >>> b = a.expand(2, -1, -1)
        >>> b
        tensor([[[0., 1., 2.],
                 [3., 4., 5.],
                 [6., 7., 8.]],
        <BLANKLINE>
                [[0., 1., 2.],
                 [3., 4., 5.],
                 [6., 7., 8.]]], dtype=oneflow.float32)
        >>> flow._C.matrix_norm(b)
        tensor([14.2829, 14.2829], dtype=oneflow.float32)
        >>> flow._C.matrix_norm(b, dim=(0, 2))
        tensor([ 3.1623, 10.0000, 17.2627], dtype=oneflow.float32)
    """,
)

