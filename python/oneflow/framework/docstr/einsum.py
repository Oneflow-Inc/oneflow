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
    oneflow.einsum,
    """
    einsum(equation, *operands) -> oneflow.Tensor

    Sums the product of the elements of the input :attr:`operands` along dimensions specified using a notation
    based on the Einstein summation convention.

    Einsum allows computing many common multi-dimensional linear algebraic array operations by representing them
    in a short-hand format based on the Einstein summation convention, given by :attr:`equation`. The details of
    this format are described below, but the general idea is to label every dimension of the input :attr:`operands`
    with some subscript and define which subscripts are part of the output. The output is then computed by summing
    the product of the elements of the :attr:`operands` along the dimensions whose subscripts are not part of the
    output. For example, matrix multiplication can be computed using einsum as `flow.einsum("ij,jk->ik", A, B)`.
    Here, j is the summation subscript and i and k the output subscripts (see section below for more details on why).

    Equation:

        The :attr:`equation` string specifies the subscripts (letters in `[a-zA-Z]`) for each dimension of
        the input :attr:`operands` in the same order as the dimensions, separating subcripts for each operand by a
        comma (','), e.g. `'ij,jk'` specify subscripts for two 2D operands. The dimensions labeled with the same subscript
        must be broadcastable, that is, their size must either match or be `1`. The exception is if a subscript is
        repeated for the same input operand, in which case the dimensions labeled with this subscript for this operand
        must match in size and the operand will be replaced by its diagonal along these dimensions. The subscripts that
        appear exactly once in the :attr:`equation` will be part of the output, sorted in increasing alphabetical order.
        The output is computed by multiplying the input :attr:`operands` element-wise, with their dimensions aligned based
        on the subscripts, and then summing out the dimensions whose subscripts are not part of the output.

        Optionally, the output subscripts can be explicitly defined by adding an arrow ('->') at the end of the equation
        followed by the subscripts for the output. For instance, the following equation computes the transpose of a
        matrix multiplication: 'ij,jk->ki'. The output subscripts must appear at least once for some input operand and
        at most once for the output.

        Ellipsis ('...') can be used in place of subscripts to broadcast the dimensions covered by the ellipsis.
        Each input operand may contain at most one ellipsis which will cover the dimensions not covered by subscripts,
        e.g. for an input operand with 5 dimensions, the ellipsis in the equation `'ab...c'` cover the third and fourth
        dimensions. The ellipsis does not need to cover the same number of dimensions across the :attr:`operands` but the
        'shape' of the ellipsis (the size of the dimensions covered by them) must broadcast together. If the output is not
        explicitly defined with the arrow ('->') notation, the ellipsis will come first in the output (left-most dimensions),
        before the subscript labels that appear exactly once for the input operands. e.g. the following equation implements
        batch matrix multiplication `'...ij,...jk'`.

        A few final notes: the equation may contain whitespaces between the different elements (subscripts, ellipsis,
        arrow and comma) but something like `'. . .'` is not valid. An empty string `''` is valid for scalar operands.

    .. note::

        ``flow.einsum`` handles ellipsis ('...') differently from NumPy in that it allows dimensions
        covered by the ellipsis to be summed over, that is, ellipsis are not required to be part of the output.

    .. note::

        This function does not optimize the given expression, so a different formula for the same computation may
        run faster or consume less memory. Projects like opt_einsum (https://optimized-einsum.readthedocs.io/en/stable/)
        can optimize the formula for you.

    Args:
        equation (String): The subscripts for the Einstein summation.
        operands (*oneflow.Tensor): The tensors to compute the Einstein summation of.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        # trace
        >>> flow.einsum('ii', flow.randn(4, 4))
        tensor(-5.0156, dtype=oneflow.float32)

        # diagonal
        >>> flow.einsum('ii->i', flow.randn(4, 4))
        tensor([-0.2733,  1.8842,  0.0184, -0.5071], dtype=oneflow.float32)

        # outer product
        >>> x = flow.randn(5)
        >>> y = flow.randn(4)
        >>> flow.einsum('i,j->ij', x, y)
        tensor([[ 2.7658,  0.5906, -1.0622, -2.1872],
                [ 1.1117,  0.2374, -0.4270, -0.8791],
                [ 1.5635,  0.3338, -0.6005, -1.2364],
                [-0.6069, -0.1296,  0.2331,  0.4799],
                [ 1.3963,  0.2981, -0.5363, -1.1042]], dtype=oneflow.float32)
        
        # batch matrix multiplication
        >>> As = flow.randn(3,2,5)
        >>> Bs = flow.randn(3,5,4)
        >>> flow.einsum('bij,bjk->bik', As, Bs)
        tensor([[[ 1.0340,  0.7739,  0.8595,  0.4059],
                    [-1.9362, -1.9822, -2.4199, -0.0150]],
                [[-0.3833, -2.5700, -1.2115,  1.1083],
                    [-1.1640, -0.6404,  0.5564, -2.1097]],
                [[-7.8882,  6.7370, -1.3566, -0.3848],
                    [ 0.8045,  1.5177, -0.8427, -0.3266]]], dtype=oneflow.float32)

        # batch permute
        >>> A = flow.randn(2, 3, 4, 5)
        >>> flow.einsum('...ij->...ji', A).shape
        oneflow.Size([2, 3, 5, 4])

        # bilinear
        >>> A = flow.randn(3,5,4)
        >>> l = flow.randn(2,5)
        >>> r = flow.randn(2,4)
        >>> flow.einsum('bn,anm,bm->ba', l, A, r)
        tensor([[-1.0065,  6.0008,  0.3527],
                [-3.8851, -1.9454, -4.9834]], dtype=oneflow.float32)

    """,
)
