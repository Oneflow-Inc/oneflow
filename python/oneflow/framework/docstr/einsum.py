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
        *operands (oneflow.Tensor): The tensors to compute the Einstein summation of.

    For example:

    .. code-block:: python

        >>> import oneflow as flow

        # trace
        >>> flow.einsum('ii', flow.arange(4*4).reshape(4,4).to(flow.float32))
        tensor(30., dtype=oneflow.float32)

        # diagonal
        >>> flow.einsum('ii->i', flow.arange(4*4).reshape(4,4).to(flow.float32))
        tensor([ 0.,  5., 10., 15.], dtype=oneflow.float32)

        # outer product
        >>> x = flow.arange(5).to(flow.float32)
        >>> y = flow.arange(4).to(flow.float32)
        >>> flow.einsum('i,j->ij', x, y)
        tensor([[ 0.,  0.,  0.,  0.],
                [ 0.,  1.,  2.,  3.],
                [ 0.,  2.,  4.,  6.],
                [ 0.,  3.,  6.,  9.],
                [ 0.,  4.,  8., 12.]], dtype=oneflow.float32)
        
        # batch matrix multiplication
        >>> As = flow.arange(3*2*5).reshape(3,2,5).to(flow.float32)
        >>> Bs = flow.arange(3*5*4).reshape(3,5,4).to(flow.float32)
        >>> flow.einsum('bij,bjk->bik', As, Bs).shape
        oneflow.Size([3, 2, 4])

        # batch permute
        >>> A = flow.randn(2, 3, 4, 5)
        >>> flow.einsum('...ij->...ji', A).shape
        oneflow.Size([2, 3, 5, 4])

        # bilinear
        >>> A = flow.randn(3,5,4)
        >>> l = flow.randn(2,5)
        >>> r = flow.randn(2,4)
        >>> flow.einsum('bn,anm,bm->ba', l, A, r).shape
        oneflow.Size([2, 3])

    """,
)
