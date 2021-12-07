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
import collections
from typing import Optional, Sequence, Union

import oneflow as flow
from oneflow.framework.tensor import register_tensor_op
from oneflow.nn.module import Module
from oneflow.nn.modules.utils import _check_axis
from oneflow.ops.transpose_util import (
    get_inversed_perm,
    get_perm_when_transpose_axis_to_last_dim,
)


@register_tensor_op("sub")
def _sub(input, other):
    """Computes the subtraction of input by other for each element, scalar and broadcast promotation are supported.
    The formula is:

    .. math::
        out = input - other
    
    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        # element-wise subtract
        >>> input = flow.Tensor(np.random.randn(2,3))
        >>> other = flow.Tensor(np.random.randn(2,3))
        >>> out = flow.sub(input,other).numpy()
        >>> out.shape
        (2, 3)

        # scalar subtract
        >>> input = 5
        >>> other = flow.Tensor(np.random.randn(2,3))
        >>> out = flow.sub(input,other).numpy()
        >>> out.shape
        (2, 3)

        # broadcast subtract
        >>> input = flow.Tensor(np.random.randn(1,1))
        >>> other = flow.Tensor(np.random.randn(2,3))
        >>> out = flow.sub(input,other).numpy()
        >>> out.shape
        (2, 3)

    """
    return flow._C.sub(input, other)


@register_tensor_op("div")
def _div(input, other):
    """Computes the division of input by other for each element, scalar and broadcast promotation are supported.
    The formula is:

    .. math::
        out = \\frac{input}{other}
    
    Args:
        input (Union[int, float, flow.Tensor]): input.
        other (Union[int, float, flow.Tensor]): other.
    
    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        # element-wise divide
        >>> input = flow.Tensor(np.random.randn(2,3))
        >>> other = flow.Tensor(np.random.randn(2,3))
        >>> out = flow.div(input,other).numpy()
        >>> out.shape
        (2, 3)

        # scalar divide
        >>> input = 5
        >>> other = flow.Tensor(np.random.randn(2,3))
        >>> out = flow.div(input,other).numpy()
        >>> out.shape
        (2, 3)

        # broadcast divide
        >>> input = flow.Tensor(np.random.randn(1,1))
        >>> other = flow.Tensor(np.random.randn(2,3))
        >>> out = flow.div(input,other).numpy()
        >>> out.shape 
        (2, 3)

    """
    return flow._C.div(input, other)


@register_tensor_op("add")
def _add(input, other):
    """Computes the addition of `input` by `other` for each element, scalar and broadcast promotation are supported.
    The formula is:

    .. math::
        out = input + other

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        # element-wise add
        >>> x = flow.Tensor(np.random.randn(2,3))
        >>> y = flow.Tensor(np.random.randn(2,3))
        >>> out = flow.add(x, y).numpy()
        >>> out.shape
        (2, 3)

        # scalar add
        >>> x = 5
        >>> y = flow.Tensor(np.random.randn(2,3))
        >>> out = flow.add(x, y).numpy()
        >>> out.shape
        (2, 3)

        # broadcast add
        >>> x = flow.Tensor(np.random.randn(1,1))
        >>> y = flow.Tensor(np.random.randn(2,3))
        >>> out = flow.add(x, y).numpy()
        >>> out.shape
        (2, 3)

    """
    return flow._C.add(input, other)


@register_tensor_op("add_")
def _add_inplace(x, y):
    """
    In-place version of :func:`oneflow.Tensor.add`.
    """
    return flow._C.add(x, y, inplace=True)


def fmod_op(input, other):
    """
    fmod(input, other, *, out=None) -> Tensor

    Computes the element-wise remainder of division.

    The dividend and divisor may contain both for integer and floating point
    numbers. The remainder has the same sign as the dividend :attr:`input`.

    Supports broadcasting to a common shape, integer and float inputs.


    Args:
        input (Tensor): the dividend
        other (Tensor or Scalar): the divisor

    Keyword args:
        out (Tensor, optional): the output tensor.

    Example::

        >>> import oneflow as flow
        >>> flow.fmod(flow.tensor([-3., -2, -1, 1, 2, 3]), 2.)
        tensor([-1., -0., -1.,  1.,  0.,  1.], dtype=oneflow.float32)
        >>> flow.fmod(flow.tensor([1, 2, 3, 4, 5.]), 1.5)
        tensor([1.0000, 0.5000, 0.0000, 1.0000, 0.5000], dtype=oneflow.float32)
        >>> flow.fmod(flow.tensor([1, 2, 3, 4., -5]), flow.tensor([4, 2, 1, 3., 1]))
        tensor([1., 0., 0., 1., -0.], dtype=oneflow.float32)

    """
    return flow._C.fmod(input, other)


@register_tensor_op("fmod")
def fmod_op_tensor(input, other):
    """

    See :func:`oneflow.fmod`
    
    """
    return fmod_op(input, other)


def addmm(x, mat1, mat2, alpha=1, beta=1):
    if len(x.shape) > 2 or len(mat1.shape) > 2 or len(mat2.shape) > 2:
        raise ValueError("input matrixes shape can not be greater than 2")
    else:
        return flow.mul(x, beta) + flow.mul(flow._C.matmul(mat1, mat2), alpha)


def addmm_op(input, mat1, mat2, alpha=1, beta=1):
    """addmm(beta=1, input, alpha=1, mat1, mat2, out=None) -> Tensor

    Performs a matrix multiplication of the matrices :attr:`mat1` and :attr:`mat2`.
    The matrix :attr:`input` is added to the final result.

    If :attr:`mat1` is a :math:`(n \\times m)` tensor, :attr:`mat2` is a
    :math:`(m \\times p)` tensor, then :attr:`input` must be
    broadcastable with a :math:`(n \\times p)` tensor
    and :attr:`out` will be a :math:`(n \\times p)` tensor.

    :attr:`alpha` and :attr:`beta` are scaling factors on matrix-vector product between
    :attr:`mat1` and :attr:`mat2` and the added matrix :attr:`input` respectively.

    .. math::
        \\text{out} = \\beta\\ \\text{input} + \\alpha\\ (\\text{mat1}_i \\mathbin{@} \\text{mat2}_i)

    For inputs of type `float` or `double`, arguments :attr:`beta` and
    :attr:`alpha` must be real numbers, otherwise they should be integers.

    Args:
        beta (Number, optional): multiplier for :attr:`input` (:math:`\\beta`)
        input (Tensor): matrix to be added
        alpha (Number, optional): multiplier for :math:`mat1 @ mat2` (:math:`\\alpha`)
        mat1 (Tensor): the first matrix to be multiplied
        mat2 (Tensor): the second matrix to be multiplied
        out (Tensor, optional): the output tensor.

    For example:

        >>> import numpy as np
        >>> import oneflow as flow
        >>> input = flow.tensor(np.array([[1,2,4],[5,11,9.1]]))
        >>> mat1 = flow.tensor(np.array([[7.3,1.9,7.3],[10.2,1,5.5]])) 
        >>> mat2 = flow.tensor(np.array([[7.3,1.9,7.3],[10.2,1,5.5],[3.7,2.2,8.1]])) 
        >>> output = flow.addmm(input, mat1, mat2)
        >>> output
        tensor([[100.6800,  33.8300, 126.8700],
                [110.0100,  43.4800, 133.6100]], dtype=oneflow.float64)
        >>> output.shape
        oneflow.Size([2, 3])

        >>> input2 = flow.tensor(np.array([1.7]))
        >>> mat1 = flow.tensor(np.array([[1,2],[5,9.1],[7.7,1.4]]))
        >>> mat2 = flow.tensor(np.array([[1,2,3.7],[5,9.1,6.8]]))
        >>> output2 = flow.addmm(input2, mat1, mat2, alpha=1, beta=2)
        >>> output2
        tensor([[14.4000, 23.6000, 20.7000],
                [53.9000, 96.2100, 83.7800],
                [18.1000, 31.5400, 41.4100]], dtype=oneflow.float64)
        >>> output2.shape
        oneflow.Size([3, 3])
    """
    return addmm(input, mat1, mat2, alpha, beta)


@register_tensor_op("addmm")
def addmm_op_tensor(input, mat1, mat2, alpha=1, beta=1):
    """
    See :func:`oneflow.addmm`
    """
    return addmm(input, mat1, mat2, alpha, beta)


class Clamp(Module):
    def __init__(self, min_value=None, max_value=None) -> None:
        super().__init__()
        if min_value is not None:
            floating_min_value = float(min_value)
            integral_min_value = int(min_value)
        if max_value is not None:
            floating_max_value = float(max_value)
            integral_max_value = int(max_value)
        if min_value is not None and max_value is not None:
            self._op = (
                flow.builtin_op("clip_by_scalar")
                .Input("x")
                .Output("y")
                .Attr("floating_min", floating_min_value)
                .Attr("integral_min", integral_min_value)
                .Attr("floating_max", floating_max_value)
                .Attr("integral_max", integral_max_value)
                .Build()
            )
        elif min_value is not None:
            self._op = (
                flow.builtin_op("clip_by_scalar_min")
                .Input("x")
                .Output("y")
                .Attr("floating_min", floating_min_value)
                .Attr("integral_min", integral_min_value)
                .Build()
            )
        elif max_value is not None:
            self._op = (
                flow.builtin_op("clip_by_scalar_max")
                .Input("x")
                .Output("y")
                .Attr("floating_max", floating_max_value)
                .Attr("integral_max", integral_max_value)
                .Build()
            )
        else:
            raise ValueError("min_value and max_value cannot be None at the same time")

    def forward(self, x):
        return self._op(x)[0]


def clamp_op(input, min=None, max=None):
    """
    Clamp all elements in :attr:`input` into the range `[` :attr:`min`, :attr:`max` `]` and return
    a resulting tensor:

    .. math::
        y_i = \\begin{cases}
            \\text{min} & \\text{if } x_i < \\text{min} \\\\
            x_i & \\text{if } \\text{min} \\leq x_i \\leq \\text{max} \\\\
            \\text{max} & \\text{if } x_i > \\text{max}
        \\end{cases}

    If :attr:`input` is of type `FloatTensor` or `DoubleTensor`, args :attr:`min`
    and :attr:`max` must be real numbers, otherwise they should be integers.

    Args:
        input (Tensor): the input tensor.
        min (Number): lower-bound of the range to be clamped to. Defaults to None.
        max (Number): upper-bound of the range to be clamped to. Defaults to None.
        out (Tensor, optional): the output tensor.

    For example:


    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> arr = np.array([0.2, 0.6, -1.5, -0.3])
        >>> input = flow.Tensor(arr)
        >>> output = flow.clamp(input, min=-0.5, max=0.5)
        >>> output
        tensor([ 0.2000,  0.5000, -0.5000, -0.3000], dtype=oneflow.float32)

        >>> arr = np.array([0.2, 0.6, -1.5, -0.3])
        >>> input = flow.Tensor(arr)
        >>> output = flow.clamp(input, min=None, max=0.5)
        >>> output
        tensor([ 0.2000,  0.5000, -1.5000, -0.3000], dtype=oneflow.float32)

        >>> arr = np.array([0.2, 0.6, -1.5, -0.3])
        >>> input = flow.Tensor(arr)
        >>> output = flow.clamp(input, min=-0.5, max=None)
        >>> output
        tensor([ 0.2000,  0.6000, -0.5000, -0.3000], dtype=oneflow.float32)

    """
    return flow._C.clamp(input, min, max)


@register_tensor_op("clamp")
def clamp_op_tensor(tensor, min=None, max=None):
    """
    See :func:`oneflow.clamp`
    """
    return Clamp(min, max)(tensor)


def clip_op(tensor, min=None, max=None):
    """
    Alias for :func:`oneflow.clamp`
    """
    return Clamp(min, max)(tensor)


@register_tensor_op("clip")
def clip_op_tensor(tensor, min=None, max=None):
    """
    See :func:`oneflow.clamp`
    """
    return Clamp(min, max)(tensor)


class Topk(Module):
    def __init__(
        self, k, dim: int = None, largest: bool = True, sorted: bool = True
    ) -> None:
        super().__init__()
        self._op_topk_last_dim = (
            flow.builtin_op("top_k")
            .Input("in")
            .Output("out")
            .Attr("k", k)
            .Attr("sorted", sorted)
            .Build()
        )
        self.dim = dim
        self.largest = largest

    def forward(self, input):
        if self.dim == None:
            self.dim = -1
        num_axes = len(input.shape)
        axis = self.dim if self.dim >= 0 else self.dim + num_axes
        assert 0 <= axis < num_axes, "axis out of range"
        if axis == num_axes - 1:
            if self.largest:
                indices = self._op_topk_last_dim(input)[0]
            else:
                neg_input = flow.mul(input, -1)
                indices = self._op_topk_last_dim(neg_input)[0]
            return (flow.gather(input, axis, indices), indices)
        else:
            perm = get_perm_when_transpose_axis_to_last_dim(num_axes, axis)
            x = flow._C.transpose(input, perm=perm)
            if self.largest:
                indices = self._op_topk_last_dim(x)[0]
            else:
                neg_input = flow.mul(x, -1)
                indices = self._op_topk_last_dim(neg_input)[0]
            indices = flow._C.transpose(indices, perm=get_inversed_perm(perm))
            return (flow.gather(input, axis, indices), indices)


@register_tensor_op("topk")
def topk_op(input, k, dim: int = None, largest: bool = True, sorted: bool = True):
    """Finds the values and indices of the k largest entries at specified axis.

    Args:
        input (oneflow.Tensor): Input Tensor
        k (int): the k in “top-k”
        dim (int, optional): the dimension to sort along. Defaults to the last dim (-1)
        largest (bool, optional): controls whether to return largest or smallest elements
        sorted (bool, optional): controls whether to return the elements in sorted order (Only Support True Now!)

    Returns:
        Tuple(oneflow.Tensor, oneflow.Tensor(dtype=int32)): A tuple of (values, indices), where
        the indices are the indices of the elements in the original input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x = np.array([[1, 3, 8, 7, 2], [1, 9, 4, 3, 2]], dtype=np.float32)
        >>> (values, indices) = flow.topk(flow.Tensor(x), k=3, dim=1)
        >>> values
        tensor([[8., 7., 3.],
                [9., 4., 3.]], dtype=oneflow.float32)
        >>> indices
        tensor([[2, 3, 1],
                [1, 2, 3]], dtype=oneflow.int32)
        >>> values.shape
        oneflow.Size([2, 3])
        >>> indices.shape
        oneflow.Size([2, 3])
        >>> (values, indices) = flow.topk(flow.Tensor(x), k=2, dim=1, largest=False)
        >>> values
        tensor([[1., 2.],
                [1., 2.]], dtype=oneflow.float32)
        >>> indices
        tensor([[0, 4],
                [0, 4]], dtype=oneflow.int32)
        >>> values.shape
        oneflow.Size([2, 2])
        >>> indices.shape
        oneflow.Size([2, 2])

    """
    return Topk(k=k, dim=dim, largest=largest, sorted=sorted)(input)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
