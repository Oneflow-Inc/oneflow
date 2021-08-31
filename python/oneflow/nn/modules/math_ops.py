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


@register_tensor_op("var")
def variance_op(input, dim=None, unbiased=True, keepdim=False):
    """Returns the variance of each row of the `input` tensor in the given dimension `dim`.

    If `keepdim` is `True`, the output tensor is of the same size as `input` except in the dimension(s) `dim` 
    where it is of size 1. Otherwise, dim is squeezed (see `flow.squeeze()`), resulting in the output 
    tensor having 1 (or `len(dim)`) fewer dimension(s).

    Args:
        input (Tensor): the input tensor.
        dim (int or tuple of python:ints): the dimension or dimensions to reduce. Defaults to None.
        unbiased (bool, optional): whether to use Bessel’s correction (:math:`\delta N = 1`). Defaults to True.
        keepdim (bool, optional): whether the output tensor has dim retained or not. Defaults to False.

    Returns:
        Tensor: The result of variance on the specified axis of input Tensor

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> np_arr = np.random.randn(2,3,4,5)
        >>> input = flow.Tensor(np_arr)
        >>> output = flow.var(input, 1, True)

    """
    input_shape = input.shape
    axis = _check_axis(dim, input_shape)
    input_shape_dim = 1
    for x in axis:
        input_shape_dim *= input_shape[x]
    if unbiased:
        input_shape_dim -= 1
    res = flow.sum(
        flow.square(input - flow.mean(input, dim=axis, keepdim=True)),
        dim=axis,
        keepdim=keepdim,
    )
    return res / input_shape_dim


def sin_op(input):
    """
    Returns a new tensor with the sine of the elements of :attr:`input`.

    .. math::

        \\text{out}_{i} = \\sin(\\text{input}_{i})

    Args:
        input (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x1 = flow.Tensor(np.array([-0.5461,  0.1347, -2.7266, -0.2746]).astype(np.float32))
        >>> out1 = flow.sin(x1)
        >>> out1
        tensor([-0.5194,  0.1343, -0.4032, -0.2712], dtype=oneflow.float32)
        >>> x2 = flow.Tensor(np.array([-1.4, 2.6, 3.7]).astype(np.float32),device=flow.device('cuda'))
        >>> out2 = flow.sin(x2)
        >>> out2
        tensor([-0.9854,  0.5155, -0.5298], device='cuda:0', dtype=oneflow.float32)

    """
    return flow._C.sin(input, False)


@register_tensor_op("sin")
def sin_op_tensor(input):
    """

    sin() -> Tensor

    See :func:`oneflow.sin`
    
    """
    return flow._C.sin(input, False)


@register_tensor_op("sin_")
def inplace_sin_op_tensor(input):
    """
    In-place version of :func:`oneflow.sin`
    
    """
    return flow._C.sin(input, True)


@register_tensor_op("rsqrt")
def rsqrt_op(input):
    """Returns a new tensor with the reciprocal of the square-root of each of
        the elements of :attr:`input`.

        .. math::
            \\text{out}_{i} = \\frac{1}{\\sqrt{\\text{input}_{i}}}

        Args:
            input (Tensor): the input tensor.

         For example:

        .. code-block:: python

            >>> import oneflow as flow
            >>> import numpy as np
            
            >>> a = flow.Tensor(np.array([1.0, 2.0, 3.0]))
            >>> out = flow.rsqrt(a).numpy()
            >>> out
            array([1.        , 0.70710677, 0.57735026], dtype=float32)
    """
    return flow._C.rsqrt(input)


@register_tensor_op("sqrt")
def sqrt_op(input):
    """Returns a new tensor with the square-root of the elements of :attr:`input`.

        .. math::
            \\text{out}_{i} = \\sqrt{\\text{input}_{i}}

        Args:
            input (Tensor): the input tensor.

         For example:

        .. code-block:: python

            >>> import oneflow as flow
            >>> import numpy as np
            
            >>> arr = np.array([1.0, 2.0, 3.0])
            >>> input = flow.Tensor(arr)
            >>> output = flow.sqrt(input).numpy()
            >>> output
            array([1.       , 1.4142135, 1.7320508], dtype=float32)
        """
    return flow._C.sqrt(input)


@register_tensor_op("square")
def square_op(input):
    """Returns a new tensor with the square of the elements of :attr:`input`.

        .. math::
            \\text{out}_{i} = \\sqrt{\\text{input}_{i}}

        Args:
            input (Tensor): the input tensor.

         For example:

        .. code-block:: python

            >>> import oneflow as flow
            >>> import numpy as np
            
            >>> arr = np.array([1.0, 2.0, 3.0])
            >>> input = flow.Tensor(arr)
            >>> output = flow.square(input).numpy()
            >>> output
            array([1., 4., 9.], dtype=float32)
        """
    return flow._C.square(input)


@register_tensor_op("std")
def std_op(input, dim, unbiased=False, keepdim=False):
    """
    Returns the standard-deviation of each row of the :attr:`input` tensor in the
    dimension :attr:`dim`. If :attr:`dim` is a list of dimensions,
    reduce over all of them.

    If keepdim is True, the output tensor is of the same size as input except in 
    the dimension(s) dim where it is of size 1. Otherwise, dim is squeezed, 
    resulting in the output tensor having 1 (or len(dim)) fewer dimension(s).

    If :attr:`unbiased` is ``False``, then the standard-deviation will be calculated
    via the biased estimator. Otherwise, Bessel's correction will be used.

    Args:
        input (Tensor): the input tensor.
        dim (int or tuple of python:ints): the dimension or dimensions to reduce.
        unbiased (bool): whether to use the unbiased estimation or not
        keepdim (bool): whether the output tensor has `dim` retained or not.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        
        >>> arr = np.array([1.0, 2.0, 3.0])
        >>> input = flow.Tensor(arr)
        >>> output = flow.std(input, dim=0).numpy()
        >>> output
        array(0.8164968, dtype=float32)

    """

    assert unbiased == False, "Only support 'unbiased=False' for now!"
    reduce_count = 1

    axis = _check_axis(dim, input.shape)
    if isinstance(axis, list) and len(axis) == 0:
        return flow.zeros(input.shape)
    else:
        if len(axis) == 0:
            reduce_count = input.nelement()
        else:
            for i in axis:
                reduce_count *= input.shape[i]
        sum = flow.sum(flow._C.square(input), axis, keepdim) / reduce_count
        square = flow._C.square(flow.sum(input, axis, keepdim) / reduce_count)
        subtract = flow._C.sub(sum, square)
        res = flow._C.sqrt(subtract)
        return res


@register_tensor_op("pow")
def pow_op(input, exponent):
    """Takes the power of each element in input with exponent and returns a tensor with the result. Exponent can be either a single float number, a single int number, or a tensor with the same shape as input.
    When exponent is a scalar value, the operation applied is:

    .. math::
        \\text{out}_i = x_i ^ \\text{exponent}
\u200b
    When exponent is a tensor, the operation applied is:

    .. math::
        \\text{out}_i = x_i ^ {\\text{exponent}_i}

    Args:
        - input (Tensor): the input tensor.
        - exponent (int, float, Tensor): the exponent.

    Returns:
        Tensor: The result of variance on the specified axis of input Tensor

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        
        >>> x = flow.Tensor(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        >>> out = flow.pow(x, 2)
        >>> out
        tensor([ 1.,  4.,  9., 16., 25., 36.], dtype=oneflow.float32)

        >>> x = flow.Tensor(np.array([1.0, 2.0, 3.0, 4.0]))
        >>> y = flow.Tensor(np.array([1.0, 2.0, 3.0, 4.0]))
        >>> out = flow.pow(x, y)
        >>> out
        tensor([  1.,   4.,  27., 256.], dtype=oneflow.float32)
        
    """
    return flow._C.pow(input, exponent)


def addmm(x, mat1, mat2, alpha=1, beta=1):
    if len(x.shape) > 2 or len(mat1.shape) > 2 or len(mat2.shape) > 2:
        raise ValueError("input matrixes shape can not be greater than 2")
    else:
        return _mul(x, beta) + _mul(flow._C.matmul(mat1, mat2), alpha)


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

    For inputs of type `FloatTensor` or `DoubleTensor`, arguments :attr:`beta` and
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
        flow.Size([2, 3])

        >>> input2 = flow.tensor(np.array([1.7]))
        >>> mat1 = flow.tensor(np.array([[1,2],[5,9.1],[7.7,1.4]]))
        >>> mat2 = flow.tensor(np.array([[1,2,3.7],[5,9.1,6.8]]))
        >>> output2 = flow.addmm(input2, mat1, mat2, alpha=1, beta=2)
        >>> output2
        tensor([[14.4000, 23.6000, 20.7000],
                [53.9000, 96.2100, 83.7800],
                [18.1000, 31.5400, 41.4100]], dtype=oneflow.float64)
        >>> output2.shape
        flow.Size([3, 3])
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
            return (flow.gather(input, indices, dim=axis), indices)
        else:
            perm = get_perm_when_transpose_axis_to_last_dim(num_axes, axis)
            x = flow._C.transpose(input, perm=perm)
            if self.largest:
                indices = self._op_topk_last_dim(x)[0]
            else:
                neg_input = flow.mul(x, -1)
                indices = self._op_topk_last_dim(neg_input)[0]
            indices = flow._C.transpose(indices, perm=get_inversed_perm(perm))
            return (flow.gather(input, indices, dim=axis), indices)


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
        flow.Size([2, 3])
        >>> indices.shape
        flow.Size([2, 3])
        >>> (values, indices) = flow.topk(flow.Tensor(x), k=2, dim=1, largest=False)
        >>> values
        tensor([[1., 2.],
                [1., 2.]], dtype=oneflow.float32)
        >>> indices
        tensor([[0, 4],
                [0, 4]], dtype=oneflow.int32)
        >>> values.shape
        flow.Size([2, 2])
        >>> indices.shape
        flow.Size([2, 2])

    """
    return Topk(k=k, dim=dim, largest=largest, sorted=sorted)(input)




if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
