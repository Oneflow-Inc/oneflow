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
from oneflow.python.nn.module import Module
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.framework.tensor import register_tensor_op


def check_dim(num_dims, input_dim):
    if input_dim == None:
        dim = input_dim
    elif isinstance(input_dim, (int, tuple)):
        if isinstance(input_dim, int):
            dim = input_dim if input_dim >= 0 else input_dim + num_dims
            if dim >= num_dims or dim < 0:
                raise IndexError("Dimension out of range")
        else:
            temp = list(input_dim)
            for i in range(len(temp)):
                temp[i] = temp[i] if temp[i] >= 0 else temp[i] + num_dims
                if temp[i] >= num_dims or temp[i] < 0:
                    raise IndexError("Dimension out of range")
            dim = temp
    else:
        raise TypeError("linalg_vector_norm(): argument 'dim' must be tuple of ints, not {}".format(type(input_dim)))
    return dim




class Vector_Norm(Module):
    def __init__(self, ord=None, dim=None, keepdim=False) -> None:
        super().__init__()
        if ord == None:
            self.ord = 2.0
        elif isinstance(ord, (int, float)):
            self.ord = float(ord)
        else:
            raise TypeError("linalg_vector_norm(): argument 'ord' must be Number, not {}".format(type(ord)))
        self.dim = dim
        self.keepdim = keepdim

    def _vector_norm(self, x, ord, dim, keepdim = False):
        if ord == 0:
            # TODO: fix error when input are all zero vector
            return flow.tensor([flow.experimental.argwhere(x).shape[0]])
        elif ord == float("inf"):
            return flow.experimental.max(flow.experimental.abs(x), dim=dim, keepdim = keepdim)
        elif ord == float("-inf"):
            return flow.experimental.min(flow.experimental.abs(x), dim=dim, keepdim = keepdim)
        else:
            return flow.experimental.pow(
                    flow.experimental.sum(
                        flow.experimental.pow(flow.experimental.abs(x), ord), dim=dim, keepdim=keepdim
                    ),
                    1.0 / ord,
                )

    def forward(self, x):
        num_dims = len(x.shape)
        dim = check_dim(num_dims, self.dim)
        if dim == None:
            return self._vector_norm(x.reshape((1, -1))[0], ord = self.ord, dim=self.dim, keepdim= self.keepdim)
        else:
            return self._vector_norm(x, ord = self.ord, dim=dim, keepdim = self.keepdim)



class Matrix_Norm(Module):
    def __init__(self, ord="fro", dim=(-2,-1), keepdim=False) -> None:
        super().__init__()
        if isinstance(ord, str) and ord in ["fro", "nuc"]:
            self.ord = ord
        elif isinstance(ord, float) and ord in [float("inf"), float("-inf")]:
            self.ord = ord
        elif isinstance(ord, int) and ord in [1,-1,2,-2]:
            self.ord = ord
        elif ord == None:
            self.ord = "fro"
        else:
            raise TypeError("linalg_matrix_norm(): argument 'ord' must be Number, not {}".format(type(ord)))
        self.ord = "fro" if ord == None else ord
        self.dim = dim
        self.keepdim = keepdim

    def _matrix_norm(self, x, ord, dim, keepdim):
        if ord == "nuc":
            raise NotImplementedError
        elif ord == "fro":
            return flow.experimental.sqrt(
                flow.experimental.sum(flow.experimental.square(x), dim=dim, keepdim= keepdim)
            )

        elif ord == float("inf"):
            return flow.experimental.max(
                flow.experimental.sum(flow.experimental.abs(x), dim=1, keepdim= keepdim)
            )
        elif ord == float("-inf"):
            return flow.experimental.min(
                flow.experimental.sum(flow.experimental.abs(x), dim=1, keepdim=keepdim)
            )

        elif ord == 1:
            return flow.experimental.max(
                flow.experimental.sum(flow.experimental.abs(x), dim=0, keepdim=keepdim)
            )
        elif ord == -1:
            return flow.experimental.min(
                flow.experimental.sum(flow.experimental.abs(x), dim=0, keepdim=keepdim)
            )
        elif ord == 2:
            raise NotImplementedError
        elif ord == -2:
            raise NotImplementedError
        else:
            raise ValueError("Invalid norm order: {}".format(ord))

    def forward(self, x):
        num_dims = len(x.shape)
        dim = check_dim(num_dims, self.dim)        
        return self._matrix_norm(x, ord = self.ord, dim= dim, keepdim= self.keepdim)


class Norm(Module):
    def __init__(self, ord=None, dim=None, keepdim=False) -> None:
        super().__init__()

        self.ord = ord
        self.dim = dim
        self.keepdim = keepdim


    def forward(self, x):
        if isinstance(self.dim, int) or (self.dim == None and self.ord == None):
            res = Vector_Norm(ord=self.ord, dim=self.dim, keepdim = self.keepdim)(x)
        elif isinstance(self.dim, tuple):
            res = Matrix_Norm(ord=self.ord, dim=self.dim, keepdim = self.keepdim)(x)
        elif self.dim == None and self.ord != None:
            assert (
                len(x.shape) <= 2
            ), "input must be 1-D or 2-D when dim is None and ord is not None"
            if len(x.shape) == 1:
                res = Vector_Norm(ord=self.ord, dim = self.dim, keepdim = self.keepdim)(x)
            else:
                res = Matrix_Norm(ord=self.ord, dim =self.dim, keepdim = self.keepdim)(x)
       
        return res


@oneflow_export("linalg.norm")
@experimental_api
def norm_op(input, ord=None, dim=None, keepdim=False):
    r"""linalg.norm(input, ord=None, dim=None, keepdim=False, *, out=None) -> Tensor

    Returns the matrix norm or vector norm of a given tensor.

    This function can calculate one of eight different types of matrix norms, or one
    of an infinite number of vector norms, depending on both the number of reduction
    dimensions and the value of the `ord` parameter.

    Args:
        input (Tensor): The input tensor. If dim is None, input must be 1-D or 2-D, unless :attr:`ord`
            is None. If both :attr:`dim` and :attr:`ord` are None, the 2-norm of the input flattened to 1-D
            will be returned. Its data type must be either a floating point or complex type. For complex
            inputs, the norm is calculated on of the absolute values of each element. If the input is
            complex and neither :attr:`dtype` nor :attr:`out` is specified, the result's data type will
            be the corresponding floating point type (e.g. float if :attr:`input` is complexfloat).

        ord (int, float, inf, -inf, 'fro', 'nuc', optional): The order of norm.
            inf refers to :attr:`float('inf')`, numpy's :attr:`inf` object, or any equivalent object.
            The following norms can be calculated:

            =====  ============================  ==========================
            ord    norm for matrices             norm for vectors
            =====  ============================  ==========================
            None   Frobenius norm                2-norm
            'fro'  Frobenius norm                -- not supported --
            'nuc'  -- not supported yet --       -- not supported --
            inf    max(sum(abs(x), dim=1))       max(abs(x))
            -inf   min(sum(abs(x), dim=1))       min(abs(x))
            0      -- not supported --           sum(x != 0)
            1      max(sum(abs(x), dim=0))       as below
            -1     min(sum(abs(x), dim=0))       as below
            2      -- not supported yet --       as below
            -2     -- not supported yet --       as below
            other  -- not supported --           sum(abs(x)**ord)**(1./ord)
            =====  ============================  ==========================

            Default: ``None``

        dim (int, 2-tuple of ints, 2-list of ints, optional): If :attr:`dim` is an int,
            vector norm will be calculated over the specified dimension. If :attr:`dim`
            is a 2-tuple of ints, matrix norm will be calculated over the specified
            dimensions. If :attr:`dim` is None, matrix norm will be calculated
            when the input tensor has two dimensions, and vector norm will be
            calculated when the input tensor has one dimension. Default: ``None``

        keepdim (bool, optional): If set to True, the reduced dimensions are retained
            in the result as dimensions with size one. Default: ``False``

        out (Tensor, optional): The output tensor.

    Examples::

        >>> import oneflow.experimental as flow
        >>> from oneflow.experimental import linalg as LA
        >>> import numpy as np
        >>> flow.enable_eager_execution()
        >>> a = flow.tensor(np.arange(9, dtype=np.float32) - 4)
        >>> a
        tensor([-4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.], dtype=oneflow.float32)
        >>> b = a.reshape((3, 3))
        >>> b
        tensor([[-4., -3., -2.],
                [-1.,  0.,  1.],
                [ 2.,  3.,  4.]], dtype=oneflow.float32)

        >>> LA.norm(a)
        tensor([7.746], dtype=oneflow.float32)
        >>> LA.norm(b)
        tensor([7.746], dtype=oneflow.float32)
        >>> LA.norm(b, 'fro')
        tensor([7.746], dtype=oneflow.float32)
        >>> LA.norm(a, float('inf'))
        tensor([4.], dtype=oneflow.float32)
        >>> LA.norm(b, float('inf'))
        tensor([9.], dtype=oneflow.float32)
        >>> LA.norm(a, -float('inf'))
        tensor([0.], dtype=oneflow.float32)
        >>> LA.norm(b, -float('inf'))
        tensor([2.], dtype=oneflow.float32)

        >>> LA.norm(a, 1)
        tensor([20.], dtype=oneflow.float32)
        >>> LA.norm(b, 1)
        tensor([7.], dtype=oneflow.float32)
        >>> LA.norm(a, -1)
        tensor([0.], dtype=oneflow.float32)
        >>> LA.norm(b, -1)
        tensor([6.], dtype=oneflow.float32)
        >>> LA.norm(a, 2)
        tensor([7.746], dtype=oneflow.float32)
        >>> LA.norm(a, -2)
        tensor([0.], dtype=oneflow.float32)
        >>> LA.norm(a, 3)
        tensor([5.848], dtype=oneflow.float32)
        >>> LA.norm(a, -3)
        tensor([0.], dtype=oneflow.float32)

    Using the :attr:`dim` argument to compute vector norms::

        >>> c = flow.tensor([[1., 2., 3.],
        ...                   [-1, 1, 4]])
        >>> LA.norm(c, dim=0)
        tensor([1.4142, 2.2361, 5.    ], dtype=oneflow.float32)
        >>> LA.norm(c, dim=1, keepdim = True)
        tensor([[3.7417],
                [4.2426]], dtype=oneflow.float32)
        >>> LA.norm(c, ord=1, dim=1)
        tensor([6., 6.], dtype=oneflow.float32)

    Using the :attr:`dim` argument to compute matrix norms::

        >>> m = flow.tensor(np.arange(8, dtype=np.float32)).reshape((2, 2, 2))
        >>> LA.norm(m, dim=(1,2))
        tensor([ 3.7417, 11.225 ], dtype=oneflow.float32)
    """
    return Norm(ord, dim, keepdim)(input)


@register_tensor_op("norm")
@experimental_api
def norm_tensor_op(input, ord=None, dim=None, keepdim=False):
    r"""
    See :func:`oneflow.experimental.linalg.norm.`
    """
    return Norm(ord, dim, keepdim)(input)

@oneflow_export("linalg.vector_norm")
@experimental_api
def vector_norm_tensor_op(input, ord=None, dim=None, keepdim=False):
    r"""
    linalg.vector_norm(input, ord=2, dim=None, keepdim=False, *, dtype=None, out=None) -> Tensor

    Computes a vector norm.

    Supports input of float, double dtypes.

    This function does not necessarily treat multidimensonal attr:`input` as a batch of
    vectors, instead:

    - If :attr:`dim`\ `= None`, :attr:`A` will be flattened before the norm is computed.
    - If :attr:`dim` is an `int` or a `tuple`, the norm will be computed over these dimensions
    and the other dimensions will be treated as batch dimensions.

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

    .. seealso::

            :func:`flow.linalg.matrix_norm` computes a matrix norm.

    Args:
        input (Tensor): tensor, flattened by default, but this behavior can be
            controlled using :attr:`dim`.
        ord (int, float, inf, -inf, 'fro', 'nuc', optional): order of norm. Default: `2`
        dim (int, Tuple[int], optional): dimensions over which to compute
            the norm. See above for the behavior when :attr:`dim`\ `= None`.
            Default: `None`
        keepdim (bool, optional): If set to `True`, the reduced dimensions are retained
            in the result as dimensions with size one. Default: `False`

    Keyword args:
        out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.
        dtype (:class:`torch.dtype`, optional): If specified, the input tensor is cast to
            :attr:`dtype` before performing the operation, and the returned tensor's type
            will be :attr:`dtype`. Default: `None`

    Returns:
        A real-valued tensor.

    Examples::

        >>> import oneflow.experimental as flow
        >>> from oneflow.experimental import linalg as LA
        >>> import numpy as np
        >>> flow.enable_eager_execution()
        >>> a = flow.tensor(np.arange(9, dtype=np.float32) - 4)
        >>> a
        tensor([-4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.], dtype=oneflow.float32)
        >>> b = a.reshape((3, 3))
        >>> b
        tensor([[-4., -3., -2.],
                [-1.,  0.,  1.],
                [ 2.,  3.,  4.]], dtype=oneflow.float32)
        >>> LA.vector_norm(a, ord=3.5)
        tensor([5.4345], dtype=oneflow.float32)
        >>> LA.vector_norm(b, ord=3.5)
        tensor([5.4345], dtype=oneflow.float32)
    """
    return Vector_Norm(ord, dim, keepdim)(input)


@oneflow_export("linalg.matrix_norm")
@experimental_api
def matrix_norm_tensor_op(input, ord=None, dim=None, keepdim=False):
    r"""
    See :func:`oneflow.experimental.linalg.norm.`
    """
    return Matrix_Norm(ord, dim, keepdim)(input)

if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
