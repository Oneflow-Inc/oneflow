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
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.framework.tensor import register_tensor_op
from oneflow.compatible.single_client.nn.module import Module


class Norm(Module):
    def __init__(self, ord=None, dim=None, keepdim=False) -> None:
        super().__init__()
        self.ord = ord
        self.dim = dim
        self.keepdim = keepdim

    def _vector_norm(self, x, ord, dim):
        if isinstance(ord, str) and ord in ["fro", "nuc"]:
            raise ValueError("Norm order {} is not supported for vectors".format(ord))
        elif isinstance(ord, float) and ord in [float("inf"), float("-inf")]:
            if ord == float("inf"):
                return flow.experimental.max(flow.experimental.abs(x), dim=dim)
            else:
                return flow.experimental.min(flow.experimental.abs(x), dim=dim)
        elif isinstance(ord, int):
            if ord == 0:
                return flow.tensor([flow.experimental.argwhere(x).shape[0]])
            else:
                return flow.experimental.pow(
                    flow.experimental.sum(
                        flow.experimental.pow(flow.experimental.abs(x), ord), dim=dim
                    ),
                    1.0 / ord,
                )
        else:
            raise ValueError("Invalid norm order: {}".format(ord))

    def _matrix_norm(self, x, ord, dim):
        if isinstance(ord, str) and ord in ["fro", "nuc"]:
            if ord == "nuc":
                raise NotImplementedError
            else:
                return flow.experimental.sqrt(
                    flow.experimental.sum(flow.experimental.square(x), dim=dim)
                )
        elif isinstance(ord, float) and ord in [float("inf"), float("-inf")]:
            if ord == float("inf"):
                return flow.experimental.max(
                    flow.experimental.sum(flow.experimental.abs(x), dim=1)
                )
            else:
                return flow.experimental.min(
                    flow.experimental.sum(flow.experimental.abs(x), dim=1)
                )
        elif isinstance(ord, int):
            if ord == 1:
                return flow.experimental.max(
                    flow.experimental.sum(flow.experimental.abs(x), dim=0)
                )
            elif ord == -1:
                return flow.experimental.min(
                    flow.experimental.sum(flow.experimental.abs(x), dim=0)
                )
            elif ord == 2:
                raise NotImplementedError
            elif ord == -2:
                raise NotImplementedError
            else:
                raise ValueError(
                    "Norm order {} is not supported for matrices".format(ord)
                )
        else:
            raise ValueError("Invalid norm order: {}".format(ord))

    def _whether_keepdim(self, x):
        if self.keepdim == True and self.dim != None:
            return flow.experimental.unsqueeze(x, self.dim)
        else:
            return x

    def forward(self, x):
        num_axes = len(x.shape)
        if self.dim == None and self.ord == None:
            res = self._vector_norm(x.reshape((1, -1))[0], ord=2, dim=self.dim)
        elif self.dim == None and self.ord != None:
            assert (
                num_axes <= 2
            ), "input must be 1-D or 2-D when dim is None and ord is not None"
            res = (
                self._vector_norm(x, self.ord, self.dim)
                if num_axes == 1
                else self._matrix_norm(x, self.ord, self.dim)
            )
        elif isinstance(self.dim, (int, tuple, list)):
            if isinstance(self.dim, int):
                self.dim = self.dim if self.dim >= 0 else self.dim + num_axes
                assert 0 <= self.dim < num_axes, "dim out of range"
                res = self._vector_norm(
                    x, ord=2 if self.ord == None else self.ord, dim=self.dim
                )
            else:
                temp = list(self.dim) if isinstance(self.dim, tuple) else self.dim
                for i in range(len(temp)):
                    temp[i] = temp[i] if temp[i] >= 0 else temp[i] + num_axes
                    assert 0 <= temp[i] < num_axes, "dim out of range"
                self.dim = temp
                res = self._matrix_norm(
                    x, ord="fro" if self.ord == None else self.ord, dim=self.dim
                )
        else:
            raise ValueError("Invalid dimension: {}".format(self.dim))
        return self._whether_keepdim(res)


def norm_op(input, ord=None, dim=None, keepdim=False):
    """linalg.norm(input, ord=None, dim=None, keepdim=False, *, out=None) -> Tensor

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

        >>> import oneflow.compatible.single_client.experimental as flow
        >>> from oneflow.compatible.single_client.experimental import linalg as LA
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
def norm_tensor_op(input, ord=None, dim=None, keepdim=False):
    """
    See :func:`oneflow.compatible.single_client.experimental.linalg.norm.`
    """
    return Norm(ord, dim, keepdim)(input)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
