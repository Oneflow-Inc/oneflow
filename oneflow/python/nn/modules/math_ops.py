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
from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.nn.module import Module
from oneflow.python.framework.tensor import register_tensor_op


def _check_axis(axis, shape):
    # TODO(yaochi): refine this function when all related ops in `python/ops/math_ops.py` migrated
    if axis is None:
        axis = list(range(len(shape)))

    if isinstance(axis, int):
        axis = [axis]

    assert isinstance(axis, (list, tuple)), "Invalid axis {}".format(axis)
    for x in axis:
        if x < 0:
            x += len(shape)
        assert x >= 0 and x < len(shape), "Invalid axis {}, len(shape): {}".format(
            axis, len(shape)
        )

    return axis


class Sum(Module):
    def __init__(
        self, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False
    ) -> None:
        super().__init__()

        self.axis = axis
        self.keepdims = keepdims
        self._op = (
            flow.builtin_op("reduce_sum")
            .Input("input_tensor")
            .Output("output_tensor")
            .Attr("keepdims", keepdims)
            .Build()
        )

    def forward(self, input):
        axis_checked = _check_axis(self.axis, input.shape)
        if len(axis_checked) == 0:
            return input
        return self._op(input, axis=axis_checked)[0]


@oneflow_export("sum")
@register_tensor_op("sum")
def _sum(input, dim=None, keepdims=False):
    r"""Computes the sum of row of elements in a tensor in the given axis, if the axis is None, sum of all elements will be caculated.
    For example:

    .. code-block:: python

        import oneflow as flow
        input = flow.Tensor(np.random.randn(4, 5, 6), dtype=flow.float32)
        of_out = flow.sum(input, dim=(2,1))

    """

    return Sum(dim, keepdims)(input)


class ScalarMul(Module):
    def __init__(self, operand) -> None:
        super().__init__()
        self._op = flow.builtin_op("scalar_mul").Input("in").Output("out")
        if isinstance(operand, int):
            self._op = (
                self._op.Attr("has_int_operand", True)
                .Attr("has_float_operand", False)
                .Attr("int_operand", operand)
                .Attr("float_operand", 0.0)
                .Build()
            )
        elif isinstance(operand, float):
            self._op = (
                self._op.Attr("has_int_operand", False)
                .Attr("has_float_operand", True)
                .Attr("int_operand", 0)
                .Attr("float_operand", operand)
                .Build()
            )
        else:
            raise ValueError("operand type can only be int or float")

    def forward(self, x):
        return self._op(x)[0]


class ScalarMulByTensor(Module):
    def __init__(self) -> None:
        super().__init__()
        self._op = (
            flow.builtin_op("scalar_mul_by_tensor")
            .Input("x")
            .Input("scalar")
            .Output("y")
            .Build()
        )

    def forward(self, x, y):
        return self._op(x, y)[0]


class ElementwiseMul(Module):
    def __init__(self) -> None:
        super().__init__()
        self._op = (
            flow.builtin_op("multiply").Input("x").Input("y").Output("out").Build()
        )

    def forward(self, x, y):
        return self._op(x, y)[0]


class BroadcastMul(Module):
    def __init__(self) -> None:
        super().__init__()
        self._op = (
            flow.builtin_op("broadcast_mul").Input("x").Input("y").Output("z").Build()
        )

    def forward(self, x, y):
        return self._op(x, y)[0]


@oneflow_export("mul")
@register_tensor_op("mul")
def _mul(x, y):
    r"""Computes the multiplication of x by y for each element, scalar and broadcast promotation are supported.
    The formula is:
    .. math::
        out = x \times y
    For example:

    .. code-block:: python

        import oneflow as flow

        # element-wise multiply
        x = flow.Tensor(np.random.randn(2,3))
        y = flow.Tensor(np.random.randn(2,3))
        out = flow.mul(x,y).numpy()
        print(out.shape) # (2,3)

        # scalar mutiply
        x = 5
        y = flow.Tensor(np.random.randn(2,3))
        out = flow.mul(x,y).numpy()
        print(out.shape) # (2,3)

        # broadcast mutiply
        x = flow.Tensor(np.random.randn(1,1))
        y = flow.Tensor(np.random.randn(2,3))
        out = flow.mul(x,y).numpy()
        print(out.shape) # (2,3)

    """

    if isinstance(x, (int, float)):
        return ScalarMul(x)(y)
    elif isinstance(y, (int, float)):
        return ScalarMul(y)(x)
    elif x.shape == y.shape:
        return ElementwiseMul()(x, y)
    elif x.shape == (1,):
        return ScalarMulByTensor()(y, x)
    elif y.shape == (1,):
        return ScalarMulByTensor()(x, y)
    else:
        return BroadcastMul()(x, y)


class Mean(Module):
    def __init__(
        self,
        axis: Optional[Union[collections.Sized, int]] = None,
        keepdims: bool = False,
    ) -> None:
        super().__init__()
        self.keepdims = keepdims
        self.axis = axis
        # TODO: add if input.is_dynamic branch like flow.math.reduce_mean
        if axis is None:
            self.axes = []
        else:
            self.axes = list(axis) if isinstance(axis, collections.Sized) else [axis]

    def forward(self, input_tensor):
        reduce_sum = flow.sum(input_tensor, dim=self.axis, keepdims=self.keepdims)
        reduce_count = 1
        if len(self.axes) == 0:
            for dim in input_tensor.shape:
                reduce_count *= dim
        else:
            for i in self.axes:
                reduce_count *= input_tensor.shape[i]
        return flow.mul(reduce_sum, 1.0 / reduce_count)


@oneflow_export("mean")
@register_tensor_op("mean")
def _mean(input_tensor, dim=None, keepdim=False):
    r"""Computes the mean of row of elements in a tensor in the given axis,
    if the axis is None, mean of all elements will be caculated.

    For example:

    .. code-block:: python

        import oneflow as flow

        input = flow.Tensor([[1, 2, 3], [4, 5, 6]])
        out = flow.mean(input)
        # out: [3.5]
        print(out.numpy())

        input = flow.Tensor([[1, 2, 3], [4, 5, 6]])
        out = flow.mean(input, axis=0)
        # out: [2.5 3.5 4.5]
        print(out.numpy())

        input = flow.Tensor([[1, 2, 3], [4, 5, 6]])
        out = flow.mean(input, axis=1)
        # out: [ 2. 5.]
        print(out.numpy())

    """

    return Mean(axis=dim, keepdims=keepdim)(input_tensor)


class ScalarSubByTensor(Module):
    def __init__(self) -> None:
        super().__init__()
        self._op = (
            flow.builtin_op("scalar_sub_by_tensor")
            .Input("x")
            .Input("scalar")
            .Output("y")
            .Build()
        )

    def forward(self, x, y):
        return self._op(x, y)[0]


class BroadcastSub(Module):
    def __init__(self) -> None:
        super().__init__()
        self._op = (
            flow.builtin_op("broadcast_sub").Input("x").Input("y").Output("z").Build()
        )

    def forward(self, x, y):
        return self._op(x, y)[0]


class ScalarAdd(Module):
    def __init__(self, operand) -> None:
        super().__init__()
        self._op = flow.builtin_op("scalar_add").Input("in").Output("out")

        if isinstance(operand, int):
            self._op = (
                self._op.Attr("has_int_operand", True)
                .Attr("has_float_operand", False)
                .Attr("int_operand", operand)
                .Attr("float_operand", 0.0)
                .Build()
            )
        elif isinstance(operand, float):
            self._op = (
                self._op.Attr("has_int_operand", False)
                .Attr("has_float_operand", True)
                .Attr("int_operand", 0)
                .Attr("float_operand", operand)
                .Build()
            )
        else:
            raise ValueError("operand type can only be int or float")

    def forward(self, x):
        return self._op(x)[0]


@oneflow_export("sub")
@register_tensor_op("sub")
def _sub(x, y):
    r"""Computes the subtraction of x by y for each element, scalar and broadcast promotation are supported.
    The formula is:
    .. math::
        out = x - y
    For example:

    .. code-block:: python

        import oneflow as flow

        # element-wise subtract
        x = flow.Tensor(np.random.randn(2,3))
        y = flow.Tensor(np.random.randn(2,3))
        out = flow.sub(x,y).numpy()
        print(out.shape) # (2,3)

        # scalar subtract
        x = 5
        y = flow.Tensor(np.random.randn(2,3))
        out = flow.sub(x,y).numpy()
        print(out.shape) # (2,3)

        # broadcast subtract
        x = flow.Tensor(np.random.randn(1,1))
        y = flow.Tensor(np.random.randn(2,3))
        out = flow.sub(x,y).numpy()
        print(out.shape) # (2,3)

    """

    if isinstance(x, (int, float)):
        return ScalarAdd(x)(ScalarMul(-1)(y))
    elif isinstance(y, (int, float)):
        return ScalarAdd(-1 * y)(x)
    elif x.shape == y.shape:
        # TODO: add element-wise op
        return BroadcastSub()(x, y)
    elif y.shape == (1,):
        return ScalarSubByTensor()(x, y)
    else:
        return BroadcastSub()(x, y)


class BroadcastDiv(Module):
    def __init__(self) -> None:
        super().__init__()
        self._op = (
            flow.builtin_op("broadcast_div").Input("x").Input("y").Output("z").Build()
        )

    def forward(self, x, y):
        return self._op(x, y)[0]


class ScalarDivByTensor(Module):
    def __init__(self) -> None:
        super().__init__()
        self._op = (
            flow.builtin_op("scalar_div_by_tensor")
            .Input("x")
            .Input("scalar")
            .Output("y")
            .Build()
        )

    def forward(self, x, scalar):
        return self._op(x, scalar)[0]


@oneflow_export("div")
@register_tensor_op("div")
def _div(x, y):
    r"""Computes the division of x by y for each element, scalar and broadcast promotation are supported.
    The formula is:
    .. math::
        out = \frac{X}{Y}
    Args:
        x (Union[int, float, flow.Tensor]): X.
        y (Union[int, float, flow.Tensor]): Y.
    For example:
    .. code-block:: python

        import oneflow as flow

        # element-wise divide
        x = flow.Tensor(np.random.randn(2,3))
        y = flow.Tensor(np.random.randn(2,3))
        out = flow.div(x,y).numpy()
        print(out.shape) # (2,3)

        # scalar divide
        x = 5
        y = flow.Tensor(np.random.randn(2,3))
        out = flow.div(x,y).numpy()
        print(out.shape) # (2,3)

        # broadcast divide
        x = flow.Tensor(np.random.randn(1,1))
        y = flow.Tensor(np.random.randn(2,3))
        out = flow.div(x,y).numpy()
        print(out.shape) # (2,3)

    """

    if isinstance(x, (int, float)):
        return ScalarMul(x)(flow.reciprocal(y))
    elif isinstance(y, (int, float)):
        if y == 0 or y == 0.0:
            y = 0.0
        else:
            y = 1.0 / (float(y))
        return ScalarMul(y)(x)
    elif x.shape == y.shape:
        return BroadcastDiv()(x, y)
    elif y.shape == (1,):
        return ScalarDivByTensor()(x, y)
    else:
        return BroadcastDiv()(x, y)


class Reciprocal(Module):
    def __init__(self) -> None:
        super().__init__()
        self._op = flow.builtin_op("reciprocal_no_nan").Input("x").Output("y").Build()

    def forward(self, x):
        return self._op(x)[0]


@oneflow_export("reciprocal")
@register_tensor_op("reciprocal")
def _reciprocal(x):
    r"""Computes the safe reciprocal of x. If x is zero, the reciprocal will
    be also set to zero.

    For example:

    .. code-block:: python

        import oneflow as flow
        x = flow.Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
        out = flow.reciprocal()(x)
        # out [[1.         0.5        0.33333334]
               [0.25       0.2        0.16666667]]

    """

    return Reciprocal()(x)


class ScalarAddByTensor(Module):
    def __init__(self) -> None:
        super().__init__()
        self._op = (
            flow.builtin_op("scalar_add_by_tensor")
            .Input("x")
            .Input("scalar")
            .Output("y")
            .Build()
        )

    def forward(self, x, y):
        return self._op(x, y)[0]


class ElementwiseAdd(Module):
    def __init__(self) -> None:
        super().__init__()
        self._op = flow.builtin_op("add_n").Input("in", 2).Output("out").Build()

    def forward(self, x, y):
        return self._op(x, y)[0]


class BroadcastAdd(Module):
    def __init__(self) -> None:
        super().__init__()
        self._op = (
            flow.builtin_op("broadcast_add").Input("x").Input("y").Output("z").Build()
        )

    def forward(self, x, y):
        return self._op(x, y)[0]


@oneflow_export("add")
@register_tensor_op("add")
def _add(x, y):
    r"""Computes the addition of x by y for each element, scalar and broadcast promotation are supported.
    The formula is:
    .. math::
        out = x + y
    For example:

    .. code-block:: python

        import oneflow as flow

        # element-wise add
        x = flow.Tensor(np.random.randn(2,3))
        y = flow.Tensor(np.random.randn(2,3))
        out = flow.add(x,y).numpy()
        print(out.shape) # (2,3)

        # scalar add
        x = 5
        y = flow.Tensor(np.random.randn(2,3))
        out = flow.add(x,y).numpy()
        print(out.shape) # (2,3)

        # broadcast add
        x = flow.Tensor(np.random.randn(1,1))
        y = flow.Tensor(np.random.randn(2,3))
        out = flow.add(x,y).numpy()
        print(out.shape) # (2,3)

    """

    if isinstance(x, (int, float)):
        return ScalarAdd(x)(y)
    elif isinstance(y, (int, float)):
        return ScalarAdd(y)(x)
    elif x.shape == y.shape:
        return ElementwiseAdd()(x, y)
    elif x.shape == (1,):
        return ScalarAddByTensor()(y, x)
    elif y.shape == (1,):
        return ScalarAddByTensor()(x, y)
    else:
        return BroadcastAdd()(x, y)


class Sin(Module):
    def __init__(self) -> None:
        super().__init__()
        self._op = flow.builtin_op("sin").Input("x").Output("y").Build()

    def forward(self, x):
        return self._op(x)[0]


@oneflow_export("sin")
@register_tensor_op("sin")
def sin_op(tensor):
    r"""
    Returns a new tensor with the sine of the elements of :attr:`input`.
    .. math::
        \text{out}_{i} = \sin(\text{input}_{i})
    Args:
        input (Tensor) – the input tensor.
    For example:
    .. code-block:: python
        import oneflow as flow
        import numpy as np
        arr = np.array([-0.5461,  0.1347, -2.7266, -0.2746])
        input = flow.Tensor(arr, dtype=flow.float32)
        output = flow.sin(input)
        # [-0.51935846  0.13429303 -0.40318328 -0.27116194]
    """
    return Sin()(tensor)


class Cos(Module):
    def __init__(self) -> None:
        super().__init__()
        self._op = flow.builtin_op("cos").Input("x").Output("y").Build()

    def forward(self, x):
        return self._op(x)[0]


@oneflow_export("cos")
@register_tensor_op("cos")
def cos_op(tensor):
    r"""
    Returns a new tensor with the cosine  of the elements of :attr:`input`.
    .. math::
        \text{out}_{i} = \cos(\text{input}_{i})
    Args:
        input (Tensor) – the input tensor.
    For example:
    .. code-block:: python
        import oneflow as flow
        import numpy as np
        arr = np.array([1.4309,  1.2706, -0.8562,  0.9796])
        input = flow.Tensor(arr, dtype=flow.float32)
        output = flow.cos(input)
        # [0.13944048 0.29570782 0.6553126  0.5573547 ]
        
    """
    return Cos()(tensor)


class Log(Module):
    def __init__(self) -> None:
        super().__init__()
        self._op = flow.builtin_op("log").Input("x").Output("y").Build()

    def forward(self, x):
        return self._op(x)[0]


@oneflow_export("log")
@register_tensor_op("log")
def log_op(tensor):
    r"""
    Returns a new tensor with the natural logarithm of the elements of :attr:`input`.
    .. math::
        y_{i} = \log_{e} (x_{i})
    Args:
        input (Tensor) – the input tensor.
    For example:
    .. code-block:: python
        import oneflow as flow
        import numpy as np
        arr = np.random.randn(2, 3, 4, 5)
        input = flow.Tensor(arr, dtype=flow.float32)
        output = flow.log(input)
        # equal to np.log(input)
        
    """
    return Log()(tensor)


class Subtract(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, y):
        if isinstance(x, (int, float)):
            return ScalarAdd(x)(-1 * y)
        elif isinstance(y, (int, float)):
            return ScalarAdd(-1 * y)(x)
        elif x.shape == y.shape:
            # TODO: add element-wise op
            return BroadcastSub()(x, y)
        elif x.shape == (1,):
            return ScalarSubByTensor()(y, x)
        elif y.shape == (1,):
            return ScalarSubByTensor()(x, y)
        else:
            return BroadcastSub()(x, y)


class Sqrt(Module):
    def __init__(self) -> None:
        super().__init__()
        self.sqrt_op = flow.builtin_op("sqrt").Input("x").Output("y").Build()

    def forward(self, input):
        return self.sqrt_op(input)[0]


@oneflow_export("sqrt")
@register_tensor_op("sqrt")
def sqrt_op(input):
    r"""Returns a new tensor with the square-root of the elements of :attr:`input`.

        .. math::
            \text{out}_{i} = \sqrt{\text{input}_{i}}

        Args:
            input (Tensor) – the input tensor.

         For example:

        .. code-block:: python

            import oneflow as flow
            import numpy as np

            arr = np.random.randn(3, 2, 5, 7)
            input = flow.Tensor(arr)
            output = flow.sqrt(input)
            # output equal to np.sqrt(arr)
        """
    return Sqrt()(input)


class Square(Module):
    def __init__(self) -> None:
        super().__init__()
        self.square_op = flow.builtin_op("square").Input("x").Output("y").Build()

    def forward(self, input):
        return self.square_op(input)[0]


@oneflow_export("square")
@register_tensor_op("square")
def square_op(input):
    r"""Returns a new tensor with the square of the elements of :attr:`input`.

        .. math::
            \text{out}_{i} = \sqrt{\text{input}_{i}}

        Args:
            input (Tensor) – the input tensor.

         For example:

        .. code-block:: python

            import oneflow as flow
            import numpy as np

            arr = np.random.randn(3, 2, 5, 7)
            input = flow.Tensor(arr)
            output = flow.square(input)
            # output equal to np.square(arr)
        """
    return Square()(input)


class Std(Module):
    def __init__(self, dim=None, unbiased=True, keepdim=False) -> None:
        super().__init__()
        assert unbiased == True, "Only support 'unbiased=True' for now!"
        self.unbiased = unbiased
        self.keepdim = keepdim
        self.dim = dim
        self.reduce_count = 1
        self.square_op = Square()
        self.sqrt_op = Sqrt()
        self.subtract_op = Subtract()

    def forward(self, x):
        self.axis = _check_axis(self.dim, x.shape)
        if isinstance(self.axis, list) and len(self.axis) == 0:
            return flow.tmp.zeros(size=x.shape)
        else:
            if len(self.axis) == 0:
                self.reduce_count = x.nelemenet()
            else:
                for i in self.axis:
                    self.reduce_count *= x.shape[i]

            sum = Sum(self.axis, self.keepdim)(self.square_op(x)) / self.reduce_count
            square = self.square_op(Sum(self.axis, self.keepdim)(x) / self.reduce_count)
            subtract = self.subtract_op(sum, square)
            res = self.sqrt_op(subtract)
            return res


@oneflow_export("tmp.std")
@register_tensor_op("std")
def std_op(tensor, dim, unbiased=True, keepdim=False):
    r"""
    Returns the standard-deviation of each row of the :attr:`input` tensor in the
    dimension :attr:`dim`. If :attr:`dim` is a list of dimensions,
    reduce over all of them.

    If keepdim is True, the output tensor is of the same size as input except in 
    the dimension(s) dim where it is of size 1. Otherwise, dim is squeezed, 
    resulting in the output tensor having 1 (or len(dim)) fewer dimension(s).

    If :attr:`unbiased` is ``False``, then the standard-deviation will be calculated
    via the biased estimator. Otherwise, Bessel's correction will be used.

    Args:
        input (Tensor) – the input tensor.
        dim (int or tuple of python:ints) – the dimension or dimensions to reduce.
        unbiased (bool) – whether to use the unbiased estimation or not
        keepdim (bool) – whether the output tensor has `dim` retained or not.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np

        arr = np.random.randn(2, 3, 4, 5)
        input = flow.Tensor(arr)
        output = flow.std(input, dim=2)

        # equal to numpy np.std(arr, axis=2)

    """
    return Std(dim, unbiased, keepdim)(tensor)
