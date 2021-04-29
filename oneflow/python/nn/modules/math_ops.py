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
from typing import Optional, Sequence, Sized, Union, List, Tuple

import oneflow as flow
from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.nn.module import Module
from oneflow.python.nn.modules.utils import (
    _single,
    _pair,
    _triple,
    _reverse_repeat_tuple,
)
from oneflow.python.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from oneflow.python.ops.nn_ops import calc_pool_padding, get_dhw_offset
from oneflow.python.framework.tensor import register_tensor_op


def _check_axis(axis, shape):
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
        self,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: bool = False,
        name: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.axis = axis
        self.keepdims = keepdims
        self._op = (
            flow.builtin_op("reduce_sum", name)
            .Input("input_tensor")
            .Output("output_tensor")
            .Build()
        )

    def forward(self, input):
        axis_checked = _check_axis(self.axis, input.shape)
        if len(axis_checked) == 0:
            return input
        return self._op(input, axis=axis_checked, keepdims=self.keepdims)[0]


@oneflow_export("sum")
@register_tensor_op("sum")
def _sum(input, dim, keepdim=False):
    r"""Computes the sum of row of elements in a tensor in the given axis, if the axis is None, sum of all elements will be caculated.
    For example:

    .. code-block:: python

        #Example
        
        input = flow.Tensor(np.random.randn(4, 5, 6), dtype=flow.float32)
        of_out = flow.sum(input, dim=(2,1))

    """

    return Sum(dim, keepdim)(input)


class ScalarMul(Module):
    def __init__(self, operand, name=None) -> None:
        super().__init__()
        self.operand = operand
        self._op = flow.builtin_op("scalar_mul", name).Input("in").Output("out").Build()

    def forward(self, x):
        if isinstance(self.operand, int):
            return self._op(
                x,
                has_int_operand=True,
                has_float_operand=False,
                int_operand=self.operand,
                float_operand=0.0,
            )[0]
        elif isinstance(self.operand, float):
            return self._op(
                x,
                has_int_operand=False,
                has_float_operand=True,
                int_operand=0,
                float_operand=self.operand,
            )[0]
        else:
            raise ValueError("operand type can only be int or float")


class ScalarMulByTensor(Module):
    def __init__(self, name=None) -> None:
        super().__init__()
        self._op = (
            flow.builtin_op("scalar_mul_by_tensor", name)
            .Input("x")
            .Input("scalar")
            .Output("y")
            .Build()
        )

    def forward(self, x, y):
        return self._op(x, y)[0]


class ElementwiseMul(Module):
    def __init__(self, name=None) -> None:
        super().__init__()
        self._op = (
            flow.builtin_op("multiply", name)
            .Input("x")
            .Input("y")
            .Output("out")
            .Build()
        )

    def forward(self, x, y):
        return self._op(x, y)[0]


class BroadcastMul(Module):
    def __init__(self, name=None) -> None:
        super().__init__()
        self._op = (
            flow.builtin_op("broadcast_mul", name)
            .Input("x")
            .Input("y")
            .Output("z")
            .Build()
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
        
        # Example
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
        name: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims
        self.name = name

    def forward(self, input_tensor):
        reduce_sum = flow.sum(input_tensor, dim=self.axis, keepdims=self.keepdims)

        # TODO: add if input.is_dynamic branch like flow.math.reduce_mean

        if self.axis is None:
            axes = []
        else:
            axes = (
                list(self.axis)
                if isinstance(self.axis, collections.Sized)
                else [self.axis]
            )
        reduce_count = 1
        if len(axes) == 0:
            for dim in input_tensor.shape:
                reduce_count *= dim
        else:
            for i in axes:
                reduce_count *= input_tensor.shape[i]
        return flow.mul(reduce_sum, 1.0 / reduce_count)


@oneflow_export("mean")
@register_tensor_op("mean")
def _mean(input_tensor, dim, keepdim):
    r"""Computes the mean of row of elements in a tensor in the given axis, if the axis is None, mean of all elements will be caculated.
    
    For example:

    .. code-block:: python

        input = flow.Tensor([[1, 2, 3], [4, 5, 6]])
        out = flow.mean(input) # out: [3.5]
        print(out.numpy())
        
        input = flow.Tensor([[1, 2, 3], [4, 5, 6]])
        out = flow.mean(input, axis=0) # out: [2.5 3.5 4.5]
        print(out.numpy())

        input = flow.Tensor([[1, 2, 3], [4, 5, 6]])
        out = flow.mean(input, axis=1) # out: [ 2. 5.]
        print(out.numpy())

    """

    return Mean(axis=dim, keepdims=keepdim)(input_tensor)


class ScalarSubByTensor(Module):
    def __init__(self, name=None) -> None:
        super().__init__()
        self._op = (
            flow.builtin_op("scalar_sub_by_tensor", name)
            .Input("x")
            .Input("scalar")
            .Output("y")
            .Build()
        )

    def forward(self, x, y):
        return self._op(x, y)[0]


class BroadcastSub(Module):
    def __init__(self, name=None) -> None:
        super().__init__()
        self._op = (
            flow.builtin_op("broadcast_sub", name)
            .Input("x")
            .Input("y")
            .Output("z")
            .Build()
        )

    def forward(self, x, y):
        return self._op(x, y)[0]


class ScalarAdd(Module):
    def __init__(self, operand, name=None) -> None:
        super().__init__()
        self._op = flow.builtin_op("scalar_add", name).Input("in").Output("out")

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
    def __init__(self, name=None) -> None:
        super().__init__()
        self._op = (
            flow.builtin_op("broadcast_div", name)
            .Input("x")
            .Input("y")
            .Output("z")
            .Build()
        )

    def forward(self, x, y):
        return self._op(x, y)[0]


class ScalarDivByTensor(Module):
    def __init__(self, name=None) -> None:
        super().__init__()
        self._op = (
            flow.builtin_op("scalar_div_by_tensor", name)
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
        name (Optional[str], optional): The name for the operation. Defaults to None.
    For example:
    .. code-block:: python

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
    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__()
        self._op = (
            flow.builtin_op("reciprocal_no_nan", name).Input("x").Output("y").Build()
        )

    def forward(self, x):
        return self._op(x)[0]


@oneflow_export("reciprocal")
@register_tensor_op("reciprocal")
def _reciprocal(x):
    r"""Computes the safe reciprocal of x. If x is zero, the reciprocal will 
    be also set to zero.
    
    Args:
        name (Optional[str], optional): The name for the operation. Defaults to None.
    
    For example: 

    .. code-block:: python 
    
        x = flow.Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
        out = flow.reciprocal()(x)
        # out [[1.         0.5        0.33333334]
               [0.25       0.2        0.16666667]]

    """

    return Reciprocal()(x)


class ScalarAdd(Module):
    def __init__(self, operand, name=None) -> None:
        super().__init__()
        self._op = flow.builtin_op("scalar_add", name).Input("in").Output("out")

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


class ScalarAddByTensor(Module):
    def __init__(self, name=None) -> None:
        super().__init__()
        self._op = (
            flow.builtin_op("scalar_add_by_tensor", name)
            .Input("x")
            .Input("scalar")
            .Output("y")
            .Build()
        )

    def forward(self, x, y):
        return self._op(x, y)[0]


class ElementwiseAdd(Module):
    def __init__(self, name=None) -> None:
        super().__init__()
        self._op = flow.builtin_op("add_n", name).Input("in", 2).Output("out").Build()

    def forward(self, x, y):
        return self._op(x, y)[0]


class BroadcastAdd(Module):
    def __init__(self, name=None) -> None:
        super().__init__()
        self._op = (
            flow.builtin_op("broadcast_add", name)
            .Input("x")
            .Input("y")
            .Output("z")
            .Build()
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

        # Example
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
