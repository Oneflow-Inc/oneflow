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

from typing import Optional, Sequence, Sized, Union
import collections
from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.nn.module import Module
from oneflow.python.nn.modules.utils import (
    _single,
    _pair,
    _triple,
    _reverse_repeat_tuple,
)
from oneflow.python.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple
from oneflow.python.ops.nn_ops import calc_pool_padding, get_dhw_offset
import oneflow.python.framework.id_util as id_util
from oneflow.python.framework.tensor import register_tensor_op_by_module


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


@oneflow_export("Sum")
class Sum(Module):
    r"""
    """

    def __init__(
        self,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: bool = False,
        name: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.axis = axis
        self._op = (
            flow.builtin_op("reduce_sum", name)
            .Input("input_tensor")
            .Output("output_tensor")
            .Attr("keepdims", keepdims)
        )

    def forward(self, input):
        axis = _check_axis(self.axis, input.shape)

        if len(axis) == 0:
            return input

        self._op = self._op.Attr("axis", axis).Build()

        return self._op(input)[0]


class ScalarMul(Module):
    def __init__(self, operand, name=None) -> None:
        super().__init__()
        self._op = flow.builtin_op("scalar_mul", name).Input("in").Output("out")

        if isinstance(operand, int):
            self._op = (
                self._op.Attr("has_int_operand", True)
                .Attr("has_float_operand", False)
                .Attr("int_operand", operand)
                .Attr("float_operand", 0.0)
            )
        elif isinstance(operand, float):
            self._op = (
                self._op.Attr("has_int_operand", False)
                .Attr("has_float_operand", True)
                .Attr("int_operand", 0)
                .Attr("float_operand", operand)
            )
        else:
            raise ValueError("operand type can only be int or float")

        self._op = self._op.Build()

    def forward(self, x):
        return self._op(x)[0]


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


@register_tensor_op_by_module("mul")
@oneflow_export("Mul")
class Mul(Module):
    r"""Compute :math:`x \times y`.

    The equation is:

    .. math::
        out = x \times y

    For example:

    .. code-block:: python

        # Example
        mul = flow.Mul()

        # element-wise multiply
        x = flow.Tensor(np.random.randn(2,3))
        y = flow.Tensor(np.random.randn(2,3))
        out = mul(x,y).numpy()
        print(out.shape) # (2,3)

        # scalar mutiply
        x = 5
        y = flow.Tensor(np.random.randn(2,3))
        out = mul(x,y).numpy()
        print(out.shape) # (2,3)

        # broadcast mutiply
        x = flow.Tensor(np.random.randn(1,1))
        y = flow.Tensor(np.random.randn(2,3))
        out = mul(x,y).numpy()
        print(out.shape) # (2,3)

    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, y):
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


@oneflow_export("Mean")
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
        reduce_sum = flow.Sum(axis=self.axis, keepdims=self.keepdims, name=self.name)(
            input_tensor
        )

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
        return flow.Mul()(reduce_sum, 1.0 / reduce_count)


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
            )
        elif isinstance(operand, float):
            self._op = (
                self._op.Attr("has_int_operand", False)
                .Attr("has_float_operand", True)
                .Attr("int_operand", 0)
                .Attr("float_operand", operand)
            )
        else:
            raise ValueError("operand type can only be int or float")

        self._op = self._op.Build()

    def forward(self, x):
        return self._op(x)[0]


@oneflow_export("Sub")
class Sub(Module):
    r"""

    """

    def __init__(self) -> None:
        super().__init__()
        pass

    def forward(self, x, y):
        if isinstance(x, (int, float)):
            return ScalarAdd(x)(ScalarMul(-1)(y))
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


@oneflow_export("Div")
class Div(Module):
    r"""Computes the division of x by y.

    The formula is:

    .. math::
        out = \frac{X}{Y}

    Args:
        x (Union[int, float, flow.Tensor]): X.
        y (Union[int, float, flow.Tensor]): Y.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    For example:

    .. code-block:: python

        # out [2.5, 4., 4.5]

    """

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__()
        self.name = name

    def forward(self, x, y):
        if isinstance(x, (int, float)):
            return ScalarMul(x)(flow.Reciprocal()(y))
            # return scalar_mul(math_unary_elementwise_ops.reciprocal_no_nan(y), x, name)
        elif isinstance(y, (int, float)):
            if y == 0 or y == 0.0:
                y = 0.0
            else:
                y = 1.0 / (float(y))
            return ScalarMul(y)(x)
        elif x.shape == y.shape:
            # TODO: add element-wise op
            return BroadcastDiv()(x, y)
            # return broadcast_div(x, y, name)
        elif x.shape == (1,):
            return ScalarDivByTensor(y, x)
        elif y.shape == (1,):
            return ScalarDivByTensor(x, y)
            # return scalar_div_by_tensor(x, y, name)
        else:
            return BroadcastDiv()(x, y)
            # broadcast_div(x, y, name)


@oneflow_export("Reciprocal")
class Reciprocal(Module):
    r"""Computes the safe reciprocal of x. If x is zero, the reciprocal will 
    be also set to zero.

    Args:
        name (Optional[str], optional): The name for the operation. Defaults to None.

    For example: 

    .. code-block:: python 
        # out [0.   0.5  0.25]
    """

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__()
        self._op = (
            flow.builtin_op("reciprocal_no_nan", name)
            .Input("x")
            .Output("y")
            .Build()
        )

    def forward(self, x):
        return self._op(x)[0]
