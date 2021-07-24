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

from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.framework.tensor import register_tensor_op
from oneflow.compatible.single_client.python.nn.module import Module
from oneflow.compatible.single_client.python.nn.modules.utils import _check_axis
from oneflow.compatible.single_client.python.ops.transpose_util import (
    get_inversed_perm,
    get_perm_when_transpose_axis_to_last_dim,
)


class ScalarMul(Module):
    def __init__(self, alpha) -> None:
        super().__init__()
        if not isinstance(alpha, (int, float)):
            raise ValueError("alpha type can only be int or float")
        self.alpha = alpha

    def forward(self, x):
        return flow.F.mul_scalar(x, self.alpha)


class ScalarMulByTensor(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, y):
        return flow.F.mul_scalar_by_tensor(x, y)


class ElementwiseMul(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, y):
        return flow.F.mul(x, y)


class BroadcastMul(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, y):
        return flow.F.broadcast_mul(x, y)


class Variance(Module):
    def __init__(self, dim: int = None, keepdim: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, input):
        axis = _check_axis(self.dim, input.shape)
        if isinstance(axis, list) and len(axis) == 0:
            return flow.experimental.zeros(size=input.shape)
        else:
            return flow.experimental.sub(
                flow.experimental.mean(
                    flow.experimental.square(input), axis, self.keepdim
                ),
                flow.experimental.square(
                    flow.experimental.mean(input, axis, self.keepdim)
                ),
            )


class ScalarSubByTensor(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, y):
        return flow.F.sub_scalar_by_tensor(x, y)


class BroadcastSub(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, y):
        return flow.F.broadcast_sub(x, y)


class ScalarAdd(Module):
    def __init__(self, alpha) -> None:
        super().__init__()
        if not isinstance(alpha, int) and (not isinstance(alpha, float)):
            raise ValueError("scalar type can only be int or float")
        self.alpha = alpha

    def forward(self, x):
        return flow.F.add_scalar(x, self.alpha)


class BroadcastDiv(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, y):
        return flow.F.broadcast_div(x, y)


class ScalarDivByTensor(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, scalar):
        return flow.F.div_scalar_by_tensor(x, scalar)


class Reciprocal(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return flow.F.reciprocal_no_nan(x)


class ScalarAddByTensor(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, y):
        return flow.F.add_scalar_by_tensor(x, y)


class ElementwiseAdd(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, y):
        return flow.F.add(x, y)


class BroadcastAdd(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, y):
        return flow.F.broadcast_add(x, y)


class Asin(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return flow.F.asin(x)


class Asinh(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return flow.F.asinh(x)


class Sin(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return flow.F.sin(x)


class Cos(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return flow.F.cos(x)


class Atan(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return flow.F.atan(x)


class Log(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return flow.F.log(x)


class Subtract(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, y):
        if isinstance(x, (int, float)):
            return ScalarAdd(x)(-1 * y)
        elif isinstance(y, (int, float)):
            return ScalarAdd(-1 * y)(x)
        elif x.shape == y.shape:
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

    def forward(self, input):
        return flow.F.sqrt(input)


class Rsqrt(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        return flow.F.rsqrt(input)


class Square(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        return flow.F.square(input)


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
            return flow.experimental.zeros(size=x.shape)
        else:
            if len(self.axis) == 0:
                self.reduce_count = x.nelement()
            else:
                for i in self.axis:
                    self.reduce_count *= x.shape[i]
            sum = (
                flow.experimental.sum(self.square_op(x), self.axis, self.keepdim)
                / self.reduce_count
            )
            square = self.square_op(
                flow.experimental.sum(x, self.axis, self.keepdim) / self.reduce_count
            )
            subtract = self.subtract_op(sum, square)
            res = self.sqrt_op(subtract)
            return res


class Pow(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, y):
        if isinstance(y, (int, float)):
            return flow.F.pow_scalar(x, alpha=y)
        else:
            return flow.F.pow(x, y)


class Addmm(Module):
    def __init__(self) -> None:
        super().__init__()
        self._matmul_op = (
            flow.builtin_op("matmul")
            .Input("a")
            .Input("b")
            .Output("out")
            .Attr("transpose_a", False)
            .Attr("transpose_b", False)
            .Attr("alpha", 1.0)
            .Build()
        )

    def forward(self, x, mat1, mat2, alpha=1, beta=1):
        if len(x.shape) > 2 or len(mat1.shape) > 2 or len(mat2.shape) > 2:
            raise ValueError("input matrixes shape can not be greater than 2")
        else:
            return _mul(x, beta) + _mul(self._matmul_op(mat1, mat2)[0], alpha)


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


class Cosh(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return flow.F.cosh(x)


class Erf(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        return flow.F.erf(input)


class Erfc(Module):
    def __init__(self) -> None:
        super().__init__()
        self.erfc_op = flow.builtin_op("erfc").Input("x").Output("y").Build()

    def forward(self, input):
        return self.erfc_op(input)[0]


class Ceil(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return flow.F.ceil(x)


class Expm1(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return flow.F.expm1(x)


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
                neg_input = flow.experimental.mul(input, -1)
                indices = self._op_topk_last_dim(neg_input)[0]
            return (flow.experimental.gather(input, indices, dim=axis), indices)
        else:
            perm = get_perm_when_transpose_axis_to_last_dim(num_axes, axis)
            x = flow.F.transpose(input, perm=perm)
            if self.largest:
                indices = self._op_topk_last_dim(x)[0]
            else:
                neg_input = flow.experimental.mul(x, -1)
                indices = self._op_topk_last_dim(neg_input)[0]
            indices = flow.F.transpose(indices, perm=get_inversed_perm(perm))
            return (flow.experimental.gather(input, indices, dim=axis), indices)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
