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
from oneflow.python.nn.common_types import _size_any_t
from oneflow.python.nn.modules.utils import _single

from typing import Optional, Union


class _ConstantBase(Module):
    def __init__(
        self,
        size: Union[_size_any_t, flow.Size],
        value: Union[float, int],
        dtype: Optional[flow.dtype],
    ) -> None:
        super().__init__()
        assert size is not None, "shape must not be None!"
        assert isinstance(
            size, (int, tuple, flow.Size)
        ), "shape should be int or tuple int!"
        size = _single(size)
        if dtype is None:
            dtype = flow.float32

        if dtype in [
            flow.int,
            flow.int64,
            flow.int32,
            flow.char,
            flow.int8,
            flow.long,
            flow.uint8,
        ]:
            floating_value = float(0)
            integer_value = int(value)
            is_floating_value = False
        elif dtype in [
            flow.float32,
            flow.float,
            flow.double,
            flow.float64,
            flow.float16,
            flow.half,
        ]:
            floating_value = float(value)
            integer_value = int(0)
            is_floating_value = True
        else:
            raise NotImplementedError("Unsupport data type")

        self._op = (
            flow.builtin_op("constant")
            .Output("out")
            .Attr("is_floating_value", is_floating_value)
            .Attr("floating_value", floating_value)
            .Attr("integer_value", integer_value)
            .Attr("dtype", dtype)
            .Attr("shape", size)
            .Build()
        )

    def forward(self):
        return self._op()[0]


class Ones(_ConstantBase):
    def __init__(self, size, dtype=None):
        super().__init__(size, 1, dtype)


@oneflow_export("ones")
@experimental_api
def ones_op(size, dtype=None):
    r"""
    Returns a tensor filled with the scalar value 1,
    with the shape defined by the variable argument `size`.

    Args:
        size(an integer or tuple of integer values): defining the shape of the output tensor. Can be \
         a variable number of arguments or a collection like a list or tuple.

    For example:

    .. code-block:: python

        import oneflow.experimental as flow

        y = flow.ones(5)
        # [1. 1. 1. 1. 1. ]

    """
    return Ones(size, dtype)()


class Zeros(_ConstantBase):
    def __init__(self, size, dtype=None):
        super().__init__(size, 0, dtype)


@oneflow_export("zeros")
@experimental_api
def zeros_op(size, dtype=None):
    r"""
    Returns a tensor filled with the scalar value 0,
    with the shape defined by the variable argument `size`.

    Args:
        size(an integer or tuple of integer values): defining the shape of the output tensor. Can be \
         a variable number of arguments or a collection like a list or tuple.

    For example:

    .. code-block:: python

        import oneflow.experimental as flow

        y = flow.zeros(5)
        # [0. 0. 0. 0. 0. ]

    """
    return Zeros(size, dtype)()


class ZerosLike(Module):
    def __init__(self):
        super().__init__()
        self._op = flow.builtin_op("zero_like").Input("like").Output("out").Build()

    def forward(self, other):
        return self._op(other)[0]


@oneflow_export("zeros_like")
@experimental_api
def zeros_like_op(other):
    r"""
    Returns a tensor filled with the scalar value 0, with the same size as input.
    flow.zeros_like(input) is equivalent to flow.zeros(input.shape, dtype=input.dtype)

    Args:
        other(Tensor): The size of input will determine size of the output tensor.

    For example:

    .. code-block:: python

        import oneflow.experimental as flow
        import numpy as np

        x = flow.Tensor(np.random.rand([5]))
        y = flow.zeros_like(x)
        # [0. 0. 0. 0. 0. ]

    """
    return ZerosLike()(other)


class OnesLike(Module):
    def __init__(self):
        super().__init__()
        self._op = flow.builtin_op("ones_like").Input("like").Output("out").Build()

    def forward(self, other):
        return self._op(other)[0]


@oneflow_export("ones_like")
@experimental_api
def ones_like_op(other):
    r"""
    Returns a tensor filled with the scalar value 1, with the same size as input.
    flow.ones_like(input) is equivalent to flow.ones(input.shape, dtype=input.dtype)

    Args:
        other(Tensor): The size of input will determine size of the output tensor.

    For example:

    .. code-block:: python

        import oneflow.experimental as flow
        import numpy as np

        x = flow.Tensor(np.random.rand([5]))
        y = flow.ones_like(x)
        # [1. 1. 1. 1. 1. ]

    """
    return OnesLike()(other)
