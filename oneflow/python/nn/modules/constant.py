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
from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.framework.tensor import register_tensor_op

from typing import Optional, Sequence, Union


class _ConstantBase(Module):
    def __init__(
        self, size: Sequence[int], value: Union[float, int], dtype: Optional[flow.dtype]
    ) -> None:
        super().__init__()
        assert size is not None, "shape must not be None!"
        assert isinstance(
            size, (int, list, tuple)
        ), "shape should be int, list or tuple!"

        if isinstance(size, int):
            size = [size]
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

        self.dtype = dtype
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
    r"""
    Returns a tensor filled with the scalar value 1, 
    with the shape defined by the variable argument `size`.

    Args:
        size (int...) – a sequence of integers defining the shape of the output tensor. 
        Can be a variable number of arguments or a collection like a list or tuple.

    For example:

    .. code-block:: python

        import oneflow as flow

        y = flow.ones(10)
        # [1 1 1 1 1 1 1 1 1 1]
        
    """

    def __init__(self, size, dtype=None):
        super().__init__(size, 1, dtype)


@oneflow_export("tmp.ones")
def ones_op(size, dtype=None):
    return Ones(size, dtype)()


class Zeros(_ConstantBase):
    r"""
    Returns a tensor filled with the scalar value 0, 
    with the shape defined by the variable argument `size`.

    Args:
        size (int...) – a sequence of integers defining the shape of the output tensor. 
        Can be a variable number of arguments or a collection like a list or tuple.

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np

        y = flow.zeros(5)

        # [0. 0. 0. 0. 0. ]

    """

    def __init__(self, size, dtype=None):
        super().__init__(size, 0, dtype)


@oneflow_export("tmp.zeros")
def zeros_op(size, dtype=None):
    return Zeros(size, dtype)()
