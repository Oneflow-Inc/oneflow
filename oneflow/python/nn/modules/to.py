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
from oneflow.python.framework.tensor import register_tensor_op
from oneflow.python.oneflow_export import oneflow_export
from typing import Optional, Union


class To(Module):
    def __init__(self, copy):
        super().__init__()
        self._copy_op = flow.builtin_op("copy").Input("in").Output("out").Build()
        self._cast_op = flow.builtin_op("cast").Input("in").Output("out").Build()
        self.copy = copy

    def forward(self, x, device, dtype):
        result = x
        if device is not None:
            if x.device != device or self.copy:
                result = self._copy_op(
                    x, device_type=device.type, device_id=device.index
                )[0]
        if dtype is not None:
            if x.dtype != dtype or self.copy:
                result = self._cast_op(result, dtype=dtype)[0]
        return result


@oneflow_export("to")
@register_tensor_op("to")
def to_op(input, *args, **kwargs):
    r"""Performs Tensor dtype and/or device conversion. 
        A flow.dtype and flow.device are inferred from the arguments of `input.to(*args, **kwargs)`.
    
    .. note::
    If the ``input`` Tensor already
    has the correct :class:`flow.dtype` and :class:`flow.device`, then ``input`` is returned.
    Otherwise, the returned tensor is a copy of ``input`` with the desired
    :class:`flow.dtype` and :class:`flow.device`.
    Args:
        input (flow.Tensor): A input tensor.
        *args (flow.Tensor or flow.device or flow.dtype): Positional arguments
        **kwargs (flow.Tensor or flow.device or flow.dtype) : Key-value arguments
    
    For example:
    .. code-block:: python
        import oneflow.experimental as flow
        import numpy as np
        arr = np.random.randint(1, 9, size=(1, 2, 3, 4))
        input = flow.Tensor(arr)
        output = input.to(dtype=flow.float32)
        print(np.array_equal(arr.astype(np.float32), output.numpy()))
        # True
    """
    copy = kwargs["copy"] if "copy" in kwargs else False
    device = kwargs["device"] if "device" in kwargs else None
    dtype = kwargs["dtype"] if "dtype" in kwargs else None
    if len(args) > 0:
        if isinstance(args[0], flow.Tensor):
            if len(args) == 2:
                copy = args[1]
            return To(copy)(args[0].device, args[0].dtype)
        elif isinstance(args[0], flow.dtype):
            if len(args) == 2:
                copy = args[1]
            return To(copy)(input, None, args[0])
        else:
            device = flow.device(args[0]) if isinstance(args[0], str) else args[0]
            if len(args) > 1:
                dtype = args[1]
                assert isinstance(dtype, flow.dtype)
            if len(args) > 2:
                copy = args[2]
            assert isinstance(device, flow.device)
            return To(copy)(input, device, dtype)
    if isinstance(device, flow.device) or isinstance(dtype, flow.dtype):
        return To(copy)(input, device, dtype)
    raise TypeError("to() received an invalid combination of arguments")
