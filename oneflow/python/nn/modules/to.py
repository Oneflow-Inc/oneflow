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
        if dtype:
            if self.copy:
                x = self._copy_op(x, device_type=device.type, device_id=device.index)[0]
            x = self._cast_op(x, dtype=dtype)[0]

        if x.device == device and not self.copy:
            return x
        return self._copy_op(x, device_type=device.type, device_id=device.index)[0]


@oneflow_export("to")
@register_tensor_op("to")
def to_op(
    input, device: Optional[Union[str, flow.device]] = None, dtype=None, copy=False
):
    r"""Performs Tensor dtype and/or device conversion. 
        A flow.dtype and flow.device are inferred from the arguments of `self.to(*args, **kwargs)`.
    
    .. note::

    If the ``self`` Tensor already
    has the correct :class:`flow.dtype` and :class:`flow.device`, then ``self`` is returned.
    Otherwise, the returned tensor is a copy of ``self`` with the desired
    :class:`flow.dtype` and :class:`flow.device`.


    Args:
        input (oneflow.Tensor): A input tensor.
        device (flow.device, optional) : Device type of the tensor you want move to.
        dtype (flow.dtype, optional) : Data type of the tensor you want cast to.
        copy (bool) : whether construct a new tensor or reuse this one.default to `False`
    
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
    if device is None:
        device = flow.device("cpu")
    if isinstance(device, str):
        device = flow.device(device)
    return To(copy)(input, device, dtype)
