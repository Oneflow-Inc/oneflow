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
def to_op(
    input, device: Optional[Union[str, flow.device]] = None, dtype=None, copy=False
):
    if isinstance(device, str):
        device = flow.device(device)
    return To(copy)(input, device, dtype)
