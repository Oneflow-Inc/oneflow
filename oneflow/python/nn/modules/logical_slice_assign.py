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


class LogicalSliceAssign(Module):
    def __init__(self, value, starts, stops, steps) -> None:
        super().__init__()
        self._op = (
            flow.builtin_op("logical_slice_assign")
            .Input("ref")
            .Input("value", value)
            .Attr("start", starts)
            .Attr("stop", stops)
            .Attr("step", steps)
            .Build()
        )

    def forward(self, x):
        return self._op(x)[0]


@oneflow_export("logical_slice_assign")
def logical_slice_assign_op(x, value, starts, stops, steps):
    return LogicalSliceAssign(value=value, starts=starts, stops=stops, steps=steps)(x)
