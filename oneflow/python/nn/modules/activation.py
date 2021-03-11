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
from oneflow.python.nn.modules.utils import _wrapper
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("nn.ReLU")
class ReLU(Module):
    def __init__(self):
        super().__init__()
        self._op = (
            flow.builtin_op("relu").Name("relu").Input("in").Output("out").Build()
        )
        self._op = _wrapper(self._op)

    def forward(self, x):
        res = self._op(x)[0]
        return res
