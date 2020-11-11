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
import oneflow.typing as tp
from util import convert_to_onnx_and_check


def test_gather_nd(test_case):
    @flow.global_function()
    def gather_nd():
        x = flow.get_variable(
            name="x",
            shape=(2, 3, 4),
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(),
        )
        y = flow.get_variable(
            name="y",
            shape=(2, 3),
            dtype=flow.int64,
            initializer=flow.random_uniform_initializer(0, 1, flow.int64),
        )
        return flow.gather_nd(x, y)

    convert_to_onnx_and_check(gather_nd, opset=11)
