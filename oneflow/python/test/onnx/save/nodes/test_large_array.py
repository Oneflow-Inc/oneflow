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
from oneflow.python.test.onnx.save.util import convert_to_onnx_and_check


def test_large_array(test_case):
    @flow.global_function()
    def add_with_large_array():
        large_shape = (256 * 1024 * 1024 + 1,)
        x = flow.get_variable(
            name="x",
            shape=large_shape,
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(),
        )
        y = flow.get_variable(
            name="y",
            shape=large_shape,
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(),
        )
        return flow.math.add_n([x, y])

    # ONNX Runtime optimizers doesn't support external data
    convert_to_onnx_and_check(
        add_with_large_array, external_data=True, ort_optimize=False
    )
