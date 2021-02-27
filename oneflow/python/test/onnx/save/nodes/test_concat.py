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


def test_concat_axis0(test_case):
    @flow.global_function()
    def concat():
        variables = []
        for i in range(4):
            variables.append(
                flow.get_variable(
                    name=str(i),
                    shape=(2, 3),
                    dtype=flow.float,
                    initializer=flow.random_uniform_initializer(),
                )
            )
        return flow.concat(variables, axis=0)

    convert_to_onnx_and_check(concat)


def test_concat_axis1(test_case):
    @flow.global_function()
    def concat():
        variables = []
        for i in range(4):
            variables.append(
                flow.get_variable(
                    name=str(i),
                    shape=(2, 3),
                    dtype=flow.float,
                    initializer=flow.random_uniform_initializer(),
                )
            )
        return flow.concat(variables, axis=1)

    convert_to_onnx_and_check(concat)
