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


def test_add_2(test_case):
    @flow.global_function()
    def add_2():
        x = flow.get_variable(
            name="x",
            shape=(2, 3),
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(),
        )
        y = flow.get_variable(
            name="y",
            shape=(2, 3),
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(),
        )
        return flow.math.add_n([x, y])

    convert_to_onnx_and_check(add_2)


def test_add_3(test_case):
    @flow.global_function()
    def add_3():
        x = flow.get_variable(
            name="x",
            shape=(2, 3),
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(),
        )
        y = flow.get_variable(
            name="y",
            shape=(2, 3),
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(),
        )
        z = flow.get_variable(
            name="z",
            shape=(2, 3),
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(),
        )
        return flow.math.add_n([x, y, z])

    convert_to_onnx_and_check(add_3)


def test_add_many(test_case):
    @flow.global_function()
    def add_many():
        variables = []
        for i in range(50):
            variables.append(
                flow.get_variable(
                    name=str(i),
                    shape=(2, 3),
                    dtype=flow.float,
                    initializer=flow.random_uniform_initializer(),
                )
            )
        return flow.math.add_n(variables)

    convert_to_onnx_and_check(add_many)
