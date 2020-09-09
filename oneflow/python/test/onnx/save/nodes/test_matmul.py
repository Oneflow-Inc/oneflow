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
from util import convert_to_onnx_and_check


def test_matmul(test_case):
    @flow.global_function()
    def matmul():
        a = flow.get_variable(
            name="a",
            shape=(2, 3),
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(),
        )
        b = flow.get_variable(
            name="b",
            shape=(3, 4),
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(),
        )
        return flow.matmul(a, b)

    convert_to_onnx_and_check(matmul)


def test_matmul_ta(test_case):
    @flow.global_function()
    def matmul():
        a = flow.get_variable(
            name="a",
            shape=(3, 2),
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(),
        )
        b = flow.get_variable(
            name="b",
            shape=(3, 4),
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(),
        )
        return flow.matmul(a, b, transpose_a=True)

    convert_to_onnx_and_check(matmul)


def test_matmul_tb(test_case):
    @flow.global_function()
    def matmul():
        a = flow.get_variable(
            name="a",
            shape=(2, 3),
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(),
        )
        b = flow.get_variable(
            name="b",
            shape=(4, 3),
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(),
        )
        return flow.matmul(a, b, transpose_b=True)

    convert_to_onnx_and_check(matmul)


def test_matmul_ta_tb(test_case):
    @flow.global_function()
    def matmul():
        a = flow.get_variable(
            name="a",
            shape=(3, 2),
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(),
        )
        b = flow.get_variable(
            name="b",
            shape=(4, 3),
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(),
        )
        return flow.matmul(a, b, transpose_a=True, transpose_b=True)

    convert_to_onnx_and_check(matmul)


def test_batch_matmul(test_case):
    @flow.global_function()
    def matmul():
        a = flow.get_variable(
            name="a",
            shape=(4, 2, 3),
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(),
        )
        b = flow.get_variable(
            name="b",
            shape=(4, 3, 4),
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(),
        )
        return flow.matmul(a, b)

    convert_to_onnx_and_check(matmul)
