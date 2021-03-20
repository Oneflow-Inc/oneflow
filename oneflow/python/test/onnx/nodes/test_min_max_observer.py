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
import numpy as np
import oneflow as flow
import oneflow.typing as tp
from util import convert_to_onnx_and_check


def test_min_max_observer_symmetric(test_case):
    @flow.global_function()
    def min_max_observer():
        x = flow.get_variable(
            name="x1",
            shape=(2, 3, 4),
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(-10, 10),
        )
        return flow.quantization.min_max_observer(x)[0]

    convert_to_onnx_and_check(min_max_observer, opset=11)


def test_min_max_observer_symmetric_zero_point(test_case):
    @flow.global_function()
    def min_max_observer():
        x = flow.get_variable(
            name="x1",
            shape=(2, 3, 4),
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(-10, 10),
        )
        return flow.quantization.min_max_observer(x)[1]

    convert_to_onnx_and_check(min_max_observer, opset=11, dtype=np.int8)


def test_min_max_observer_affine(test_case):
    @flow.global_function()
    def min_max_observer():
        x = flow.get_variable(
            name="x1",
            shape=(2, 3, 4),
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(-10, 10),
        )
        return flow.quantization.min_max_observer(x, quantization_scheme="affine")[0]

    convert_to_onnx_and_check(min_max_observer, opset=11)


def test_min_max_observer_affine_zero_point(test_case):
    @flow.global_function()
    def min_max_observer():
        x = flow.get_variable(
            name="x1",
            shape=(2, 3, 4),
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(-10, 10),
        )
        return flow.quantization.min_max_observer(x, quantization_scheme="affine")[1]

    convert_to_onnx_and_check(min_max_observer, opset=11, dtype=np.uint8)


def test_min_max_observer_symmetric_not_per_channel(test_case):
    @flow.global_function()
    def min_max_observer():
        x = flow.get_variable(
            name="x1",
            shape=(2, 3, 4),
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(-10, 10),
        )
        return flow.quantization.min_max_observer(x, per_layer_quantization=False)[0]

    convert_to_onnx_and_check(min_max_observer, opset=11)


def test_min_max_observer_symmetric_not_per_channel_zero_point(test_case):
    @flow.global_function()
    def min_max_observer():
        x = flow.get_variable(
            name="x1",
            shape=(2, 3, 4),
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(-10, 10),
        )
        return flow.quantization.min_max_observer(x, per_layer_quantization=False)[1]

    convert_to_onnx_and_check(min_max_observer, opset=11, dtype=np.int8)


def test_min_max_observer_affine_not_per_channel(test_case):
    @flow.global_function()
    def min_max_observer():
        x = flow.get_variable(
            name="x1",
            shape=(2, 3, 4),
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(-10, 10),
        )
        return flow.quantization.min_max_observer(
            x, per_layer_quantization=False, quantization_scheme="affine"
        )[0]

    convert_to_onnx_and_check(min_max_observer, opset=11)


def test_min_max_observer_affine_not_per_channel_zero_point(test_case):
    @flow.global_function()
    def min_max_observer():
        x = flow.get_variable(
            name="x1",
            shape=(2, 3, 4),
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(-10, 10),
        )
        return flow.quantization.min_max_observer(
            x, per_layer_quantization=False, quantization_scheme="affine"
        )[1]

    convert_to_onnx_and_check(min_max_observer, opset=11, dtype=np.uint8)


def test_min_max_observer_cambricon(test_case):
    @flow.global_function()
    def min_max_observer():
        x = flow.get_variable(
            name="x1",
            shape=(2, 3, 4),
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(-10, 10),
        )
        return flow.quantization.min_max_observer(x, quantization_formula="cambricon")[
            0
        ]

    convert_to_onnx_and_check(min_max_observer, opset=11)


def test_min_max_observer_cambricon_zero_point(test_case):
    @flow.global_function()
    def min_max_observer():
        x = flow.get_variable(
            name="x1",
            shape=(2, 3, 4),
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(-10, 10),
        )
        return flow.quantization.min_max_observer(x, quantization_formula="cambricon")[
            1
        ]

    convert_to_onnx_and_check(min_max_observer, opset=11, dtype=np.int8)

