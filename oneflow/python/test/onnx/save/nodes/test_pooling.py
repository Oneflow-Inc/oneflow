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


def test_max_pooling_2d_k3s1_valid_nhwc(test_case):
    @flow.global_function()
    def max_pooling_2d_k3s1_valid_nhwc(x: tp.Numpy.Placeholder((2, 3, 5, 4))):
        x += flow.get_variable(
            name="v1",
            shape=(1, 1),
            dtype=flow.float,
            initializer=flow.zeros_initializer(),
        )
        return flow.nn.max_pool2d(
            x, ksize=3, strides=1, padding="VALID", data_format="NHWC"
        )

    convert_to_onnx_and_check(max_pooling_2d_k3s1_valid_nhwc)


def test_max_pooling_2d_k3s1_same_nhwc(test_case):
    @flow.global_function()
    def max_pooling_2d_k3s1_same_nhwc(x: tp.Numpy.Placeholder((2, 3, 5, 4))):
        x += flow.get_variable(
            name="v1",
            shape=(1, 1),
            dtype=flow.float,
            initializer=flow.zeros_initializer(),
        )
        return flow.nn.max_pool2d(
            x, ksize=3, strides=1, padding="SAME", data_format="NHWC"
        )

    convert_to_onnx_and_check(max_pooling_2d_k3s1_same_nhwc)


def test_max_pooling_2d_k2s2_same_nhwc(test_case):
    @flow.global_function()
    def max_pooling_2d_k2s2_same_nhwc(x: tp.Numpy.Placeholder((2, 3, 5, 4))):
        x += flow.get_variable(
            name="v1",
            shape=(1, 1),
            dtype=flow.float,
            initializer=flow.zeros_initializer(),
        )
        return flow.nn.max_pool2d(
            x, ksize=2, strides=2, padding="SAME", data_format="NHWC"
        )

    convert_to_onnx_and_check(max_pooling_2d_k2s2_same_nhwc)


def test_max_pooling_2d_k2s2_same_nchw(test_case):
    @flow.global_function()
    def max_pooling_2d_k2s2_same_nchw(x: tp.Numpy.Placeholder((2, 3, 5, 4))):
        x += flow.get_variable(
            name="v1",
            shape=(1, 1),
            dtype=flow.float,
            initializer=flow.zeros_initializer(),
        )
        return flow.nn.max_pool2d(
            x, ksize=2, strides=2, padding="SAME", data_format="NCHW"
        )

    convert_to_onnx_and_check(max_pooling_2d_k2s2_same_nchw)


def test_max_pooling_2d_k3s1_valid_nchw(test_case):
    @flow.global_function()
    def max_pooling_2d_k3s1_valid_nchw(x: tp.Numpy.Placeholder((2, 3, 5, 4))):
        x += flow.get_variable(
            name="v1",
            shape=(1, 1),
            dtype=flow.float,
            initializer=flow.zeros_initializer(),
        )
        return flow.nn.max_pool2d(
            x, ksize=3, strides=1, padding="VALID", data_format="NCHW"
        )

    convert_to_onnx_and_check(max_pooling_2d_k3s1_valid_nchw)


def test_avg_pooling_2d_k3s1_valid_nhwc(test_case):
    @flow.global_function()
    def avg_pooling_2d_k3s1_valid_nhwc(x: tp.Numpy.Placeholder((2, 3, 5, 4))):
        x += flow.get_variable(
            name="v1",
            shape=(1, 1),
            dtype=flow.float,
            initializer=flow.zeros_initializer(),
        )
        return flow.nn.avg_pool2d(
            x, ksize=3, strides=1, padding="VALID", data_format="NHWC"
        )

    convert_to_onnx_and_check(avg_pooling_2d_k3s1_valid_nhwc)


def test_avg_pooling_2d_k3s1_same_nhwc(test_case):
    @flow.global_function()
    def avg_pooling_2d_k3s1_same_nhwc(x: tp.Numpy.Placeholder((2, 3, 5, 4))):
        x += flow.get_variable(
            name="v1",
            shape=(1, 1),
            dtype=flow.float,
            initializer=flow.zeros_initializer(),
        )
        return flow.nn.avg_pool2d(
            x, ksize=3, strides=1, padding="SAME", data_format="NHWC"
        )

    convert_to_onnx_and_check(avg_pooling_2d_k3s1_same_nhwc)


def test_avg_pooling_2d_k2s2_same_nhwc(test_case):
    @flow.global_function()
    def avg_pooling_2d_k2s2_same_nhwc(x: tp.Numpy.Placeholder((2, 3, 5, 4))):
        x += flow.get_variable(
            name="v1",
            shape=(1, 1),
            dtype=flow.float,
            initializer=flow.zeros_initializer(),
        )
        return flow.nn.avg_pool2d(
            x, ksize=2, strides=2, padding="SAME", data_format="NHWC"
        )

    convert_to_onnx_and_check(avg_pooling_2d_k2s2_same_nhwc)


def test_avg_pooling_2d_k3s1_valid_nchw(test_case):
    @flow.global_function()
    def avg_pooling_2d_k3s1_valid_nchw(x: tp.Numpy.Placeholder((2, 3, 5, 4))):
        x += flow.get_variable(
            name="v1",
            shape=(1, 1),
            dtype=flow.float,
            initializer=flow.zeros_initializer(),
        )
        return flow.nn.avg_pool2d(
            x, ksize=3, strides=1, padding="VALID", data_format="NCHW"
        )

    convert_to_onnx_and_check(avg_pooling_2d_k3s1_valid_nchw)


def test_avg_pooling_2d_k2s2_same_nchw(test_case):
    @flow.global_function()
    def avg_pooling_2d_k2s2_same_nchw(x: tp.Numpy.Placeholder((2, 3, 5, 4))):
        x += flow.get_variable(
            name="v1",
            shape=(1, 1),
            dtype=flow.float,
            initializer=flow.zeros_initializer(),
        )
        return flow.nn.avg_pool2d(
            x, ksize=2, strides=2, padding="SAME", data_format="NCHW"
        )

    convert_to_onnx_and_check(avg_pooling_2d_k2s2_same_nchw)
