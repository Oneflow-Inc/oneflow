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

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)

initializer = flow.random_uniform_initializer()
initer_args = {"kernel_initializer": initializer, "bias_initializer": initializer}


def test_conv2d_k2d1_valid(test_case):
    @flow.global_function(func_config)
    def conv2d_k3s1_valid(x=flow.FixedTensorDef((2, 4, 3, 5))):
        return flow.layers.conv2d(
            x, 6, kernel_size=3, strides=1, padding="VALID", **initer_args
        )

    convert_to_onnx_and_check(conv2d_k3s1_valid)


def test_conv2d_s2_valid(test_case):
    @flow.global_function(func_config)
    def conv2d_s2_valid(x=flow.FixedTensorDef((2, 4, 3, 5))):
        return flow.layers.conv2d(
            x, 6, kernel_size=1, strides=2, padding="VALID", **initer_args
        )

    convert_to_onnx_and_check(conv2d_s2_valid)


def test_conv2d_s2_same(test_case):
    @flow.global_function(func_config)
    def conv2d_s2_same(x=flow.FixedTensorDef((2, 4, 3, 5))):
        return flow.layers.conv2d(
            x, 6, kernel_size=3, strides=2, padding="SAME", **initer_args
        )

    convert_to_onnx_and_check(conv2d_s2_same)


def test_conv2d_k3s1_nhwc_valid(test_case):
    @flow.global_function(func_config)
    def conv2d_k3s1_nhwc_valid(x=flow.FixedTensorDef((2, 3, 5, 4))):
        return flow.layers.conv2d(
            x,
            6,
            kernel_size=3,
            strides=1,
            padding="VALID",
            data_format="NHWC",
            **initer_args
        )

    convert_to_onnx_and_check(conv2d_k3s1_nhwc_valid)


def test_conv2d_k3s1_nhwc_same_d2(test_case):
    @flow.global_function(func_config)
    def conv2d(x=flow.FixedTensorDef((2, 7, 5, 4))):
        return flow.layers.conv2d(
            x,
            6,
            kernel_size=3,
            strides=1,
            dilation_rate=2,
            padding="SAME",
            data_format="NHWC",
            **initer_args
        )

    convert_to_onnx_and_check(conv2d)


def test_conv2d_k3s1_nchw_same_g2(test_case):
    @flow.global_function(func_config)
    def conv2d(x=flow.FixedTensorDef((2, 4, 5, 3))):
        return flow.layers.conv2d(
            x,
            6,
            kernel_size=3,
            strides=1,
            groups=2,
            padding="SAME",
            data_format="NCHW",
            **initer_args
        )

    convert_to_onnx_and_check(conv2d)


def test_conv2d_k3s1_nchw_same_depthwise(test_case):
    @flow.global_function(func_config)
    def conv2d(x=flow.FixedTensorDef((2, 4, 5, 3))):
        return flow.layers.conv2d(
            x,
            4,
            kernel_size=3,
            strides=1,
            groups=4,
            padding="SAME",
            data_format="NCHW",
            **initer_args
        )

    convert_to_onnx_and_check(conv2d)
