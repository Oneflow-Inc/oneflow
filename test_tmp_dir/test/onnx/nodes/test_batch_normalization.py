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


def test_bn_nchw(test_case):
    @flow.global_function()
    def bn(x: tp.Numpy.Placeholder((3, 4, 2, 5))):
        params_shape = (4,)
        mean = flow.get_variable(
            name="mean",
            shape=params_shape,
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(),
        )
        variance = flow.get_variable(
            name="var",
            shape=params_shape,
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(),
        )
        gamma = flow.get_variable(
            name="gamma",
            shape=params_shape,
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(),
        )
        beta = flow.get_variable(
            name="beta",
            shape=params_shape,
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(),
        )
        return flow.nn.batch_normalization(x, mean, variance, beta, gamma, 1e-5, axis=1)

    convert_to_onnx_and_check(bn)


def test_bn_nhwc(test_case):
    @flow.global_function()
    def bn(x: tp.Numpy.Placeholder((3, 4, 2, 5))):
        params_shape = (5,)
        mean = flow.get_variable(
            name="mean",
            shape=params_shape,
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(),
        )
        variance = flow.get_variable(
            name="var",
            shape=params_shape,
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(),
        )
        gamma = flow.get_variable(
            name="gamma",
            shape=params_shape,
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(),
        )
        beta = flow.get_variable(
            name="beta",
            shape=params_shape,
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(),
        )
        return flow.nn.batch_normalization(
            x, mean, variance, beta, gamma, 1e-5, axis=-1
        )

    convert_to_onnx_and_check(bn)
