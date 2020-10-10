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


def test_bias_add_nchw(test_case):
    @flow.global_function()
    def bias_add_nchw(x: tp.Numpy.Placeholder((3, 4, 2, 5))):
        y = flow.get_variable(
            name="y",
            shape=(4,),
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(),
        )
        return flow.nn.bias_add(x, y, "NCHW")

    convert_to_onnx_and_check(bias_add_nchw)


def test_bias_add_nhwc(test_case):
    @flow.global_function()
    def bias_add_nhwc(x: tp.Numpy.Placeholder((3, 4, 2, 5))):
        y = flow.get_variable(
            name="y",
            shape=(5,),
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(),
        )
        return flow.nn.bias_add(x, y, "NHWC")

    convert_to_onnx_and_check(bias_add_nhwc)
