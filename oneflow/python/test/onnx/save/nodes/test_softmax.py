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


def test_softmax(test_case):
    @flow.global_function()
    def softmax(x: tp.Numpy.Placeholder((3, 5))):
        return flow.nn.softmax(x)

    convert_to_onnx_and_check(softmax)


def test_softmax_with_axis(test_case):
    @flow.global_function()
    def softmax(x: tp.Numpy.Placeholder((3, 5, 4))):
        return flow.nn.softmax(x, axis=1)

    convert_to_onnx_and_check(softmax)
