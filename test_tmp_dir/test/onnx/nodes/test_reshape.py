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


def test_reshape(test_case):
    @flow.global_function()
    def reshape(x: tp.Numpy.Placeholder((3, 4, 2, 5))):
        return flow.reshape(x, (4, 30))

    convert_to_onnx_and_check(reshape)
