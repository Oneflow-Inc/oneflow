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
import unittest
from collections import OrderedDict

import numpy as np
import oneflow as flow
from test_util import GenArgList


def _test_tensor_buffer_to_list_of_tensors(shape, shape_list, value_list):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_logical_view(flow.scope.mirrored_view())
    func_config.default_data_type(flow.float)

    @flow.global_function(function_config=func_config)
    def TestTensorBufferToListOfTensorsJob():
        with flow.scope.placement("cpu", "0:0"):
            x = flow.gen_tensor_buffer(shape, shape_list, value_list)
            y = flow.tensor_buffer_to_list_of_tensors(x, (100, 100), flow.float)
            return y
    
    x1, x2, x3, x4 = TestTensorBufferToListOfTensorsJob().get()
    print(x1.numpy_list()[0])
    print(x2.numpy_list()[0])
    print(x3.numpy_list()[0])
    print(x4.numpy_list()[0])


@flow.unittest.skip_unless_1n1d()
class TestTensorBufferToListOfTensors(flow.unittest.TestCase):
    _test_tensor_buffer_to_list_of_tensors((2, 2), [(10, 10), (100, 100), (10, 10), (100, 100)], [0.0, 1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
