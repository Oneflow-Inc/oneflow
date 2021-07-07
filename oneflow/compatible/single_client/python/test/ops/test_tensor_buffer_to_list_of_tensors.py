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
import numpy as np
from collections import OrderedDict

import oneflow as flow
from test_util import GenArgList, type_name_to_flow_type


def _run_test(shape, shape_list, value_list, data_type):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_logical_view(flow.scope.consistent_view())
    func_config.default_data_type(flow.float)

    @flow.global_function(function_config=func_config)
    def TestTensorBufferToListOfTensorsJob():
        with flow.scope.placement("cpu", "0:0"):
            x = flow.gen_tensor_buffer(
                shape, shape_list, value_list, type_name_to_flow_type[data_type]
            )
            y = flow.tensor_buffer_to_list_of_tensors(
                x, (100, 100), type_name_to_flow_type[data_type], True
            )
            return y

    out_0, out_1, out_2, out_3 = TestTensorBufferToListOfTensorsJob().get()
    assert np.array_equal(out_0.numpy_list()[0], np.zeros((10, 10), np.float))
    assert np.array_equal(out_1.numpy_list()[0], np.ones((50, 50), np.float))
    assert np.array_equal(out_2.numpy_list()[0], np.ones((20, 80), np.float) * 2.0)
    assert np.array_equal(out_3.numpy_list()[0], np.ones((100, 100), np.float) * 3.0)


def gen_arg_list():
    arg_dict = OrderedDict()
    arg_dict["shape"] = [(2, 2), (4,)]
    arg_dict["shape_list"] = [[(10, 10), (50, 50), (20, 80), (100, 100)]]
    arg_dict["value_list"] = [[0.0, 1.0, 2.0, 3.0]]
    arg_dict["data_type"] = ["float32", "double", "int32", "int64"]

    return GenArgList(arg_dict)


@flow.unittest.skip_unless_1n1d()
class TestTensorBufferToListOfTensors(flow.unittest.TestCase):
    def test_tensor_buffer_to_list_of_tensors(test_case):
        for arg in gen_arg_list():
            _run_test(*arg)


if __name__ == "__main__":
    unittest.main()
