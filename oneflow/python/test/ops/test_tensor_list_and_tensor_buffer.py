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
import oneflow as flow
import oneflow.typing as oft


def _of_tensor_list_to_tensor_buffer(test_case, verbose=False):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_logical_view(flow.scope.mirrored_view())

    @flow.global_function(function_config=func_config)
    def job_fn(x_def: oft.ListListNumpy.Placeholder(shape=(2, 5, 4), dtype=flow.float)):
        x = flow.tensor_list_to_tensor_buffer(x_def)
        return flow.tensor_buffer_to_tensor_list(x, shape=(5, 4), dtype=flow.float)

    input_1 = np.random.rand(1, 3, 4).astype(np.float32)
    input_2 = np.random.rand(1, 2, 4).astype(np.float32)
    ret = job_fn([[input_1, input_2]]).get()
    ret_arr_list = ret.numpy_lists()

    if verbose:
        print("input_1 =", input_1)
        print("input_2 =", input_2)
        print("ret_arr_list =", ret_arr_list)

    test_case.assertTrue(np.array_equal(input_1, ret_arr_list[0][0]))
    test_case.assertTrue(np.array_equal(input_2, ret_arr_list[0][1]))


@flow.unittest.skip_unless_1n1d()
class TestTensorListAndTensorBuffer(flow.unittest.TestCase):
    def test_tensor_list_and_tensor_buffer_conversion(test_case):
        _of_tensor_list_to_tensor_buffer(test_case)


if __name__ == "__main__":
    unittest.main()
