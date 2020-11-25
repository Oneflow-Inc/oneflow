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


def _run_test(shape, shape_list, value_list):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_logical_view(flow.scope.mirrored_view())
    func_config.default_data_type(flow.float)

    @flow.global_function(function_config=func_config)
    def TestTensorBufferToListOfTensorsJob():
        with flow.scope.placement("cpu", "0:0"):
            x = flow.gen_tensor_buffer(shape, shape_list, value_list)
            y = flow.tensor_buffer_to_list_of_tensors(x, (100, 100), flow.float, True)
            return y

    out_0, out_1, out_2, out_3 = TestTensorBufferToListOfTensorsJob().get()
    assert np.array_equal(out_0.numpy_list()[0], np.zeros((10, 10), np.float))
    assert np.array_equal(out_1.numpy_list()[0], np.ones((50, 50), np.float))
    assert np.array_equal(out_2.numpy_list()[0], np.ones((20, 80), np.float) * 2.0)
    assert np.array_equal(out_3.numpy_list()[0], np.ones((100, 100), np.float) * 3.0)


@flow.unittest.skip_unless_1n1d()
class TestTensorBufferToListOfTensors(flow.unittest.TestCase):
    shape_list = [(10, 10), (50, 50), (20, 80), (100, 100)]
    value_list = [0.0, 1.0, 2.0, 3.0]
    _run_test((2, 2), shape_list, value_list)
    _run_test((4,), shape_list, value_list)


if __name__ == "__main__":
    unittest.main()
