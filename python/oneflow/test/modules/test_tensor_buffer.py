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
from oneflow.test_utils.test_util import GenArgList, type_name_to_flow_type

import oneflow as flow
import oneflow.unittest


def _test_tensor_buffer_convert(test_case, device):
    input = flow.tensor(
        np.random.rand(16, 24, 32, 36), dtype=flow.float32, device=flow.device(device)
    )
    tensor_buffer = flow.tensor_to_tensor_buffer(input, instance_dims=2)
    orig_tensor = flow.tensor_buffer_to_tensor(
        tensor_buffer, dtype=flow.float32, instance_shape=[32, 36]
    )
    test_case.assertTrue(np.array_equal(input.numpy(), orig_tensor.numpy()))


@flow.unittest.skip_unless_1n1d()
class TestTensorBufferOps(flow.unittest.TestCase):
    def test_tensor_buffer_convert(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_tensor_buffer_convert]
        arg_dict["device"] = ["cpu"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
