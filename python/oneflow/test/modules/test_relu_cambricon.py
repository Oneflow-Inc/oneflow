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
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest


def np_relu(x):
    return np.where(x > 0, x, 0)

def _test_relu_forward(test_case, shape, device, dtype):
    arr = np.random.randn(*shape)
    x = flow.tensor(
        arr, device=flow.device(device), dtype=dtype
    )
    of_out = flow.relu(x)
    np_out = np_relu(arr).astype(arr.dtype)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.001, 0.001))


@flow.unittest.skip_unless_1n1d()
class TestReluCambriconModule(flow.unittest.TestCase):
    def test_relu(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_relu_forward,
        ]
        arg_dict["shape"] = [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]
        arg_dict["device"] = ["mlu"]
        arg_dict["dtype"] = [flow.float16, flow.float32]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

if __name__ == "__main__":
    unittest.main()
