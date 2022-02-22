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
from test_util import GenArgList

import oneflow as flow
import oneflow.nn as nn
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


def _test_triu(test_case, diagonal, device):
    arr_shape = (4, 4, 8)
    np_arr = np.random.randn(*arr_shape)
    input_tensor = flow.tensor(
        np_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    output = flow.triu(input_tensor, diagonal=diagonal)
    np_out = np.triu(np_arr, diagonal)
    test_case.assertTrue(np.allclose(output.numpy(), np_out, 1e-06, 1e-06))
    output = output.sum()
    output.backward()
    np_grad = np.triu(np.ones(shape=arr_shape, dtype=np.float32), diagonal)
    test_case.assertTrue(np.allclose(input_tensor.grad.numpy(), np_grad, 1e-06, 1e-06))


@flow.unittest.skip_unless_1n1d()
class TestTriu(flow.unittest.TestCase):
    def test_triu(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_triu]
        arg_dict["diagonal"] = [2, -1]
        arg_dict["device"] = ["cuda", "cpu"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @autotest(auto_backward=False, check_graph=True)
    def test_triu_with_0_size_data(test_case):
        device = random_device()
        x = random_tensor(4, 2, 1, 0, 3).to(device)
        y = torch.triu(x)
        return y


if __name__ == "__main__":
    unittest.main()
