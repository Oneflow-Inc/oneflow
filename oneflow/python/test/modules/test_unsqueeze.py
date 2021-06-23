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

import oneflow.experimental as flow
from test_util import GenArgList


def _test_unsqueeze(test_case, device):
    np_arr = np.random.rand(2, 6, 9, 3)
    x = flow.Tensor(np_arr, device=flow.device(device))
    y = flow.unsqueeze(x, dim=1)
    output = np.expand_dims(np_arr, axis=1)
    test_case.assertTrue(np.allclose(output, y.numpy(), 1e-05, 1e-05))


def _test_unsqueeze_tensor_function(test_case, device):
    np_arr = np.random.rand(2, 3, 4)
    x = flow.Tensor(np_arr, device=flow.device(device))
    y = x.unsqueeze(dim=2)
    output = np.expand_dims(np_arr, axis=2)
    test_case.assertTrue(np.allclose(output, y.numpy(), 1e-05, 1e-05))


def _test_unsqueeze_different_dim(test_case, device):
    np_arr = np.random.rand(4, 5, 6, 7)
    x = flow.Tensor(np_arr, device=flow.device(device))
    for axis in range(-5, 5):
        y = flow.unsqueeze(x, dim=axis)
        output = np.expand_dims(np_arr, axis=axis)
        test_case.assertTrue(np.allclose(output, y.numpy(), 1e-05, 1e-05))


def _test_unsqueeze_backward(test_case, device):
    np_arr = np.random.rand(2, 3, 4, 5)
    x = flow.Tensor(np_arr, device=flow.device(device), requires_grad=True)
    y = flow.unsqueeze(x, dim=1).sum()
    y.backward()
    test_case.assertTrue(np.allclose(x.grad.numpy(), np.ones((2, 3, 4, 5)), 1e-5, 1e-5))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestUnsqueeze(flow.unittest.TestCase):
    def test_unsqueeze(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_unsqueeze,
            _test_unsqueeze_tensor_function,
            _test_unsqueeze_different_dim,
            _test_unsqueeze_backward,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
