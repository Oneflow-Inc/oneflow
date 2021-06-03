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


def _test_acos_forward(test_case, device):
    input = flow.Tensor(np.random.randn(2, 6, 5, 3), device=flow.device(device))
    of_out = flow.acos(input)
    np_out = np.arccos(input.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5, equal_nan=True))


def _test_acos_tensor_function_forward(test_case, device):
    input = flow.Tensor(np.random.randn(8, 11, 9, 7), device=flow.device(device))
    of_out = input.acos()
    np_out = np.arccos(input.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5, equal_nan=True))


def _test_acos_backward(test_case, device):
    input = flow.Tensor(
        np.random.randn(8, 11, 9, 7), requires_grad=True, device=flow.device(device)
    )
    of_out = flow.acos(input).sum()
    of_out.backward()
    np_grad = -1.0 / np.sqrt( 1 - np.square(input.numpy()))
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-4, 1e-4, equal_nan=True))


def _test_acos_tensor_function_backward(test_case, device):
    input = flow.Tensor(
        np.random.randn(8, 11, 9, 7), requires_grad=True, device=flow.device(device)
    )
    of_out = input.acos().sum()
    of_out.backward()
    np_grad = -1.0 / np.sqrt(1 - np.square(input.numpy()))
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-4, 1e-4, equal_nan=True))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestAcos(flow.unittest.TestCase):
    def test_acos(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_acos_forward,
            _test_acos_tensor_function_forward,
            _test_acos_backward,
            _test_acos_tensor_function_backward,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
    