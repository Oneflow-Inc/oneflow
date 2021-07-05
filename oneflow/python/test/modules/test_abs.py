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


def _test_abs_forward(test_case, device):
    input = flow.Tensor(np.random.randn(2, 3).astype(np.float32))
    of_out = flow.abs(input)
    np_out = np.abs(input.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))
    test_case.assertTrue(np.allclose(input.abs().numpy(), np_out, 1e-5, 1e-5))


def _test_abs_tensor_function_forward(test_case, device):
    x = np.random.randn(2, 3).astype(np.float32)
    input = flow.Tensor(x, dtype=flow.float32)
    np_out = np.abs(x)
    of_out = input.abs()
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))


def _test_abs_backward(test_case, device):
    np_input = np.random.randn(2, 3).astype(np.float32)
    input = flow.Tensor(np_input, dtype=flow.float32, requires_grad=True)
    of_out = flow.abs(input).sum()
    of_out.backward()
    np_grad = np.where(np_input > 0, 1, -1)
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-5, 1e-5))


def _test_abs_tensor_function_backward(test_case, device):
    np_input = np.random.randn(2, 3).astype(np.float32)
    input = flow.Tensor(np_input, dtype=flow.float32, requires_grad=True)
    of_out = input.abs().sum()
    of_out.backward()
    np_grad = np.where(np_input > 0, 1, -1)
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-5, 1e-5))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestAbs(flow.unittest.TestCase):
    def test_cosh(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_abs_forward,
            _test_abs_tensor_function_forward,
            _test_abs_backward,
            _test_abs_tensor_function_backward,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
