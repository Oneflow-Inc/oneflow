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
import oneflow.typing as tp
from oneflow.python.test.modules.test_util import GenArgList


def _to_device(self, device):
    if device == "cpu":
        return self
    else:
        return self.to(device)


def _test_linear_no_bias(test_case, device):
    linear = flow.nn.Linear(3, 8, False)
    linear = _to_device(linear, device)
    input_arr = np.array(
        [
            [-0.94630778, -0.83378579, -0.87060891],
            [2.0289922, -0.28708987, -2.18369248],
            [0.35217619, -0.67095644, -1.58943879],
            [0.08086036, -1.81075924, 1.20752494],
            [0.8901075, -0.49976737, -1.07153746],
            [-0.44872912, -1.07275683, 0.06256855],
            [-0.22556897, 0.74798368, 0.90416439],
            [0.48339456, -2.32742195, -0.59321527],
        ],
        dtype=np.float32,
    )
    np_weight = np.ones((3, 8)).astype(np.float32)
    np_weight.fill(2.3)
    x = flow.Tensor(input_arr, device=flow.device(device))
    flow.nn.init.constant_(linear.weight, 2.3)
    of_out = linear(x)
    np_out = np.matmul(input_arr, np_weight)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))


def _test_linear_with_bias(test_case, device):
    linear = flow.nn.Linear(3, 8)
    linear = _to_device(linear, device)
    input_arr = np.array(
        [
            [-0.94630778, -0.83378579, -0.87060891],
            [2.0289922, -0.28708987, -2.18369248],
            [0.35217619, -0.67095644, -1.58943879],
            [0.08086036, -1.81075924, 1.20752494],
            [0.8901075, -0.49976737, -1.07153746],
            [-0.44872912, -1.07275683, 0.06256855],
            [-0.22556897, 0.74798368, 0.90416439],
            [0.48339456, -2.32742195, -0.59321527],
        ],
        dtype=np.float32,
    )
    np_weight = np.ones((3, 8)).astype(np.float32)
    np_weight.fill(2.068758)
    np_bias = np.ones((8))
    np_bias.fill(0.23)
    x = flow.Tensor(input_arr, device=flow.device(device))
    flow.nn.init.constant_(linear.weight, 2.068758)
    flow.nn.init.constant_(linear.bias, 0.23)
    of_out = linear(x)
    np_out = np.matmul(input_arr, np_weight)
    np_out += np_bias
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))


def _test_linear_3_dimension_input(test_case, device):
    input_arr = np.random.randn(2, 3, 4)
    x = flow.Tensor(input_arr, device=flow.device(device))
    linear = flow.nn.Linear(4, 5, True)
    linear = _to_device(linear, device)
    flow.nn.init.constant_(linear.weight, 5.6)
    flow.nn.init.constant_(linear.bias, 0.78)
    of_out = linear(x)

    np_weight = np.ones((4, 5)).astype(np.float32)
    np_weight.fill(5.6)
    np_bias = np.ones((5))
    np_bias.fill(0.78)
    np_out = np.matmul(input_arr, np_weight)
    np_out += np_bias

    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))


def _test_linear_4_dimension_input(test_case, device):
    input_arr = np.random.randn(4, 5, 6, 7)
    x = flow.Tensor(input_arr, device=flow.device(device))
    linear = flow.nn.Linear(7, 3, False)
    linear = _to_device(linear, device)
    flow.nn.init.constant_(linear.weight, 11.3)
    of_out = linear(x)

    np_weight = np.ones((7, 3)).astype(np.float32)
    np_weight.fill(11.3)
    np_out = np.matmul(input_arr, np_weight)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))


def _test_identity(test_case, device):
    linear = flow.nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
    linear = _to_device(linear, device)
    x = flow.Tensor(np.random.rand(2, 3, 4, 5), device=flow.device(device))
    y = linear(x)
    test_case.assertTrue(np.array_equal(x.numpy(), y.numpy()))


def _test_linear_backward_with_bias(test_case, device):
    linear = flow.nn.Linear(3, 8)
    linear = _to_device(linear, device)
    x = flow.Tensor(
        [
            [-0.94630778, -0.83378579, -0.87060891],
            [2.0289922, -0.28708987, -2.18369248],
            [0.35217619, -0.67095644, -1.58943879],
            [0.08086036, -1.81075924, 1.20752494],
            [0.8901075, -0.49976737, -1.07153746],
            [-0.44872912, -1.07275683, 0.06256855],
            [-0.22556897, 0.74798368, 0.90416439],
            [0.48339456, -2.32742195, -0.59321527],
        ],
        device=flow.device(device),
        requires_grad=True,
    )
    flow.nn.init.constant_(linear.weight, 2.068758)
    flow.nn.init.constant_(linear.bias, 0.23)
    of_out = linear(x)
    of_out = of_out.sum()
    of_out.backward()

    np_grad = np.array(
        [
            [16.5501, 16.5501, 16.5501],
            [16.5501, 16.5501, 16.5501],
            [16.5501, 16.5501, 16.5501],
            [16.5501, 16.5501, 16.5501],
            [16.5501, 16.5501, 16.5501],
            [16.5501, 16.5501, 16.5501],
            [16.5501, 16.5501, 16.5501],
            [16.5501, 16.5501, 16.5501],
        ]
    )
    test_case.assertTrue(np.allclose(np_grad, x.grad.numpy(), 1e-4, 1e-4))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestLinear(flow.unittest.TestCase):
    def test_linear_forward(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_linear_no_bias,
            _test_linear_with_bias,
            _test_linear_3_dimension_input,
            _test_linear_4_dimension_input,
            _test_identity,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    def test_linear_backward(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_linear_backward_with_bias,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
