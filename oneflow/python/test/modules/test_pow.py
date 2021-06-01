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


def _test_pow_scalar_impl(test_case, shape, scalar, device):
    np_input = 10 * np.random.rand(*shape)
    of_input = flow.Tensor(np_input, dtype=flow.float32, device=flow.device(device))
    of_out = flow.pow(of_input, scalar)
    np_out = np.power(np_input, scalar)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))


def _test_pow_elementwise_impl(test_case, shape, scalar, device):
    np_input_x = 10 * np.random.rand(*shape)
    np_input_y = np.random.randint(1, 3, shape) + np.random.randn(*shape)
    of_input_x = flow.Tensor(np_input_x, dtype=flow.float32, device=flow.device(device))
    of_input_y = flow.Tensor(np_input_y, dtype=flow.float32, device=flow.device(device))
    of_out = flow.pow(of_input_x, of_input_y)
    np_out = np.power(np_input_x, np_input_y)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))


def _test_pow_backward_impl(test_case, device):
    # elementwise-pow backward test
    np_input_x = np.array(
        [[0.86895168, 0.51427012, 0.8693118], [0.27302601, 0.68126282, 0.85506865]]
    )
    np_input_y = np.array(
        [[0.42736459, 0.0727016, 0.90737411], [0.7220017, 0.32741095, 0.49669031]]
    )
    np_x_grad = np.array([[0.4632, 0.1347, 0.9192], [1.0358, 0.4238, 0.5374]])
    np_y_grad = np.array([[-0.1323, -0.6336, -0.1233], [-0.5085, -0.3385, -0.1449]])

    def test_x_y_grad():
        of_input_x = flow.Tensor(
            np_input_x,
            dtype=flow.float32,
            device=flow.device(device),
            requires_grad=True,
        )
        of_input_y = flow.Tensor(
            np_input_y,
            dtype=flow.float32,
            device=flow.device(device),
            requires_grad=True,
        )
        of_out = flow.pow(of_input_x, of_input_y)
        of_out_sum = of_out.sum()
        of_out_sum.backward()
        test_case.assertTrue(
            np.allclose(of_input_x.grad.numpy(), np_x_grad, 1e-4, 1e-4)
        )
        test_case.assertTrue(
            np.allclose(of_input_y.grad.numpy(), np_y_grad, 1e-4, 1e-4)
        )

    def test_x_grad():
        of_input_x = flow.Tensor(
            np_input_x,
            dtype=flow.float32,
            device=flow.device(device),
            requires_grad=True,
        )
        of_input_y = flow.Tensor(
            np_input_y, dtype=flow.float32, device=flow.device(device)
        )
        of_out = flow.pow(of_input_x, of_input_y)
        of_out_sum = of_out.sum()
        of_out_sum.backward()
        test_case.assertTrue(
            np.allclose(of_input_x.grad.numpy(), np_x_grad, 1e-4, 1e-4)
        )

    def test_y_grad():
        of_input_x = flow.Tensor(
            np_input_x, dtype=flow.float32, device=flow.device(device)
        )
        of_input_y = flow.Tensor(
            np_input_y,
            dtype=flow.float32,
            device=flow.device(device),
            requires_grad=True,
        )
        of_out = flow.pow(of_input_x, of_input_y)
        of_out_sum = of_out.sum()
        of_out_sum.backward()
        test_case.assertTrue(
            np.allclose(of_input_y.grad.numpy(), np_y_grad, 1e-4, 1e-4)
        )

    test_x_y_grad()
    test_x_grad()
    test_y_grad()

    # TODO(liupeihong): scalar-pow backward test
    # ...


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestPow(flow.unittest.TestCase):
    def test_pow_forward(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 3), (2, 3, 4, 5)]
        arg_dict["scalar"] = [2.1, 0.8]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_pow_scalar_impl(test_case, *arg)
            _test_pow_elementwise_impl(test_case, *arg)

    def test_pow_backward(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_pow_backward_impl(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
