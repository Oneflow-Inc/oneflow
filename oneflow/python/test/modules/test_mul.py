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


def _test_mul_impl(test_case, device):
    x = flow.Tensor(
        np.random.randn(2, 3), device=flow.device(device), requires_grad=True
    )
    y = flow.Tensor(
        np.random.randn(2, 3), device=flow.device(device), requires_grad=True
    )
    of_out = flow.mul(x, y)
    np_out = np.multiply(x.numpy(), y.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))
    of_out = of_out.sum()
    of_out.backward()
    np_grad_x = y.numpy()
    np_grad_y = x.numpy()
    test_case.assertTrue(np.allclose(x.grad.numpy(), np_grad_x, 1e-5, 1e-5))
    test_case.assertTrue(np.allclose(y.grad.numpy(), np_grad_y, 1e-5, 1e-5))

    x = 5
    y = flow.Tensor(np.random.randn(2, 3), device=flow.device(device))
    of_out = flow.mul(x, y)
    np_out = np.multiply(x, y.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))

    x = flow.Tensor(np.random.randn(2, 3), device=flow.device(device))
    y = 5
    of_out = flow.mul(x, y)
    np_out = np.multiply(x.numpy(), y)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))

    x = flow.Tensor(
        np.random.randn(1, 1), device=flow.device(device), requires_grad=True
    )
    y = flow.Tensor(
        np.random.randn(2, 3), device=flow.device(device), requires_grad=True
    )
    of_out = flow.mul(x, y)
    np_out = np.multiply(x.numpy(), y.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))

    of_out = of_out.sum()
    of_out.backward()
    test_case.assertTrue(np.allclose(x.grad.numpy(), np.sum(y.numpy()), 1e-5, 1e-5))
    test_case.assertTrue(np.allclose(y.grad.numpy(), x.numpy(), 1e-5, 1e-5))

    x = flow.Tensor(
        np.random.randn(1, 1), device=flow.device(device), requires_grad=True
    )
    y = flow.Tensor(
        np.random.randn(2, 3, 4), device=flow.device(device), requires_grad=True
    )
    of_out = flow.mul(x, y)
    np_out = np.multiply(x.numpy(), y.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))

    of_out = of_out.sum()
    of_out.backward()
    test_case.assertTrue(np.allclose(x.grad.numpy(), np.sum(y.numpy()), 1e-5, 1e-5))
    test_case.assertTrue(np.allclose(y.grad.numpy(), x.numpy(), 1e-5, 1e-5))

    x = flow.Tensor(
        np.random.randn(1, 1), device=flow.device(device), requires_grad=True
    )
    y = flow.Tensor(
        np.random.randn(2, 3, 4, 5), device=flow.device(device), requires_grad=True
    )
    of_out = flow.mul(x, y)
    np_out = np.multiply(x.numpy(), y.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))

    of_out = of_out.sum()
    of_out.backward()
    test_case.assertTrue(np.allclose(x.grad.numpy(), np.sum(y.numpy()), 1e-5, 1e-5))
    test_case.assertTrue(np.allclose(y.grad.numpy(), x.numpy(), 1e-5, 1e-5))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestMulModule(flow.unittest.TestCase):
    def test_mul(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_mul_impl,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
