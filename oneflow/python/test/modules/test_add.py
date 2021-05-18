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
from test_util import GenArgList
from collections import OrderedDict

import numpy as np

import oneflow.experimental as flow


def _test_add_forward(test_case, device):
    x = flow.Tensor(np.random.randn(2, 3), device=flow.device(device))
    y = flow.Tensor(np.random.randn(2, 3), device=flow.device(device))
    of_out = flow.add(x, y)
    np_out = np.add(x.numpy(), y.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

    x = 5
    y = flow.Tensor(np.random.randn(2, 3), device=flow.device(device))
    of_out = flow.add(x, y)
    np_out = np.add(x, y.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

    x = flow.Tensor(np.random.randn(2, 3), device=flow.device(device))
    y = 5
    of_out = flow.add(x, y)
    np_out = np.add(x.numpy(), y)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

    x = flow.Tensor(np.random.randn(2, 3), device=flow.device(device))
    y = flow.Tensor(np.array([5.0]), device=flow.device(device))
    of_out = flow.add(x, y)
    np_out = np.add(x.numpy(), y.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

    x = flow.Tensor(np.random.randn(1, 1), device=flow.device(device))
    y = flow.Tensor(np.random.randn(2, 3), device=flow.device(device))
    of_out = flow.add(x, y)
    np_out = np.add(x.numpy(), y.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))


def _test_add_backward(test_case, device):
    x = 5
    y = flow.Tensor(
        np.random.randn(2, 3), requires_grad=True, device=flow.device(device)
    )
    of_out = flow.add(x, y).sum()
    of_out.backward()
    test_case.assertTrue(np.allclose(y.grad.numpy(), np.ones((2, 3)), 1e-4, 1e-4))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestAddModule(flow.unittest.TestCase):
    def test_add_forward(test_case):
        arg_dict = OrderedDict()
        arg_dict["fun"] = [_test_add_forward, _test_add_backward]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
