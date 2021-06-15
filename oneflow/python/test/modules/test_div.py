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


def _test_div_impl(test_case, shape, device):
    x = flow.Tensor(np.random.randn(*shape), device=flow.device(device))
    y = flow.Tensor(np.random.randn(*shape), device=flow.device(device))
    of_out = flow.div(x, y)
    np_out = np.divide(x.numpy(), y.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

    x = 5
    y = flow.Tensor(np.random.randn(*shape), device=flow.device(device))
    of_out = flow.div(x, y)
    np_out = np.divide(x, y.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

    x = flow.Tensor(np.random.randn(*shape), device=flow.device(device))
    y = 5
    of_out = flow.div(x, y)
    np_out = np.divide(x.numpy(), y)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

    x = flow.Tensor(np.random.randn(*shape), device=flow.device(device))
    y = flow.Tensor(np.random.randn(1, 1), device=flow.device(device))
    of_out = flow.div(x, y)
    np_out = np.divide(x.numpy(), y.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

    x = flow.Tensor(np.array([5.0]), device=flow.device(device))
    y = flow.Tensor(np.random.randn(*shape), device=flow.device(device))
    of_out = flow.div(x, y)
    np_out = np.divide(x.numpy(), y.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

    x = flow.Tensor(
        np.random.randn(*shape), device=flow.device(device), requires_grad=True
    )
    y = flow.Tensor(np.array([5.0]), device=flow.device(device), requires_grad=True)
    of_out = flow.div(x, y)
    np_out = np.divide(x.numpy(), y.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))
    of_out = of_out.sum()
    of_out.backward()
    np_grad_x = np.full(shape, 0.2)
    test_case.assertTrue(np.allclose(x.grad.numpy(), np_grad_x, 1e-4, 1e-4))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestDiv(flow.unittest.TestCase):
    def test_div(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 3), (2, 3, 4), (2, 4, 5, 6)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_div_impl(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
