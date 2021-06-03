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


def _test_sum_impl(test_case, device):
    input = flow.Tensor(
        np.random.randn(2, 3), dtype=flow.float32, device=flow.device(device)
    )
    of_out = flow.sum(input, dim=0)
    np_out = np.sum(input.numpy(), axis=0)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))

    input = flow.Tensor(
        np.random.randn(2, 3), dtype=flow.float32, device=flow.device(device)
    )
    of_out = flow.sum(input, dim=0)
    np_out = np.sum(input.numpy(), axis=0)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))

    input = flow.Tensor(
        np.random.randn(2, 3), dtype=flow.float32, device=flow.device(device)
    )
    of_out = flow.sum(input, dim=1)
    of_out2 = input.sum(dim=1)
    np_out = np.sum(input.numpy(), axis=1)
    test_case.assertTrue(np.allclose(of_out2.numpy(), of_out.numpy(), 1e-5, 1e-5))
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))

    input = flow.Tensor(
        np.random.randn(4, 5, 6),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    of_out = flow.sum(input, dim=(2, 1))
    np_out = np.sum(input.numpy(), axis=(2, 1))
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))
    of_out = of_out.sum()
    of_out.backward()
    np_grad = np.ones((4, 5, 6))
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-5, 1e-5))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestSumModule(flow.unittest.TestCase):
    def test_sum(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_sum_impl(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
