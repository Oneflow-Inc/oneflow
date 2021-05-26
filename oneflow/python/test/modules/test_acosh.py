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


def _test_acosh_impl(test_case, shape, device):
    np_input = np.random.randn(*shape)
    of_input = flow.Tensor(
        np_input, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    of_out = flow.acosh(of_input)
    np_out = np.arccosh(np_input)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4, equal_nan=True))

    of_out = of_out.sum()
    of_out.backward()
    np_grad = 1. / np.sqrt(np.square(np_input) - 1)
    test_case.assertTrue(np.allclose(of_input.grad.numpy(), np_grad, 1e-4, 1e-4, equal_nan=True))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestAcosh(flow.unittest.TestCase):
    def test_acosh(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 3), (2, 4, 5, 6)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_acosh_impl(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
