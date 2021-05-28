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


def _test_arange(test_case, device):
    np_out = np.arange(13, dtype=np.float32)
    of_out = flow.arange(13, device=device, dtype=flow.float32)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))


def _test_arange_step_prarm(test_case, device):
    np_out = np.arange(0, 20, 2)
    of_out = flow.arange(0, 20, step=2, device=device)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))


def _test_arange_more_params(test_case, device):
    np_out = np.arange(0, 100, 3)
    of_out = flow.arange(start=0, end=100, step=3, device=device)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))


def _test_arange_backward(test_case, device):
    np_out = np.arange(13)
    x = flow.arange(13, device=device)
    x.requires_grad = True
    y = x.sum()
    y.backward()
    test_case.assertTrue(np.allclose(x.grad.numpy(), np.ones(13), 1e-5, 1e-5))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestArange(flow.unittest.TestCase):
    def test_transpose(test_case):
        arg_dict = OrderedDict()
        arg_dict["function_test"] = [
            _test_arange,
            _test_arange_step_prarm,
            _test_arange_more_params,
            _test_arange_backward,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
