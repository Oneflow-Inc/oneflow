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


def _test_log1p(test_case, shape, device):
    input_arr = np.exp(np.random.randn(*shape)) - 1
    np_out = np.log1p(input_arr)
    x = flow.Tensor(
        input_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    of_out = flow.log1p(x)
    test_case.assertTrue(
        np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5, equal_nan=True)
    )

    of_out = of_out.sum()
    of_out.backward()
    np_out_grad = 1.0 / (1 + input_arr)
    test_case.assertTrue(
        np.allclose(x.grad.numpy(), np_out_grad, 1e-4, 1e-4, equal_nan=True)
    )


def _test_log1p_tensor_function(test_case, shape, device):
    input_arr = np.exp(np.random.randn(*shape)) - 1
    np_out = np.log1p(input_arr)
    x = flow.Tensor(
        input_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    of_out = x.log1p()
    test_case.assertTrue(
        np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5, equal_nan=True)
    )

    of_out = of_out.sum()
    of_out.backward()
    np_out_grad = 1.0 / (1 + input_arr)
    test_case.assertTrue(
        np.allclose(x.grad.numpy(), np_out_grad, 1e-4, 1e-4, equal_nan=True)
    )


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestLog1p(flow.unittest.TestCase):
    def test_log1p(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_log1p, _test_log1p_tensor_function]
        arg_dict["shape"] = [(2,), (2, 3), (2, 3, 4, 5)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
