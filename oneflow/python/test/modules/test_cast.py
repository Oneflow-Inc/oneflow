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


def _test_cast_float2int(test_case, shape, device):
    np_arr = np.random.randn(*shape).astype(np.float32)
    input = flow.Tensor(np_arr, dtype=flow.float32, device=flow.device(device),)
    output = flow.cast(input, flow.int8)
    np_out = np_arr.astype(np.int8)
    test_case.assertTrue(np.array_equal(output.numpy(), np_out))


def _test_cast_int2float(test_case, shape, device):
    np_arr = np.random.randn(*shape).astype(np.int8)
    input = flow.Tensor(np_arr, dtype=flow.int8, device=flow.device(device),)
    output = flow.cast(input, flow.float32)
    np_out = np_arr.astype(np.float32)
    test_case.assertTrue(np.array_equal(output.numpy(), np_out))


def _test_cast_backward(test_case, shape, device):
    np_arr = np.random.randn(*shape).astype(np.int8)
    input = flow.Tensor(
        np_arr, dtype=flow.int8, device=flow.device(device), requires_grad=True
    )
    of_out = flow.cast(input, flow.float32)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = np.ones(shape)
    test_case.assertTrue(np.array_equal(input.grad.numpy(), np_grad))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestCast(flow.unittest.TestCase):
    def test_cast(test_case):
        arg_dict = OrderedDict()
        arg_dict["function_test"] = [
            _test_cast_float2int,
            _test_cast_int2float,
            _test_cast_backward,
        ]
        arg_dict["shape"] = [(2, 3), (2, 4, 5, 6)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
