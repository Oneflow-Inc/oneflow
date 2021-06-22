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


def _test_argsort_dim_negative(test_case, device):
    input = flow.Tensor(
        np.random.randn(2, 6, 5, 3), dtype=flow.float32, device=flow.device(device),
    )
    axis = -1
    of_out = flow.argsort(input, dim=axis)
    np_out = np.argsort(input.numpy(), axis=axis)
    test_case.assertTrue(np.array_equal(of_out.numpy().flatten(), np_out.flatten()))


def _test_tensor_argsort(test_case, device):
    input = flow.Tensor(
        np.random.randn(2, 6, 5, 3), dtype=flow.float32, device=flow.device(device),
    )
    axis = 0
    of_out = input.argsort(dim=axis)
    np_out = np.argsort(input.numpy(), axis=axis)
    test_case.assertTrue(np.array_equal(of_out.numpy().shape, np_out.shape))
    test_case.assertTrue(np.array_equal(of_out.numpy().flatten(), np_out.flatten()))


def _test_argsort_dim_positive(test_case, device):
    input = flow.Tensor(
        np.random.randn(2, 6, 5, 3), dtype=flow.float32, device=flow.device(device),
    )
    axis = 1
    of_out = flow.argsort(input, dim=axis)
    np_out = np.argsort(input.numpy(), axis=axis)
    test_case.assertTrue(np.array_equal(of_out.numpy().flatten(), np_out.flatten()))


def _test_argsort_dim_descending(test_case, device):
    input = flow.Tensor(
        np.random.randn(2, 6, 5, 3), dtype=flow.float32, device=flow.device(device),
    )
    axis = 0
    of_out = flow.argsort(input, dim=axis, descending=True)
    np_out = np.argsort(-input.numpy(), axis=axis)
    test_case.assertTrue(np.array_equal(of_out.numpy().flatten(), np_out.flatten()))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestArgsort(flow.unittest.TestCase):
    def test_argsort(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_argsort_dim_negative,
            _test_tensor_argsort,
            _test_argsort_dim_positive,
            _test_argsort_dim_descending,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
