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


def _test_argmax_v1(test_case, device):
    input = flow.Tensor(
        np.random.randn(2, 6, 5, 3), dtype=flow.float32, device=flow.device(device),
    )
    axis = -1
    of_out = flow.argmax(input, dim=axis)
    np_out = np.argmax(input.numpy(), axis=axis)
    test_case.assertTrue(np.array_equal(of_out.numpy().flatten(), np_out.flatten()))


def _test_tensor_argmax(test_case, device):
    input = flow.Tensor(
        np.random.randn(2, 6, 5, 3), dtype=flow.float32, device=flow.device(device),
    )
    axis = 0
    of_out = input.argmax(dim=axis)
    np_out = np.argmax(input.numpy(), axis=axis)
    test_case.assertTrue(np.array_equal(of_out.numpy().shape, np_out.shape))
    test_case.assertTrue(np.array_equal(of_out.numpy().flatten(), np_out.flatten()))


def _test_argmax_v3(test_case, device):
    input = flow.Tensor(
        np.random.randn(2, 6, 5, 3), dtype=flow.float32, device=flow.device(device),
    )
    axis = 1
    of_out = flow.argmax(input, dim=axis)
    np_out = np.argmax(input.numpy(), axis=axis)
    test_case.assertTrue(np.array_equal(of_out.numpy().flatten(), np_out.flatten()))


def _test_argmax_keepdims(test_case, device):
    input = flow.Tensor(
        np.random.randn(2, 6, 5, 3), dtype=flow.float32, device=flow.device(device),
    )
    axis = 0
    of_out = input.argmax(axis, True)
    np_out = np.argmax(input.numpy(), axis=axis)
    np_out = np.expand_dims(np_out, axis=axis)

    test_case.assertTrue(np.array_equal(of_out.numpy().shape, np_out.shape))
    test_case.assertTrue(np.array_equal(of_out.numpy().flatten(), np_out.flatten()))


def _test_argmax_dim_equal_none(test_case, device):
    input = flow.Tensor(
        np.random.randn(2, 6, 5, 3), dtype=flow.float32, device=flow.device(device),
    )
    of_out = input.argmax()
    np_out = np.argmax(input.numpy().flatten(), axis=0)
    test_case.assertTrue(np.array_equal(of_out.numpy().flatten(), np_out.flatten()))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestArgmax(flow.unittest.TestCase):
    def test_transpose(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_argmax_v1,
            _test_tensor_argmax,
            _test_argmax_v3,
            _test_argmax_keepdims,
            _test_argmax_dim_equal_none,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
