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
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


def _test_argmax_axis_negative(test_case, device):
    input = flow.tensor(
        np.random.randn(2, 6, 5, 3), dtype=flow.float32, device=flow.device(device)
    )
    axis = -1
    of_out = flow.argmax(input, dim=axis)
    np_out = np.argmax(input.numpy(), axis=axis)
    test_case.assertTrue(np.array_equal(of_out.numpy().flatten(), np_out.flatten()))


def _test_tensor_argmax(test_case, device):
    input = flow.tensor(
        np.random.randn(2, 6, 5, 3), dtype=flow.float32, device=flow.device(device)
    )
    axis = 0
    of_out = input.argmax(dim=axis)
    np_out = np.argmax(input.numpy(), axis=axis)
    test_case.assertTrue(np.array_equal(of_out.numpy().shape, np_out.shape))
    test_case.assertTrue(np.array_equal(of_out.numpy().flatten(), np_out.flatten()))


def _test_argmax_axis_postive(test_case, device):
    input = flow.tensor(
        np.random.randn(2, 6, 5, 3), dtype=flow.float32, device=flow.device(device)
    )
    axis = 1
    of_out = flow.argmax(input, dim=axis)
    np_out = np.argmax(input.numpy(), axis=axis)
    test_case.assertTrue(np.array_equal(of_out.numpy().flatten(), np_out.flatten()))


def _test_argmax_keepdims(test_case, device):
    input = flow.tensor(
        np.random.randn(2, 6, 5, 3), dtype=flow.float32, device=flow.device(device)
    )
    axis = 0
    of_out = input.argmax(axis, True)
    np_out = np.argmax(input.numpy(), axis=axis)
    np_out = np.expand_dims(np_out, axis=axis)
    test_case.assertTrue(np.array_equal(of_out.numpy().shape, np_out.shape))
    test_case.assertTrue(np.array_equal(of_out.numpy().flatten(), np_out.flatten()))


def _test_argmax_dim_equal_none(test_case, device):
    input = flow.tensor(
        np.random.randn(2, 6, 5, 3), dtype=flow.float32, device=flow.device(device)
    )
    of_out = input.argmax()
    np_out = np.argmax(input.numpy().flatten(), axis=0)
    test_case.assertTrue(np.array_equal(of_out.numpy().flatten(), np_out.flatten()))


@flow.unittest.skip_unless_1n1d()
class TestArgmax(flow.unittest.TestCase):
    def test_argmax(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_argmax_axis_negative,
            _test_tensor_argmax,
            _test_argmax_axis_postive,
            _test_argmax_keepdims,
            _test_argmax_dim_equal_none,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @autotest(n=5, auto_backward=False, rtol=1e-5, atol=1e-5, check_graph=True)
    def test_argmax_with_random_data(test_case):
        device = random_device()
        ndim = random(1, 6).to(int)
        x = random_tensor(ndim=ndim).to(device)
        y = torch.argmax(x, dim=random(0, ndim).to(int), keepdim=random().to(bool))
        return y

    @profile(torch.argmax)
    def profile_argmax(test_case):
        torch.argmax(torch.ones(100000))
        torch.argmax(torch.ones(1000000))


if __name__ == "__main__":
    unittest.main()
