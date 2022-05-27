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

from oneflow.test_utils.automated_test_util import *
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


def _test_reshape(test_case, device):
    x = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    ).astype(np.float32)
    input = flow.tensor(x, dtype=flow.float32, device=flow.device(device))
    of_shape = flow.reshape(input, shape=[2, 2, 2, -1]).numpy().shape
    np_shape = (2, 2, 2, 2)
    test_case.assertTrue(np.array_equal(of_shape, np_shape))


def _test_reshape_tuple(test_case, device):
    x = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    ).astype(np.float32)
    input = flow.tensor(x, dtype=flow.float32, device=flow.device(device))
    of_shape = flow.reshape(input, shape=(2, 2, 2, -1)).numpy().shape
    np_shape = (2, 2, 2, 2)
    test_case.assertTrue(np.array_equal(of_shape, np_shape))


def _test_reshape_backward(test_case, device):
    x = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    ).astype(np.float32)
    input = flow.tensor(
        x, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    of_out = flow.reshape(input, shape=[2, 2, 2, -1]).sum()
    of_out.backward()
    np_grad = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    test_case.assertTrue(np.allclose(np_grad, input.grad.numpy(), 0.0001, 0.0001))


def _test_reshape_scalar(test_case, device):
    x = flow.tensor(2.0, device=flow.device(device))
    test_case.assertTrue(np.array_equal(x.shape, ()))
    a = flow.reshape(x, (1,))
    test_case.assertTrue(np.array_equal(a.shape, (1,)))
    b = flow.reshape(x, (1, 1, 1, 1,))
    test_case.assertTrue(np.array_equal(b.shape, (1, 1, 1, 1)))
    c = flow.reshape(b, ())
    test_case.assertTrue(np.array_equal(c.shape, ()))
    d = flow.reshape(x, ())
    test_case.assertTrue(np.array_equal(d.shape, ()))


@flow.unittest.skip_unless_1n1d()
class TestModule(flow.unittest.TestCase):
    def test_reshape(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_reshape,
            _test_reshape_tuple,
            _test_reshape_backward,
            _test_reshape_scalar,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @autotest(check_graph=True)
    def test_reshape_flow_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=4).to(device)
        y = torch.reshape(x, shape=(-1,))
        return y

    @autotest(check_graph=True)
    def test_reshape_flow_with_0dim_data(test_case):
        device = random_device()
        x = random_tensor(ndim=0).to(device)
        y = torch.reshape(x, shape=(-1,))
        return y

    @autotest(auto_backward=False, check_graph=True)
    def test_reshape_with_0_size_data(test_case):
        device = random_device()
        x = random_tensor(4, 2, 0, 3).to(device)
        y = torch.reshape(
            x, shape=(random(0, 5).to(int).value(), 0, random(0, 5).to(int).value())
        )
        return y

    @autotest(auto_backward=False, check_graph=True)
    def test_reshape_flow_bool_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=4).to(device=device, dtype=torch.bool)
        y = torch.reshape(x, shape=(-1,))
        return y

    @profile(torch.reshape)
    def profile_reshape(test_case):
        torch.reshape(torch.ones(50, 20), (20, 50))


if __name__ == "__main__":
    unittest.main()
