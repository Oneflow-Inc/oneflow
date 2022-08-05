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

from cgi import test
import unittest
from collections import OrderedDict

import numpy as np
from random import shuffle

from oneflow.test_utils.automated_test_util import *
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest


def _test_transpose(test_case, device):
    input = flow.tensor(
        np.random.randn(2, 6, 5, 3), dtype=flow.float32, device=flow.device(device)
    )
    of_out = flow.transpose(input, 0, 1)
    np_out = input.numpy().transpose((1, 0, 2, 3))
    test_case.assertTrue(np.array_equal(of_out.numpy().flatten(), np_out.flatten()))


def _test_tensor_transpose(test_case, device):
    input = flow.tensor(
        np.random.randn(2, 6, 5, 3), dtype=flow.float32, device=flow.device(device)
    )
    of_out = input.transpose(0, 1)
    np_out = input.numpy().transpose((1, 0, 2, 3))
    test_case.assertTrue(np.array_equal(of_out.numpy().flatten(), np_out.flatten()))


def _test_tranpose_negative_dim(test_case, device):
    input = flow.tensor(
        np.random.randn(2, 6, 5, 3), dtype=flow.float32, device=flow.device(device)
    )
    of_out = flow.transpose(input, -4, -3)
    np_out = input.numpy().transpose((1, 0, 2, 3))
    test_case.assertTrue(np.array_equal(of_out.numpy().flatten(), np_out.flatten()))


def _test_transpose_backward(test_case, device):
    x = flow.tensor(
        np.random.randn(2, 6, 5, 3),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    y = flow.transpose(x, 0, 1).sum()
    y.backward()
    test_case.assertTrue(
        np.allclose(x.grad.numpy(), np.ones((2, 6, 5, 3)), 1e-05, 1e-05)
    )


def _test_transpose_backward_v2(test_case, device):
    x = flow.tensor(
        np.random.randn(2, 3, 4, 5),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    y = flow.transpose(x, 3, 1).sum()
    y.backward()
    test_case.assertTrue(
        np.allclose(x.grad.numpy(), np.ones((2, 3, 4, 5)), 1e-05, 1e-05)
    )


@flow.unittest.skip_unless_1n1d()
class TestTranspose(flow.unittest.TestCase):
    def test_transpose(test_case):
        arg_dict = OrderedDict()
        arg_dict["fun"] = [
            _test_transpose,
            _test_tensor_transpose,
            _test_tranpose_negative_dim,
            _test_transpose_backward,
            _test_transpose_backward_v2,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @autotest(n=10, check_graph=True)
    def test_transpose_flow_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=4).to(device)
        y = torch.transpose(x, dim0=random(1, 3).to(int), dim1=random(1, 3).to(int))
        return y

    @autotest(n=10, check_graph=True)
    def test_transpose_with_stride(test_case):
        device = random_device()
        x = random_tensor(ndim=4).to(device)
        permute_list = [0, 1, 2, 3]
        shuffle(permute_list)
        x = x.permute(permute_list)
        y = torch.transpose(x, dim0=random(1, 3).to(int), dim1=random(1, 3).to(int))
        return y

    @autotest(n=10, auto_backward=False, check_graph=True)
    def test_transpose_with_0_size_data(test_case):
        device = random_device()
        x = random_tensor(4, 2, 3, 0, 4).to(device)
        y = torch.transpose(x, dim0=random(1, 3).to(int), dim1=random(1, 3).to(int))
        return y

    @autotest(n=10, auto_backward=False, check_graph=True)
    def test_transpose_flow_bool_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=4).to(device=device, dtype=torch.bool)
        y = torch.transpose(x, dim0=random(1, 3).to(int), dim1=random(1, 3).to(int))
        return y


if __name__ == "__main__":
    unittest.main()
