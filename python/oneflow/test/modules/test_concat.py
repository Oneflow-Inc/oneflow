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
from test_util import GenArgList

import oneflow as flow
import oneflow.unittest
from automated_test_util import *


def _test_concat_origin(test_case, device):
    input1 = flow.Tensor(
        np.random.randn(2, 6, 5, 3), dtype=flow.float32, device=flow.device(device)
    )
    input2 = flow.Tensor(
        np.random.randn(2, 6, 5, 3), dtype=flow.float32, device=flow.device(device)
    )
    of_out = flow.cat([input1, input2], dim=0)
    np_out = np.concatenate((input1.numpy(), input2.numpy()), axis=0)
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


def _test_concat_with_axis_one(test_case, device):
    input1 = flow.Tensor(
        np.random.randn(2, 6, 5, 3), dtype=flow.float32, device=flow.device(device)
    )
    input2 = flow.Tensor(
        np.random.randn(2, 6, 5, 3), dtype=flow.float32, device=flow.device(device)
    )
    of_out = flow.cat([input1, input2], dim=1)
    np_out = np.concatenate((input1.numpy(), input2.numpy()), axis=1)
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


def _test_concat_with_three_tensor(test_case, device):
    input1 = flow.Tensor(
        np.random.randn(2, 6, 5, 3), dtype=flow.float32, device=flow.device(device)
    )
    input2 = flow.Tensor(
        np.random.randn(2, 6, 5, 3), dtype=flow.float32, device=flow.device(device)
    )
    input3 = flow.Tensor(
        np.random.randn(2, 6, 5, 3), dtype=flow.float32, device=flow.device(device)
    )
    of_out = flow.cat([input1, input2, input3], dim=1)
    np_out = np.concatenate((input1.numpy(), input2.numpy(), input3.numpy()), axis=1)
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


def _test_concat_with_three_tensor_backward(test_case, device):
    input1 = flow.Tensor(
        np.random.randn(2, 6, 5, 3),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    input2 = flow.Tensor(
        np.random.randn(2, 6, 5, 3),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    input3 = flow.Tensor(
        np.random.randn(2, 6, 5, 3),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    of_out = flow.cat([input1, input2, input3], dim=1)
    of_out = of_out.sum()
    of_out.backward()
    test_case.assertTrue(
        np.allclose(input1.grad.numpy(), np.ones((2, 6, 5, 3)), 0.0001, 0.0001)
    )
    test_case.assertTrue(
        np.allclose(input2.grad.numpy(), np.ones((2, 6, 5, 3)), 0.0001, 0.0001)
    )
    test_case.assertTrue(
        np.allclose(input3.grad.numpy(), np.ones((2, 6, 5, 3)), 0.0001, 0.0001)
    )


def _test_concat_grad_and_no_grad(test_case, device):
    input1 = flow.Tensor(
        np.random.randn(2, 6, 5, 3),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    input2 = flow.Tensor(
        np.random.randn(2, 6, 5, 3),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=False,
    )
    of_out = flow.cat([input1, input2], dim=1)
    of_out = of_out.sum()
    of_out.backward()
    test_case.assertTrue(
        np.allclose(input1.grad.numpy(), np.ones((2, 6, 5, 3)), 0.0001, 0.0001)
    )


@flow.unittest.skip_unless_1n1d()
class TestModule(flow.unittest.TestCase):
    def test_concat(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_concat_origin,
            _test_concat_with_axis_one,
            _test_concat_with_three_tensor,
            _test_concat_with_three_tensor_backward,
            _test_concat_grad_and_no_grad,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @autotest(n=10, auto_backward=False)
    def test_concat_with_0shape_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(4, 2, 3, 2, 4).to(device)
        y = random_pytorch_tensor(4, 2, 3, random(0, 3), 4).to(device)
        z = torch.cat((x, y), dim=2)
        return z


if __name__ == "__main__":
    unittest.main()
