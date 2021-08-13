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


def _test_negtive(test_case, shape, device):
    np_input = np.random.randn(*shape)
    input = flow.Tensor(np_input, dtype=flow.float32, device=flow.device(device))
    of_out = flow.negative(input)
    np_out = -input.numpy()
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


def _test_negative_neg(test_case, shape, device):
    np_input = np.random.randn(*shape)
    input = flow.Tensor(np_input, dtype=flow.float32, device=flow.device(device))
    of_out = flow.neg(input)
    np_out = -input.numpy()
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


def _test_tensor_negative(test_case, shape, device):
    np_input = np.random.randn(*shape)
    input = flow.Tensor(np_input, dtype=flow.float32, device=flow.device(device))
    of_out = input.negative()
    np_out = -input.numpy()
    test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))


def _test_negative_backward(test_case, shape, device):
    np_input = np.random.randn(*shape)
    input = flow.Tensor(
        np_input, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    of_out = flow.negative(input)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = -np.ones(shape)
    test_case.assertTrue(
        np.allclose(input.grad.numpy(), np_grad, atol=1e-05, rtol=1e-05)
    )


@flow.unittest.skip_unless_1n1d()
class TestNegativeModule(flow.unittest.TestCase):
    def test_negative(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_negtive,
            _test_negative_neg,
            _test_tensor_negative,
            _test_negative_backward,
        ]
        arg_dict["shape"] = [(2, 3), (2, 4, 5, 6)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @autotest(auto_backward=False)
    def test_ne_with_0shape_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(4, 2, 3, 0, 5).to(device)
        y1 = torch.negative(x)
        y2 = torch.neg(x)
        y3 = -x
        return (y1, y2, y3)


if __name__ == "__main__":
    unittest.main()
