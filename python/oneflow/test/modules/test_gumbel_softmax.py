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
from oneflow.test_utils.test_util import GenArgList, type_name_to_flow_type
from oneflow.test_utils.automated_test_util import *

import oneflow as flow
import oneflow.nn.functional as F
import oneflow.unittest


def _test_gumbel_softmax(test_case, tau, dim, device, dtype):
    dtype = type_name_to_flow_type[dtype]
    x = flow.tensor(np.random.randn(20, 32), dtype=dtype, device=flow.device(device),)
    y_soft = F.gumbel_softmax(x, tau=tau, dim=dim)
    y_hard = F.gumbel_softmax(x, tau=tau, dim=dim, hard=True)
    test_case.assertEqual(x.shape, y_soft.shape)
    test_case.assertEqual(x.shape, y_hard.shape)
    test_case.assertEqual(x.dtype, y_soft.dtype)
    test_case.assertEqual(x.dtype, y_hard.dtype)


def _test_gumbel_softmax_hard(test_case, tau, dim, device, dtype):
    dtype = type_name_to_flow_type[dtype]
    x = flow.tensor(np.random.randn(45, 23), dtype=dtype, device=flow.device(device),)
    y_hard = F.gumbel_softmax(x, tau=tau, dim=dim, hard=True)
    test_case.assertEqual(y_hard.min(), 0)
    if dim == -1:
        test_case.assertEqual(y_hard.sum().item(), 45)
    elif dim == 0:
        test_case.assertEqual(y_hard.sum().item(), 23)


def _test_gumbel_softmax_backward(test_case, tau, dim, device, dtype):
    dtype = type_name_to_flow_type[dtype]
    x_np = np.random.rand(10, 10)
    x_soft = flow.tensor(
        x_np, dtype=dtype, device=flow.device(device), requires_grad=True,
    )
    x_hard = flow.tensor(
        x_np, dtype=dtype, device=flow.device(device), requires_grad=True,
    )
    y_soft = F.gumbel_softmax(x_soft, tau, dim=dim)
    y_hard = F.gumbel_softmax(x_hard, tau, dim=dim, hard=False)

    y_soft.mean().backward()
    y_hard.mean().backward()

    np.testing.assert_allclose(
        x_hard.grad.numpy(), x_soft.grad.numpy(), rtol=1e-5, atol=1e-5, verbose=True
    )


def _test_gumbel_softmax_half(test_case, tau, dim, device):
    x = flow.tensor(np.random.randn(20, 32), device=flow.device(device),).to(
        flow.float16
    )
    y_soft = F.gumbel_softmax(x, tau=tau, dim=dim)
    y_hard = F.gumbel_softmax(x, tau=tau, dim=dim, hard=True)
    test_case.assertEqual(x.shape, y_soft.shape)
    test_case.assertEqual(x.shape, y_hard.shape)
    test_case.assertEqual(x.dtype, y_soft.dtype)
    test_case.assertEqual(x.dtype, y_hard.dtype)


@flow.unittest.skip_unless_1n1d()
class TestGumbelSoftmaxModule(flow.unittest.TestCase):
    @autotest()
    def test_gumbel_softmax(test_case):
        arg_dict = OrderedDict()
        arg_dict["fun"] = [
            _test_gumbel_softmax,
            _test_gumbel_softmax_hard,
            _test_gumbel_softmax_backward,
        ]
        arg_dict["tau"] = [1, 2, 0.5]
        arg_dict["dim"] = [0, -1]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["dtype"] = ["float32", "double"]

        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @autotest()
    def test_leakyrelu_module_with_half_random_data(test_case):
        arg_dict = OrderedDict()
        arg_dict["fun"] = [
            _test_gumbel_softmax_half,
        ]
        arg_dict["tau"] = [1, 2, 0.5]
        arg_dict["dim"] = [0, -1]
        arg_dict["device"] = ["cuda"]

        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
