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
from test_util import GenArgList

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


def _test_div_impl(test_case, shape, device):
    x = flow.tensor(
        np.random.randn(*shape), dtype=flow.float32, device=flow.device(device)
    )
    y = flow.tensor(
        np.random.randn(*shape), dtype=flow.float32, device=flow.device(device)
    )
    of_out = flow.div(x, y)
    np_out = np.divide(x.numpy(), y.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))
    x = 5
    y = flow.tensor(
        np.random.randn(*shape), dtype=flow.float32, device=flow.device(device)
    )
    of_out = flow.div(x, y)
    np_out = np.divide(x, y.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))
    x = flow.tensor(
        np.random.randn(*shape), dtype=flow.float32, device=flow.device(device)
    )
    y = 5
    of_out = flow.div(x, y)
    np_out = np.divide(x.numpy(), y)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))
    x = flow.tensor(
        np.random.randn(*shape), dtype=flow.float32, device=flow.device(device)
    )
    y = flow.tensor(
        np.random.randn(1, 1), dtype=flow.float32, device=flow.device(device)
    )
    of_out = flow.div(x, y)
    np_out = np.divide(x.numpy(), y.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))
    x = flow.tensor(np.array([5.0]), dtype=flow.float32, device=flow.device(device))
    y = flow.tensor(
        np.random.randn(*shape), dtype=flow.float32, device=flow.device(device)
    )
    of_out = flow.div(x, y)
    np_out = np.divide(x.numpy(), y.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))
    x = flow.tensor(
        np.random.randn(*shape),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    y = flow.tensor(
        np.array([5.0]),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    of_out = flow.div(x, y)
    np_out = np.divide(x.numpy(), y.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))
    of_out = of_out.sum()
    of_out.backward()
    np_grad_x = np.full(shape, 0.2)
    test_case.assertTrue(np.allclose(x.grad.numpy(), np_grad_x, 0.0001, 0.0001))


@flow.unittest.skip_unless_1n1d()
class TestDiv(flow.unittest.TestCase):
    def test_div(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 3), (2, 3, 4), (2, 4, 5, 6)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_div_impl(test_case, *arg)

    def test_div_against_pytorch(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_type"] = [test_flow_against_pytorch, test_tensor_against_pytorch]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["op"] = ["div"]
        for arg in GenArgList(arg_dict):
            arg[0](
                test_case,
                arg[2],
                extra_annotations={"other": flow.Tensor},
                extra_generators={
                    "input": random_tensor(ndim=2, dim0=2, dim1=3),
                    "other": random_tensor(ndim=2, dim0=2, dim1=3),
                },
                device=arg[1],
            )
            arg[0](
                test_case,
                arg[2],
                extra_annotations={"other": float},
                extra_generators={
                    "input": random_tensor(ndim=2, dim0=2, dim1=3),
                    "other": random(0, 5),
                },
                device=arg[1],
            )

    @autotest(auto_backward=False, check_graph=True)
    def test_0_size_div(test_case):
        device = random_device()
        x = random_pytorch_tensor(4, 2, 1, 0, 3).to(device)
        y = random_pytorch_tensor(4, 2, 1, 0, 3).to(device)
        z = x / y
        return z

    @autotest(auto_backward=False, check_graph=True)
    def test_0dim_div(test_case):
        device = random_device()
        x = random_pytorch_tensor(ndim=0).to(device)
        y = random_pytorch_tensor(ndim=0).to(device)
        z = x / y
        return z


if __name__ == "__main__":
    unittest.main()
