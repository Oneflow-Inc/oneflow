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

    @autotest(n=5, auto_backward=False, check_graph=True)
    def test_random_dim_div(test_case):
        device = random_device()
        dim0 = random(low=1, high=4).to(int)
        dim1 = random(low=1, high=4).to(int)
        x = random_tensor(ndim=2, dim0=dim0, dim1=dim1).to(device)
        y = random_tensor(ndim=2, dim0=dim0, dim1=dim1).to(device)
        z = x / y
        return z

    @autotest(n=5, auto_backward=False, check_graph=True)
    def test_random_dim_scalar_div(test_case):
        device = random_device()
        dim0 = random(low=1, high=4).to(int)
        dim1 = random(low=1, high=4).to(int)
        x = random_tensor(ndim=2, dim0=dim0, dim1=dim1).to(device)
        y = random_tensor(ndim=0).to(device)
        z = x / y
        return z

    @autotest(n=5, auto_backward=False, check_graph=True)
    def test_0_size_div(test_case):
        device = random_device()
        x = random_tensor(4, 2, 1, 0, 3).to(device)
        y = random_tensor(4, 2, 1, 0, 3).to(device)
        z = x / y
        return z

    @autotest(n=5, auto_backward=False, check_graph=True)
    def test_0dim_div(test_case):
        device = random_device()
        x = random_tensor(ndim=0).to(device)
        y = random_tensor(ndim=0).to(device)
        z = x / y
        return z

    @autotest(n=3)
    def test_non_contiguous_inplace_div(test_case):
        device = random_device()
        x = random_tensor(2, 2, 4).to(device)
        y = x + 1
        y = y[:, 1:3]
        y /= random_tensor(2, 2, 2).to(device)
        return y

    @autotest(n=3, check_graph=False)
    def test_int_dtype_inplace_div(test_case):
        num_elems = 20
        flow_out = flow.arange(num_elems) / num_elems
        torch_out = torch.arange(num_elems) / num_elems
        test_case.assertTrue(np.allclose(flow_out.numpy(), torch_out.numpy()))

    @autotest(n=5)
    def test_scalar_div_with_random_devices(test_case):
        x1_device = random_device()
        x2_device = random_device()
        x1 = random_tensor(2, 2, 3).to(x1_device).mean()
        x2 = random_tensor(2, 2, 3).to(x2_device)
        y = x1 / x2
        return y

    @profile(torch.div)
    def profile_div(test_case):
        input1 = torch.ones(16, 10, 128, 128)
        input2 = torch.ones(16, 10, 128, 128)
        torch.div(input1, input2)


@flow.unittest.skip_unless_1n1d()
class TestDivRoundmode(flow.unittest.TestCase):
    @autotest(n=3)
    def test_random_dim_div_floor(test_case):
        device = random_device()
        dim0 = random(low=1, high=4).to(int)
        dim1 = random(low=1, high=4).to(int)
        x = random_tensor(ndim=2, dim0=dim0, dim1=dim1).to(device)
        y = random_tensor(ndim=2, dim0=dim0, dim1=dim1).to(device)
        z = torch.div(x, y, rounding_mode="floor")
        return z

    @autotest(n=3)
    def test_random_dim_div_trunc(test_case):
        device = random_device()
        dim0 = random(low=1, high=4).to(int)
        dim1 = random(low=1, high=4).to(int)
        x = random_tensor(ndim=2, dim0=dim0, dim1=dim1).to(device)
        y = random_tensor(ndim=2, dim0=dim0, dim1=dim1).to(device)
        z = torch.div(x, y, rounding_mode="trunc")
        return z

    @autotest(n=3)
    def test_scalar_div_mode_floor(test_case):
        device = random_device()
        x1 = random(low=1, high=5).to(float)
        x2 = random_tensor(2, 2, 3).to(device)
        y = torch.div(x1, x2, rounding_mode="floor")
        return y

    @autotest(n=3)
    def test_scalar_div_mode_trunc(test_case):
        device = random_device()
        x1 = random(low=1, high=5).to(float)
        x2 = random_tensor(2, 2, 3).to(device)
        y = torch.div(x1, x2, rounding_mode="trunc")
        return y

    @autotest(n=3)
    def test_scalar_div_mode_floor2(test_case):
        device = random_device()
        x1 = random(low=1, high=5).to(float)
        x2 = random_tensor(2, 2, 3).to(device)
        y = torch.div(x2, x1, rounding_mode="floor")
        return y

    @autotest(n=3)
    def test_scalar_div_mode_trunc2(test_case):
        device = random_device()
        x1 = random(low=1, high=5).to(float)
        x2 = random_tensor(2, 2, 3).to(device)
        y = torch.div(x2, x1, rounding_mode="trunc")
        return y


if __name__ == "__main__":
    unittest.main()
