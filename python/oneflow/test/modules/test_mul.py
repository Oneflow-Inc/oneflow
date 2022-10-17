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


def _test_mul_impl(test_case, device):
    x = flow.tensor(
        np.random.randn(2, 3),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    y = flow.tensor(
        np.random.randn(2, 3),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    of_out = flow.mul(x, y)
    np_out = np.multiply(x.numpy(), y.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    of_out = of_out.sum()
    of_out.backward()
    np_grad_x = y.numpy()
    np_grad_y = x.numpy()
    test_case.assertTrue(np.allclose(x.grad.numpy(), np_grad_x, 1e-05, 1e-05))
    test_case.assertTrue(np.allclose(y.grad.numpy(), np_grad_y, 1e-05, 1e-05))
    x = 5
    y = flow.tensor(
        np.random.randn(2, 3), dtype=flow.float32, device=flow.device(device)
    )
    of_out = flow.mul(x, y)
    np_out = np.multiply(x, y.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    x = flow.tensor(
        np.random.randn(2, 3), dtype=flow.float32, device=flow.device(device)
    )
    y = 5
    of_out = flow.mul(x, y)
    np_out = np.multiply(x.numpy(), y)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    x = flow.tensor(
        np.random.randn(1, 1),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    y = flow.tensor(
        np.random.randn(2, 3),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    of_out = flow.mul(x, y)
    np_out = np.multiply(x.numpy(), y.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    of_out = of_out.sum()
    of_out.backward()
    test_case.assertTrue(np.allclose(x.grad.numpy(), np.sum(y.numpy()), 1e-05, 1e-05))
    test_case.assertTrue(np.allclose(y.grad.numpy(), x.numpy(), 1e-05, 1e-05))
    x = flow.tensor(
        np.random.randn(1, 1),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    y = flow.tensor(
        np.random.randn(2, 3, 4),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    of_out = flow.mul(x, y)
    np_out = np.multiply(x.numpy(), y.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    of_out = of_out.sum()
    of_out.backward()
    test_case.assertTrue(np.allclose(x.grad.numpy(), np.sum(y.numpy()), 1e-05, 1e-05))
    test_case.assertTrue(np.allclose(y.grad.numpy(), x.numpy(), 1e-05, 1e-05))
    x = flow.tensor(
        np.random.randn(1, 1),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    y = flow.tensor(
        np.random.randn(2, 3, 4, 5),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    of_out = flow.mul(x, y)
    np_out = np.multiply(x.numpy(), y.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    of_out = of_out.sum()
    of_out.backward()
    test_case.assertTrue(np.allclose(x.grad.numpy(), np.sum(y.numpy()), 1e-05, 1e-05))
    test_case.assertTrue(np.allclose(y.grad.numpy(), x.numpy(), 1e-05, 1e-05))


def inplace_mul_tensors_helper(test_case, device, arr_0, arr_y):
    of_x = flow.tensor(
        arr_0, dtype=flow.float32, device=flow.device(device), requires_grad=True,
    )
    of_inplace_x = of_x + 1
    of_y = flow.tensor(
        arr_y, dtype=flow.float32, device=flow.device(device), requires_grad=True,
    )
    id_inpalce_x = id(of_inplace_x)
    of_inplace_x.mul_(of_y)
    test_case.assertTrue(
        np.allclose(of_inplace_x.numpy(), np.multiply(arr_0 + 1, arr_y), 1e-05, 1e-05)
    )
    test_case.assertTrue(id_inpalce_x == id(of_inplace_x))
    of_inplace_x = of_inplace_x.sum()
    of_inplace_x.backward()
    test_case.assertTrue(np.allclose(arr_y, of_x.grad.numpy(), 1e-05, 1e-05))
    test_case.assertTrue(np.allclose(arr_0 + 1, of_y.grad.numpy(), 1e-05, 1e-05))


def _test_inplace_mul_tensors(test_case, device):
    arr_0 = np.random.rand(3, 5)
    arr_y = np.random.rand(3, 5)
    inplace_mul_tensors_helper(test_case, device, arr_0, arr_y)


def _test_inplace_mul_scalar(test_case, device):
    arr = np.random.rand(2, 3, 4)
    of_x = flow.tensor(
        arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    y = 3.25
    of_inplace_x = of_x + 1
    id_x_before = id(of_inplace_x)
    of_inplace_x.mul_(y)
    test_case.assertTrue(id_x_before == id(of_inplace_x))
    test_case.assertTrue(np.allclose(of_inplace_x.numpy(), np.multiply(arr + 1, y)))

    of_x = flow.tensor(
        arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    of_inplace_x = of_x + 1
    of_inplace_x_id_before = id(of_inplace_x)
    of_inplace_x.mul_(y)
    test_case.assertTrue(of_inplace_x_id_before == id(of_inplace_x))
    test_case.assertTrue(
        np.allclose(of_inplace_x.numpy(), np.multiply(arr + 1, y), 1e-05, 1e-05)
    )
    of_inplace_x = of_inplace_x.sum()
    of_inplace_x.backward()
    test_case.assertTrue(
        np.allclose(np.full(arr.shape, y), of_x.grad.numpy(), 1e-05, 1e-05)
    )


def _test_mul_inplace_0size_tensor(test_case, device):
    targets = flow.randn((0, 6), device=flow.device(device))
    height, width = 640, 640
    targets[:, 2:] *= flow.tensor(
        (width, height, width, height), device=flow.device(device)
    )
    test_case.assertTrue(np.array_equal(targets.size(), (0, 6)))


@flow.unittest.skip_unless_1n1d()
class TestMulModule(flow.unittest.TestCase):
    def test_mul(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_mul_impl,
            _test_inplace_mul_tensors,
            _test_inplace_mul_scalar,
            _test_mul_inplace_0size_tensor,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @autotest(check_graph=True)
    def test_broadcast_mul(test_case):
        device = random_device()
        x_0 = random_tensor(ndim=3, dim0=4, dim1=2, dim2=3).to(device)
        y = random_tensor(ndim=2, dim0=2, dim1=3).to(device)
        x = x_0 + 1
        x.mul_(y)
        return x

    @autotest(n=3)
    def test_non_contiguous_inplace_mul(test_case):
        device = random_device()
        x = random_tensor(2, 2, 4).to(device)
        y = x + 1
        y = y[:, 1:3]
        y *= random_tensor(2, 2, 2).to(device)
        return y

    @autotest(n=5)
    def test_scalar_mul_with_random_devices(test_case):
        x1_device = random_device()
        x2_device = random_device()
        x1 = random_tensor(2, 2, 3).to(x1_device).mean()
        x2 = random_tensor(2, 2, 3).to(x2_device)
        y = x1 * x2
        return y


if __name__ == "__main__":
    unittest.main()
