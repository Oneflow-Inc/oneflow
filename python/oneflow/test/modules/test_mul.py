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
import torch

from oneflow.test_utils.automated_test_util import *
from test_util import GenArgList

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


def test_inplace_mul_id(test_case, device):
    x = flow.tensor(
        np.random.randn(2, 3),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=False,
    )
    y = flow.tensor(
        np.random.randn(2, 3),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=False,
    )
    id_x_before = id(x)
    id_y_before = id(y)
    out = x.mul_(y)
    test_case.assertTrue(id_x_before == id(x))
    test_case.assertTrue(id_x_before == id(out))
    test_case.assertTrue(id_y_before == id(y))


def inplace_mul_tensors_helper(test_case, device, arr_0, arr_y):
    pt_x = torch.tensor(
        arr_0,
        dtype=torch.float32,
        device=torch.device(device),
        requires_grad=True,
    )
    pt_inplace_x = pt_x + 1
    pt_y = torch.tensor(
        arr_y,
        dtype=torch.float32,
        device=torch.device(device),
        requires_grad=True,
    )
    of_x = flow.tensor(
        arr_0,
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    of_inplace_x = of_x + 1
    of_y = flow.tensor(
        arr_y,
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    id_inpalce_x = id(of_inplace_x)
    pt_out = pt_inplace_x.mul_(pt_y)
    of_out = of_inplace_x.mul_(of_y)
    test_case.assertTrue(np.allclose(of_out.numpy(), pt_out.clone().detach().cpu().numpy(), 1e-05, 1e-05))
    test_case.assertTrue(id_inpalce_x == id(of_inplace_x))
    pt_inplace_x = pt_inplace_x.sum()
    pt_inplace_x.backward()
    of_inplace_x = of_inplace_x.sum()
    of_inplace_x.backward()
    test_case.assertTrue(np.allclose(pt_x.grad.cpu().numpy(), of_x.grad.numpy(), 1e-05, 1e-05))
    test_case.assertTrue(np.allclose(pt_y.grad.cpu().numpy(), of_y.grad.numpy(), 1e-05, 1e-05))

def test_inplace_mul_tensors(test_case, device):
    arr_0 = np.random.rand(3, 5)
    arr_y = np.random.rand(3, 5)
    inplace_mul_tensors_helper(test_case, device, arr_0, arr_y)
    # test inplace boardcast mul
    arr_0 = np.random.rand(5, 1, 4, 1)
    arr_y = np.random.rand(1, 4, 1)
    inplace_mul_tensors_helper(test_case, device, arr_0, arr_y)
    arr_0 = np.random.rand(2, 3, 4, 5)
    arr_y = np.random.rand(1, 1)
    inplace_mul_tensors_helper(test_case, device, arr_0, arr_y)


def test_inplace_mul_scalar(test_case, device):
    arr = np.random.rand(2, 3, 4)
    of_x = flow.tensor(
        arr,
        flow.float32,
        device=flow.device(device),
        requires_grad=True
    )
    y = 3.25
    id_x_before = id(of_x)
    of_x.mul_(y)
    test_case.assertTrue(id_x_before == id(x))
    test_case.assertTrue(np.allclose(of_x.numpy(), np.multiply(arr, y)))

    of_x = flow.tensor(
        arr,
        flow.float32,
        device=flow.device(device),
        requires_grad=True
    )
    pt_x = torch.tensor(
        arr,
        flow.float32,
        device=torch.device(device),
        requires_grad=True
    )
    of_inplace_x_id_before = id(of_inplace_x)
    of_inplace_x = of_x + 1
    pt_inplace_x = pt_x + 1
    of_inplace_x.mul_(y)
    pt_inplace_x.mul_(y)
    test_case.assertTrue(of_inplace_x_id_before == id(of_inplace_x))
    test_case.assertTrue(np.allclose(of_out.numpy(), pt_out.clone().detach().cpu().numpy(), 1e-05, 1e-05))
    pt_inplace_x = pt_inplace_x.sum()
    pt_inplace_x.backward()
    of_inplace_x = of_inplace_x.sum()
    of_inplace_x.backward()
    test_case.assertTrue(np.allclose(pt_x.cpu().grad.numpy(), of_x.grad.numpy(), 1e-05, 1e-05))
    test_case.assertTrue(np.allclose(pt_y.grad.numpy(), of_y.grad.numpy(), 1e-05, 1e-05))


@flow.unittest.skip_unless_1n1d()
class TestMulModule(flow.unittest.TestCase):
    def test_mul(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_mul_impl, test_inplace_mul_tensors, test_inplace_mul_id]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
