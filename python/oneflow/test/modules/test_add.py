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


def _test_add_forward(test_case, shape, device):
    x = flow.tensor(np.random.randn(*shape), device=flow.device(device))
    y = flow.tensor(np.random.randn(*shape), device=flow.device(device))
    of_out = flow.add(x, y)
    np_out = np.add(x.numpy(), y.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))
    x = 5
    y = flow.tensor(np.random.randn(*shape), device=flow.device(device))
    of_out = flow.add(x, y)
    np_out = np.add(x, y.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))
    x = flow.tensor(np.random.randn(*shape), device=flow.device(device))
    y = 5
    of_out = flow.add(x, y)
    np_out = np.add(x.numpy(), y)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))
    x = flow.tensor(np.random.randn(*shape), device=flow.device(device))
    y = flow.tensor(np.array([5.0]), device=flow.device(device))
    of_out = flow.add(x, y)
    np_out = np.add(x.numpy(), y.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))
    x = flow.tensor(np.random.randn(1, 1), device=flow.device(device))
    y = flow.tensor(np.random.randn(*shape), device=flow.device(device))
    of_out = flow.add(x, y)
    np_out = np.add(x.numpy(), y.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))


def _test_add_backward(test_case, shape, device):
    x = 5
    y = flow.tensor(
        np.random.randn(*shape), requires_grad=True, device=flow.device(device)
    )
    of_out = flow.add(x, y).sum()
    of_out.backward()
    test_case.assertTrue(
        np.allclose(y.grad.numpy(), np.ones(shape=shape), 0.0001, 0.0001)
    )


def _test_inplace_add(test_case, shape, device):
    np_x = np.random.randn(*shape)
    of_x = flow.tensor(
        np_x, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    of_x_inplace = of_x + 1
    id_old = id(of_x_inplace)
    of_x_inplace.add_(5)
    test_case.assertEqual(id_old, id(of_x_inplace))
    np_out = np_x + 1 + 5
    test_case.assertTrue(np.allclose(of_x_inplace.numpy(), np_out, 1e-05, 1e-05))
    of_x_inplace = of_x_inplace.sum()
    of_x_inplace.backward()
    test_case.assertTrue(np.allclose(of_x.grad.numpy(), np.ones(shape), 1e-05, 1e-05))

    of_x = flow.tensor(
        np_x, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    of_y = flow.tensor(
        np.random.randn(*shape),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=False,
    )
    of_x_inplace = of_x + 1
    id_old = id(of_x_inplace)
    of_x_inplace.add_(of_y)
    test_case.assertEqual(id_old, id(of_x_inplace))
    np_out = np_x + 1 + of_y.numpy()
    test_case.assertTrue(np.allclose(of_x_inplace.numpy(), np_out, 1e-05, 1e-05))
    of_x_inplace = of_x_inplace.sum()
    of_x_inplace.backward()
    test_case.assertTrue(np.allclose(of_x.grad.numpy(), np.ones(shape), 1e-05, 1e-05))

    of_x = flow.tensor(
        np_x, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    of_y = flow.tensor(
        np.random.randn(*shape),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=False,
    )
    of_x_inplace = of_x + 1
    id_old = id(of_x_inplace)
    of_x_inplace += of_y
    test_case.assertEqual(id_old, id(of_x_inplace))
    np_out = np_x + 1 + of_y.numpy()
    test_case.assertTrue(np.allclose(of_x_inplace.numpy(), np_out, 1e-05, 1e-05))
    of_x_inplace = of_x_inplace.sum()
    of_x_inplace.backward()
    test_case.assertTrue(np.allclose(of_x.grad.numpy(), np.ones(shape), 1e-05, 1e-05))

    of_x = flow.tensor(
        np_x, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    of_y = flow.tensor(
        np.array([5.0]),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=False,
    )
    of_x_inplace = of_x + 1
    id_old = id(of_x_inplace)
    of_x_inplace.add_(of_y)
    test_case.assertEqual(id_old, id(of_x_inplace))
    np_out = np_x + 6
    test_case.assertTrue(np.allclose(of_x_inplace.numpy(), np_out, 1e-05, 1e-05))
    of_x_inplace = of_x_inplace.sum()
    of_x_inplace.backward()
    test_case.assertTrue(np.allclose(of_x.grad.numpy(), np.ones(shape), 1e-05, 1e-05))

    of_x = flow.tensor(
        np_x, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    np_y = np.random.randn(*shape[:-1], 1)
    of_y = flow.tensor(
        np_y, dtype=flow.float32, device=flow.device(device), requires_grad=False
    )
    of_x_inplace = of_x + 1
    id_old = id(of_x_inplace)
    of_x_inplace.add_(of_y)
    test_case.assertEqual(id_old, id(of_x_inplace))
    np_out = np_x + 1 + np_y
    test_case.assertTrue(np.allclose(of_x_inplace.numpy(), np_out, 1e-05, 1e-05))
    of_x_inplace = of_x_inplace.sum()
    of_x_inplace.backward()
    test_case.assertTrue(np.allclose(of_x.grad.numpy(), np.ones(shape), 1e-05, 1e-05))


def _test_inplace_add_with_type_promotion(test_case, shape, device):
    x = flow.tensor(
        np.random.randn(*shape), device=flow.device(device), dtype=flow.float16
    )
    y = flow.tensor(
        np.random.randn(*shape), device=flow.device(device), dtype=flow.float32
    )
    x += y
    test_case.assertTrue(x.dtype == flow.float16)


@flow.unittest.skip_unless_1n1d()
class TestAddModule(flow.unittest.TestCase):
    def test_add(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_add_forward,
            _test_add_backward,
            _test_inplace_add,
            _test_inplace_add_with_type_promotion,
        ]
        arg_dict["shape"] = [(2, 3), (2, 3, 4), (2, 3, 4, 5)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @autotest(n=5)
    def test_0_size_add(test_case):
        device = random_device()
        x = random_tensor(2, 0, 3).to(device)
        y = random_tensor(2, 1, 3).to(device)
        out = x + y
        return out

    @autotest(n=3, auto_backward=False)
    def test_0dim_inplace_add(test_case):
        device = random_device()
        x = random_tensor(2, 2, 3, requires_grad=False).to(device)
        y = random_tensor(1, 10).to(device)
        x += y.mean()
        return x

    @autotest(n=5)
    def test_0dim_two_inplace_add(test_case):
        device = random_device()
        x = random_tensor(2, 2, 3).to(device).mean()
        y = random_tensor(2, 2, 3).to(device)
        x += y.mean()
        return x

    @autotest(n=3)
    def test_add_with_alpha(test_case):
        device = random_device()
        x1 = random_tensor(2, 2, 3).to(device).mean()
        x2 = random_tensor(2, 2, 3).to(device).mean()
        x3 = random_tensor(2, 2, 3).to(device).mean()
        y = random_tensor(2, 2, 3).to(device)
        s = random().to(float)
        alpha = random().to(float)
        z1 = torch.add(x1, y, alpha=alpha)
        z2 = torch.add(x2, s, alpha=alpha)
        z3 = torch.add(s, x3, alpha=alpha)
        return z1, z2, z3

    @autotest(auto_backward=False)
    def test_bool_add(test_case):
        device = random_device()
        x = random_tensor(2, 1, 3).to(device, torch.bool)
        y = random_tensor(2, 1, 3).to(device, torch.bool)
        out = x + y
        return out

    @autotest(auto_backward=False)
    def test_0shape_bool_add(test_case):
        device = random_device()
        x = random_tensor(2, 0, 3).to(device, torch.bool)
        y = random_tensor(2, 1, 3).to(device, torch.bool)
        out = x + y
        return out

    @autotest(n=3, auto_backward=False)
    def test_0dim_bool_inplace_add(test_case):
        device = random_device()
        x = random_tensor(2, 2, 3, requires_grad=False).to(device, torch.bool)
        y = random_tensor(1, 10).to(device)
        x += y.mean().to(torch.bool)
        return x

    @autotest(auto_backward=False)
    def test_0dim_two_inplace_add(test_case):
        device = random_device()
        x = random_tensor(2, 2, 3).to(device).mean().to(torch.bool)
        y = random_tensor(2, 2, 3).to(device)
        return x
        x += y.mean().to(torch.bool)

    @autotest(n=3)
    def test_add_with_alpha_0dim(test_case):
        device = random_device()
        x1 = random_tensor(ndim=0).to(device).mean()
        x2 = random_tensor(ndim=0).to(device).mean()
        x3 = random_tensor(ndim=0).to(device).mean()
        y = random_tensor(ndim=0).to(device)
        s = random().to(float)
        alpha = random().to(float)
        z1 = torch.add(x1, y, alpha=alpha)
        z2 = torch.add(x2, s, alpha=alpha)
        z3 = torch.add(s, x3, alpha=alpha)
        return z1, z2, z3

    @profile(torch.add)
    def profile_add(test_case):
        torch.add(torch.ones(100), 20)
        torch.add(torch.ones(100), torch.ones(100, 1), alpha=10)

    @autotest(n=3)
    def test_non_contiguous_inplace_add(test_case):
        device = random_device()
        x = random_tensor(2, 2, 4).to(device)
        y = x + 1
        y = y[:, 1:3]
        y += random_tensor(2, 2, 2).to(device)
        return y

    @autotest(n=5)
    def test_scalar_add_with_random_devices(test_case):
        x1_device = random_device()
        x2_device = random_device()
        x1 = random_tensor(2, 2, 3).to(x1_device).mean()
        x2 = random_tensor(2, 2, 3).to(x2_device)
        y = x1 + x2
        return y


if __name__ == "__main__":
    unittest.main()
