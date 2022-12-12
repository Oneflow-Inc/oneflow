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


def _test_where(test_case, device):
    x = flow.tensor(
        np.array([[-0.462, 0.3139], [0.3898, -0.7197], [0.0478, -0.1657]]),
        dtype=flow.float32,
        device=flow.device(device),
    )
    y = flow.tensor(
        np.ones(shape=(3, 2)), dtype=flow.float32, device=flow.device(device)
    )
    condition = flow.tensor(
        np.array([[0, 1], [1, 0], [1, 0]]), dtype=flow.int32, device=flow.device(device)
    )
    of_out = flow.where(condition, x, y)
    np_out = np.array([[1.0, 0.3139], [0.3898, 1.0], [0.0478, 1.0]])
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))


def _test_where_broadcast(test_case, device):
    x = flow.tensor(
        np.array([[[-0.462, 0.3139], [0.3898, -0.7197], [0.0478, -0.1657]]]),
        dtype=flow.float32,
        device=flow.device(device),
    )
    y = flow.tensor(
        np.ones(shape=(3, 3, 2)), dtype=flow.float32, device=flow.device(device)
    )
    condition = flow.tensor(
        np.array([[[0, 1], [1, 0], [1, 0]]]),
        dtype=flow.int32,
        device=flow.device(device),
    )
    of_out = flow.where(condition, x, y)
    np_out = np.array(
        [
            [[1.0, 0.3139], [0.3898, 1.0], [0.0478, 1.0]],
            [[1.0, 0.3139], [0.3898, 1.0], [0.0478, 1.0]],
            [[1.0, 0.3139], [0.3898, 1.0], [0.0478, 1.0]],
        ]
    )
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))


def _test_where_scalar(test_case, device):
    x = 0.5
    y = 2.0
    condition = flow.tensor(np.array([1]), dtype=flow.int32)
    of_out = flow.where(condition, x, y)
    test_case.assertTrue(of_out.dtype == flow.float32)
    np_out = np.array([0.5])
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    flow.set_default_dtype(flow.double)
    of_out = flow.where(condition, x, y)
    test_case.assertTrue(of_out.dtype == flow.double)
    flow.set_default_dtype(flow.float16)
    of_out = flow.where(condition, x, y)
    test_case.assertTrue(of_out.dtype == flow.float16)
    flow.set_default_dtype(flow.bfloat16)
    of_out = flow.where(condition, x, y)
    test_case.assertTrue(of_out.dtype == flow.bfloat16)


def _test_where_dim4(test_case, device):
    x = flow.tensor(
        np.array([[[[-0.462, 0.3139], [0.3898, -0.7197], [0.0478, -0.1657]]]]),
        dtype=flow.float32,
        device=flow.device(device),
    )
    y = flow.tensor(
        np.ones(shape=(1, 1, 3, 2)), dtype=flow.float32, device=flow.device(device)
    )
    condition = flow.tensor(
        np.array([[[[0, 1], [1, 0], [1, 0]]]]),
        dtype=flow.int32,
        device=flow.device(device),
    )
    of_out = flow.where(condition, x, y)
    np_out = np.array([[[[1.0, 0.3139], [0.3898, 1.0], [0.0478, 1.0]]]])
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))


def _test_where_backward(test_case, device):
    x = flow.tensor(
        np.array([[-0.462, 0.3139], [0.3898, -0.7197], [0.0478, -0.1657]]),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    y = flow.tensor(
        np.ones(shape=(3, 2)),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    condition = flow.tensor(
        np.array([[0, 1], [1, 0], [1, 0]]), dtype=flow.int32, device=flow.device(device)
    )
    of_out = flow.where(condition, x, y)
    of_out = of_out.sum()
    of_out.backward()
    test_case.assertTrue(
        np.allclose(x.grad.numpy(), condition.numpy() == 1, 1e-05, 1e-05)
    )
    test_case.assertTrue(
        np.allclose(y.grad.numpy(), condition.numpy() == 0, 1e-05, 1e-05)
    )


def _test_where_broadcast_backward(test_case, device):
    x = flow.tensor(
        np.array([[[-0.462, 0.3139], [0.3898, -0.7197], [0.0478, -0.1657]]]),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    y = flow.tensor(
        np.ones(shape=(3, 3, 2)),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    condition = flow.tensor(
        np.array([[[0, 1], [1, 0], [1, 0]]]),
        dtype=flow.int32,
        device=flow.device(device),
    )
    of_out = flow.where(condition, x, y)
    of_out = of_out.sum()
    of_out.backward()
    x_grad = [[[0.0, 3.0], [3.0, 0.0], [3.0, 0.0]]]
    test_case.assertTrue(np.allclose(x.grad.numpy(), x_grad, 1e-05, 1e-05))
    y_grad = [
        [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
    ]
    test_case.assertTrue(np.allclose(y.grad.numpy(), y_grad, 1e-05, 1e-05))


def _test_where_broadcast_x_backward(test_case, device):
    x = flow.tensor(
        np.array([[[-0.462, 0.3139], [0.3898, -0.7197], [0.0478, -0.1657]]]),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    y = flow.tensor(
        np.ones(shape=(3, 3, 2)), dtype=flow.float32, device=flow.device(device)
    )
    condition = flow.tensor(
        np.array([[[0, 1], [1, 0], [1, 0]]]),
        dtype=flow.int32,
        device=flow.device(device),
    )
    of_out = flow.where(condition, x, y)
    of_out = of_out.sum()
    of_out.backward()
    x_grad = [[[0.0, 3.0], [3.0, 0.0], [3.0, 0.0]]]
    test_case.assertTrue(np.allclose(x.grad.numpy(), x_grad, 1e-05, 1e-05))


def _test_where_x_y_none(test_case, device):
    condition = flow.tensor(
        np.array([[[-0.462, 0.3139], [0.3898, -0.7197], [0.0478, -0.1657]]]),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    of_out = flow.where(condition)
    of_nonzero = flow.nonzero(condition, as_tuple=True)
    for i in range(len(of_out)):
        test_case.assertTrue(
            np.allclose(of_out[i].numpy(), of_nonzero[i].numpy(), 1e-05, 1e-05)
        )


def _test_where_scalar(test_case, device):
    x = flow.randn(5, 5)
    y = flow.where(x > 0, x, 0.0)
    test_case.assertTrue(np.array_equal(y.size(), (5, 5)))
    y = flow.where(x > 0, 0.0, x)
    test_case.assertTrue(np.array_equal(y.size(), (5, 5)))


@flow.unittest.skip_unless_1n1d()
class TestWhere(flow.unittest.TestCase):
    def test_where(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_where,
            _test_where_broadcast,
            _test_where_scalar,
            _test_where_dim4,
            _test_where_backward,
            _test_where_broadcast_backward,
            _test_where_broadcast_x_backward,
            _test_where_x_y_none,
            _test_where_scalar,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @autotest(n=5)
    def test_flow_where_tensor_with_random_data(test_case):
        k1 = random(2, 6)
        k2 = random(2, 6)
        device = random_device()
        cond = random_tensor(ndim=2, dim0=k1, dim1=k2).to(device)
        x = random_tensor(ndim=2, dim0=k1, dim1=k2).to(device)
        y = random_tensor(ndim=2, dim0=k1, dim1=k2).to(device)
        return torch.where(cond > 0, x, y)

    @autotest(n=5)
    def test_flow_where_tensor_with_0dim_data(test_case):
        k1 = random(2, 6)
        k2 = random(2, 6)
        device = random_device()
        cond = random_tensor(ndim=2, dim0=k1, dim1=k2).to(device)
        x = random_tensor(ndim=0).to(device)
        y = random_tensor(ndim=0).to(device)
        return torch.where(cond > 0, x, y)

    @autotest(n=5)
    def test_flow_where_tensor_broadcast_with_random_data(test_case):
        k1 = random(2, 6)
        k2 = random(2, 6)
        device = random_device()
        cond = random_tensor(ndim=2, dim0=k1, dim1=k2).to(device)
        x = random_tensor(ndim=2, dim0=1, dim1=k2).to(device)
        y = random_tensor(ndim=2, dim0=k1, dim1=1).to(device)
        return torch.where(cond > 0, x, y)

    @autotest(n=5)
    def test_flow_where_scalar_x_with_random_data(test_case):
        k1 = random(2, 6)
        k2 = random(2, 6)
        device = random_device()
        cond = random_tensor(ndim=2, dim0=k1, dim1=k2).to(device)
        x = random().to(float)
        y = random_tensor(ndim=2, dim0=k1, dim1=k2, dtype=float).to(
            device=device, dtype=torch.float64
        )
        return torch.where(cond > 0, x, y)

    @autotest(n=5)
    def test_flow_where_scalar_x_broadcast_with_random_data(test_case):
        k1 = random(2, 6)
        k2 = random(2, 6)
        device = random_device()
        cond = random_tensor(ndim=2, dim0=1, dim1=k2).to(device)
        x = random().to(float)
        y = random_tensor(ndim=2, dim0=k1, dim1=1, dtype=float).to(
            device=device, dtype=torch.float64
        )
        return torch.where(cond > 0, x, y)

    @autotest(n=5, auto_backward=False, check_graph=True)
    def test_flow_where_scalar_x_int_with_random_data(test_case):
        k1 = random(2, 6)
        k2 = random(2, 6)
        device = random_device()
        cond = random_tensor(ndim=2, dim0=k1, dim1=k2).to(device)
        x = random().to(int)
        y = random_tensor(ndim=2, dim0=k1, dim1=k2, dtype=int).to(device)
        return torch.where(cond > 0, x, y)

    @autotest(n=5)
    def test_flow_where_scalar_y_with_random_data(test_case):
        k1 = random(2, 6)
        k2 = random(2, 6)
        device = random_device()
        cond = random_tensor(ndim=2, dim0=k1, dim1=k2).to(device)
        x = random_tensor(ndim=2, dim0=k1, dim1=k2, dtype=float).to(
            device=device, dtype=torch.float64
        )
        y = random().to(float)
        return torch.where(cond > 0, x, y)

    @autotest(n=5)
    def test_flow_where_scalar_y_broadcast_with_random_data(test_case):
        k1 = random(2, 6)
        k2 = random(2, 6)
        device = random_device()
        cond = random_tensor(ndim=2, dim0=1, dim1=k2).to(device)
        x = random_tensor(ndim=2, dim0=k1, dim1=1, dtype=float).to(
            device=device, dtype=torch.float64
        )
        y = random().to(float)
        return torch.where(cond > 0, x, y)

    @autotest(n=5, auto_backward=False, check_graph=True)
    def test_flow_where_scalar_y_int_with_random_data(test_case):
        k1 = random(2, 6)
        k2 = random(2, 6)
        device = random_device()
        cond = random_tensor(ndim=2, dim0=k1, dim1=k2).to(device)
        x = random_tensor(ndim=2, dim0=k1, dim1=k2, dtype=int).to(device)
        y = random().to(int)
        return torch.where(cond > 0, x, y)

    @autotest(n=5, auto_backward=False, check_graph=True)
    def test_flow_where_scalar_xy_with_random_data(test_case):
        k1 = random(2, 6)
        k2 = random(2, 6)
        device = random_device()
        cond = random_tensor(ndim=2, dim0=k1, dim1=k2).to(device)
        x = random().to(float)
        y = random().to(float)
        return torch.where(cond > 0, x, y)

    @autotest(n=5, auto_backward=False, check_graph=True)
    def test_flow_where_scalar_xy_int_with_random_data(test_case):
        k1 = random(2, 6)
        k2 = random(2, 6)
        device = random_device()
        cond = random_tensor(ndim=2, dim0=k1, dim1=k2).to(device)
        x = random().to(int)
        y = random().to(int)
        return torch.where(cond > 0, x, y)

    @autotest(n=5, auto_backward=False, check_graph=True)
    def test_flow_where_tensor_bool_with_random_data(test_case):
        k1 = random(2, 6)
        k2 = random(2, 6)
        device = random_device()
        cond = random_tensor(ndim=2, dim0=k1, dim1=k2).to(device)
        x = random_tensor(ndim=2, dim0=k1, dim1=k2).to(device=device, dtype=torch.bool)
        y = random_tensor(ndim=2, dim0=k1, dim1=k2).to(device=device, dtype=torch.bool)
        return torch.where(cond > 0, x, y)

    @autotest(n=5, auto_backward=False, check_graph=True)
    def test_flow_where_tensor_broadcast_bool_with_random_data(test_case):
        k1 = random(2, 6)
        k2 = random(2, 6)
        device = random_device()
        cond = random_tensor(ndim=2, dim0=k1, dim1=k2).to(device)
        x = random_tensor(ndim=2, dim0=1, dim1=k2).to(device=device, dtype=torch.bool)
        y = random_tensor(ndim=2, dim0=k1, dim1=1).to(device=device, dtype=torch.bool)
        return torch.where(cond > 0, x, y)

    @autotest(n=5, auto_backward=False, check_graph=True)
    def test_flow_where_scalar_x_bool_with_random_data(test_case):
        k1 = random(2, 6)
        k2 = random(2, 6)
        device = random_device()
        cond = random_tensor(ndim=2, dim0=k1, dim1=k2).to(device)
        x = random().to(bool)
        y = random_tensor(ndim=2, dim0=k1, dim1=k2, dtype=float).to(
            device=device, dtype=torch.bool
        )
        return torch.where(cond > 0, x, y)

    @autotest(n=5, auto_backward=False, check_graph=True)
    def test_flow_where_scalar_x_broadcast_bool_with_random_data(test_case):
        k1 = random(2, 6)
        k2 = random(2, 6)
        device = random_device()
        cond = random_tensor(ndim=2, dim0=1, dim1=k2).to(device)
        x = random().to(bool)
        y = random_tensor(ndim=2, dim0=k1, dim1=1, dtype=float).to(
            device=device, dtype=torch.bool
        )
        return torch.where(cond > 0, x, y)

    @autotest(n=5, auto_backward=False, check_graph=True)
    def test_flow_where_scalar_y_bool_with_random_data(test_case):
        k1 = random(2, 6)
        k2 = random(2, 6)
        device = random_device()
        cond = random_tensor(ndim=2, dim0=k1, dim1=k2).to(device)
        x = random_tensor(ndim=2, dim0=k1, dim1=k2, dtype=float).to(
            device=device, dtype=torch.bool
        )
        y = random().to(bool)
        return torch.where(cond > 0, x, y)

    @autotest(n=5, auto_backward=False, check_graph=True)
    def test_flow_where_scalar_y_broadcast_bool_with_random_data(test_case):
        k1 = random(2, 6)
        k2 = random(2, 6)
        device = random_device()
        cond = random_tensor(ndim=2, dim0=1, dim1=k2).to(device)
        x = random_tensor(ndim=2, dim0=k1, dim1=1, dtype=float).to(
            device=device, dtype=torch.bool
        )
        y = random().to(bool)
        return torch.where(cond > 0, x, y)

    @autotest(n=5, auto_backward=False, check_graph=True)
    def test_flow_where_scalar_xy_bool_with_random_data(test_case):
        k1 = random(2, 6)
        k2 = random(2, 6)
        device = random_device()
        cond = random_tensor(ndim=2, dim0=k1, dim1=k2).to(device)
        x = random().to(bool)
        y = random().to(bool)
        return torch.where(cond > 0, x, y)


if __name__ == "__main__":
    unittest.main()
