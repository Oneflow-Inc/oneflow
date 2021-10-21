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
from automated_test_util import *
from scipy import special
from test_util import GenArgList

import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
class TestReLUModule(flow.unittest.TestCase):
    @autotest()
    def test_relu_module_with_random_data(test_case):
        m = torch.nn.ReLU()
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_pytorch_tensor().to(device)
        y = m(x)
        return y

    @autotest(auto_backward=False)
    def test_relu_module_with_0shape_data(test_case):
        m = torch.nn.ReLU()
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_pytorch_tensor(4, 2, 3, 0, 3).to(device)
        y = m(x)
        return y


@flow.unittest.skip_unless_1n1d()
class TestReLU6Module(flow.unittest.TestCase):
    @autotest()
    def test_relu6_module_with_random_data(test_case):
        m = torch.nn.ReLU6()
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_pytorch_tensor().to(device)
        y = m(x)
        return y

    @autotest(auto_backward=False)
    def test_relu6_module_with_0shape_data(test_case):
        m = torch.nn.ReLU6()
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_pytorch_tensor(4, 2, 3, 0, 3).to(device)
        y = m(x)
        return y


@flow.unittest.skip_unless_1n1d()
class TestTanh(flow.unittest.TestCase):
    @autotest()
    def test_tanh_module_with_random_data(test_case):
        m = torch.nn.Tanh()
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_pytorch_tensor().to(device)
        y = m(x)
        return y

    @autotest(auto_backward=False)
    def test_tanh_module_with_0shapedata(test_case):
        m = torch.nn.Tanh()
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_pytorch_tensor(4, 2, 3, 0, 3).to(device)
        y = m(x)
        return y

    @autotest()
    def test_flow_tanh_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        y = torch.tanh(x)
        return y

    @autotest(auto_backward=False)
    def test_flow_tanh_with_0shape_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(4, 2, 3, 0, 3).to(device)
        y = torch.tanh(x)
        return y


@flow.unittest.skip_unless_1n1d()
class TestELUModule(flow.unittest.TestCase):
    @autotest()
    def test_elu_module_with_random_data(test_case):
        m = torch.nn.ELU(alpha=random() | nothing())
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_pytorch_tensor().to(device)
        y = m(x)
        return y

    @autotest(auto_backward=False)
    def test_elu_module_with_0shape_data(test_case):
        m = torch.nn.ELU(alpha=random() | nothing())
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_pytorch_tensor(4, 2, 3, 0, 3).to(device)
        y = m(x)
        return y


@flow.unittest.skip_unless_1n1d()
class TestGelu(flow.unittest.TestCase):
    @autotest()
    def test_gelu_module_with_random_data(test_case):
        m = torch.nn.GELU()
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_pytorch_tensor().to(device)
        y = m(x)
        return y


def numpy_softmax(x, axis):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def numpy_logsoftmax(x, dim):
    e_x = np.exp(x - np.max(x, axis=dim, keepdims=True))
    return np.log(e_x / e_x.sum(axis=dim, keepdims=True))


def numpy_softplus(x, beta, threshold):
    return np.where(
        x * beta > threshold, x, 1.0 / beta * np.log(1.0 + np.exp(beta * x))
    )


def numpy_mish_grad(x):
    f = 1 + np.exp(x)
    y_grad = (f * f - 1) / (f * f + 1) + x * (4 * f * (f - 1)) / (
        (f * f + 1) * (f * f + 1)
    )
    return y_grad


@flow.unittest.skip_unless_1n1d()
class TestSigmoid(flow.unittest.TestCase):
    @autotest()
    def test_sigmoid_module_with_random_data(test_case):
        m = torch.nn.Sigmoid()
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_pytorch_tensor().to(device)
        y = m(x)
        return y

    @autotest()
    def test_sigmoid_flow_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        y = torch.sigmoid(x)
        return y

    @autotest()
    def test_sigmoid_tensor_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        y = x.sigmoid()
        return y


def _test_softmax(test_case, device):
    axis = 0
    m = flow.nn.Softmax(dim=axis)
    arr = np.random.randn(2, 3, 4, 5)
    x = flow.Tensor(arr, device=flow.device(device))
    y = m(x)
    output = numpy_softmax(arr, axis)
    test_case.assertTrue(np.allclose(y.numpy(), output, 1e-05, 1e-05))


def _test_softmax_dim_1(test_case, device):
    axis = 1
    m = flow.nn.Softmax(dim=axis)
    arr = np.random.randn(9, 7, 8, 16)
    x = flow.Tensor(arr, device=flow.device(device))
    y = m(x)
    output = numpy_softmax(arr, axis)
    test_case.assertTrue(np.allclose(y.numpy(), output, 1e-05, 1e-05))


def _test_softmax_dim_2(test_case, device):
    axis = 2
    m = flow.nn.Softmax(dim=axis)
    arr = np.random.randn(2, 5, 6, 3)
    x = flow.Tensor(arr, device=flow.device(device))
    y = m(x)
    output = numpy_softmax(arr, axis)
    test_case.assertTrue(np.allclose(y.numpy(), output, 1e-05, 1e-05))


def _test_softmax_dim_3(test_case, device):
    axis = 3
    m = flow.nn.Softmax(dim=axis)
    arr = np.random.randn(1, 3, 4, 7)
    x = flow.Tensor(arr, device=flow.device(device))
    y = m(x)
    output = numpy_softmax(arr, axis)
    test_case.assertTrue(np.allclose(y.numpy(), output, 1e-05, 1e-05))
    axis2 = -1
    m2 = flow.nn.Softmax(dim=axis)
    y2 = m(x)
    output2 = numpy_softmax(arr, axis)
    test_case.assertTrue(np.allclose(y2.numpy(), output2, 1e-05, 1e-05))


def _test_softmax_backward_normal(test_case, device):
    x_grad = np.zeros((2, 3, 4, 5))
    axis = 0
    m = flow.nn.Softmax(dim=axis)
    x = flow.Tensor(
        np.random.randn(2, 3, 4, 5),
        requires_grad=True,
        device=flow.device(device),
        dtype=flow.float64,
    )
    y = m(x).sum()
    y.backward()
    test_case.assertTrue(np.allclose(x.grad.numpy(), x_grad, 1e-05, 1e-05))


def _test_softmax_backward_1_dim(test_case, device):
    a = flow.tensor(
        [1, 2], dtype=flow.float64, device=flow.device(device), requires_grad=True
    )
    b = flow.tensor(
        [3, 4], dtype=flow.float64, device=flow.device(device), requires_grad=True
    )
    c = a * b
    m = flow.nn.Softmax(dim=None)
    d = m(c)
    d[0].backward()
    a_grad = np.array([0.01994417, -0.0265922267])
    test_case.assertTrue(np.allclose(a.grad.numpy(), a_grad, 1e-05, 1e-05))


@flow.unittest.skip_unless_1n1d()
class TestSoftmax(flow.unittest.TestCase):
    def test_softmax(test_case):
        arg_dict = OrderedDict()
        arg_dict["fun"] = [
            _test_softmax,
            _test_softmax_dim_1,
            _test_softmax_dim_2,
            _test_softmax_dim_3,
            _test_softmax_backward_normal,
            _test_softmax_backward_1_dim,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


@flow.unittest.skip_unless_1n1d()
class TestHardsigmoidModule(flow.unittest.TestCase):
    @autotest()
    def test_hardsigmoid_module_with_random_data(test_case):
        m = torch.nn.Hardsigmoid()
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_pytorch_tensor().to(device)
        y = m(x)
        return y


def _test_logsoftmax(test_case, device):
    dim = 1
    m = flow.nn.LogSoftmax(dim)
    input_arr = np.random.randn(4, 7)
    x = flow.Tensor(input_arr, device=flow.device(device))
    y = m(x)
    output = numpy_logsoftmax(input_arr, dim)
    test_case.assertTrue(np.allclose(y.numpy(), output, 1e-05, 1e-05))


def _test_logsoftmax_dim_2(test_case, device):
    dim = 2
    m = flow.nn.LogSoftmax(dim)
    input_arr = np.random.randn(3, 4, 5)
    x = flow.Tensor(input_arr, device=flow.device(device))
    y = m(x)
    output = numpy_logsoftmax(input_arr, dim)
    test_case.assertTrue(np.allclose(y.numpy(), output, 1e-05, 1e-05))


def _test_logsoftmax_dim_3(test_case, device):
    dim = 3
    m = flow.nn.LogSoftmax(dim)
    input_arr = np.random.randn(8, 9, 7, 3)
    x = flow.Tensor(input_arr, device=flow.device(device))
    y = m(x)
    output = numpy_logsoftmax(input_arr, dim)
    test_case.assertTrue(np.allclose(y.numpy(), output, 1e-05, 1e-05))


def _test_logsoftmax_backward(test_case, device):
    axis = 0
    m = flow.nn.LogSoftmax(axis)
    input_arr = np.array(
        [
            [
                [
                    [2.0, 1.0, 9.0, 3.0, 4.0],
                    [1.0, 6.0, 7.0, 1.0, 4.0],
                    [4.0, 7.0, 5.0, 8.0, 1.0],
                    [9.0, 5.0, 7.0, 8.0, 5.0],
                ],
                [
                    [1.0, 1.0, 5.0, 3.0, 5.0],
                    [3.0, 6.0, 3.0, 7.0, 8.0],
                    [8.0, 8.0, 1.0, 2.0, 6.0],
                    [3.0, 5.0, 6.0, 1.0, 1.0],
                ],
                [
                    [8.0, 3.0, 6.0, 3.0, 7.0],
                    [8.0, 5.0, 1.0, 2.0, 7.0],
                    [3.0, 9.0, 4.0, 6.0, 5.0],
                    [5.0, 1.0, 2.0, 3.0, 6.0],
                ],
            ],
            [
                [
                    [3.0, 5.0, 3.0, 1.0, 7.0],
                    [5.0, 2.0, 6.0, 3.0, 5.0],
                    [5.0, 1.0, 8.0, 6.0, 9.0],
                    [9.0, 8.0, 4.0, 5.0, 1.0],
                ],
                [
                    [7.0, 5.0, 7.0, 1.0, 6.0],
                    [3.0, 3.0, 6.0, 6.0, 7.0],
                    [9.0, 4.0, 1.0, 5.0, 7.0],
                    [7.0, 6.0, 9.0, 8.0, 6.0],
                ],
                [
                    [6.0, 7.0, 5.0, 3.0, 9.0],
                    [4.0, 1.0, 2.0, 3.0, 2.0],
                    [4.0, 3.0, 8.0, 7.0, 8.0],
                    [1.0, 3.0, 8.0, 6.0, 2.0],
                ],
            ],
        ]
    )
    x = flow.Tensor(
        input_arr, requires_grad=True, device=flow.device(device), dtype=flow.float64
    )
    x_grad = np.array(
        [
            [
                [
                    [0.46211716, 0.96402758, -0.99505475, -0.76159416, 0.90514825],
                    [0.96402758, -0.96402758, -0.46211716, 0.76159416, 0.46211716],
                    [0.46211716, -0.99505475, 0.90514825, -0.76159416, 0.9993293],
                    [0.0, 0.90514825, -0.90514825, -0.90514825, -0.96402758],
                ],
                [
                    [0.99505475, 0.96402758, 0.76159416, -0.76159416, 0.46211716],
                    [0.0, -0.90514825, 0.90514825, -0.46211716, -0.46211716],
                    [0.46211716, -0.96402758, 0.0, 0.90514825, 0.46211716],
                    [0.96402758, 0.46211716, 0.90514825, 0.9981779, 0.9866143],
                ],
                [
                    [-0.76159416, 0.96402758, -0.46211716, 0.0, 0.76159416],
                    [-0.96402758, -0.96402758, 0.46211716, 0.46211716, -0.9866143],
                    [0.46211716, -0.99505475, 0.96402758, 0.46211716, 0.90514825],
                    [-0.96402758, 0.76159416, 0.99505475, 0.90514825, -0.96402758],
                ],
            ],
            [
                [
                    [-0.46211716, -0.96402758, 0.99505475, 0.76159416, -0.90514825],
                    [-0.96402758, 0.96402758, 0.46211716, -0.76159416, -0.46211716],
                    [-0.46211716, 0.99505475, -0.90514825, 0.76159416, -0.9993293],
                    [0.0, -0.90514825, 0.90514825, 0.90514825, 0.96402758],
                ],
                [
                    [-0.99505475, -0.96402758, -0.76159416, 0.76159416, -0.46211716],
                    [0.0, 0.90514825, -0.90514825, 0.46211716, 0.46211716],
                    [-0.46211716, 0.96402758, 0.0, -0.90514825, -0.46211716],
                    [-0.96402758, -0.46211716, -0.90514825, -0.9981779, -0.9866143],
                ],
                [
                    [0.76159416, -0.96402758, 0.46211716, 0.0, -0.76159416],
                    [0.96402758, 0.96402758, -0.46211716, -0.46211716, 0.9866143],
                    [-0.46211716, 0.99505475, -0.96402758, -0.46211716, -0.90514825],
                    [0.96402758, -0.76159416, -0.99505475, -0.90514825, 0.96402758],
                ],
            ],
        ]
    )
    y = m(x).sum()
    y.backward()
    test_case.assertTrue(np.allclose(x.grad.numpy(), x_grad, 1e-05, 1e-05))


@flow.unittest.skip_unless_1n1d()
class TestLogSoftmax(flow.unittest.TestCase):
    def test_log_softmax(test_case):
        arg_dict = OrderedDict()
        arg_dict["fun"] = [
            _test_logsoftmax,
            _test_logsoftmax_dim_2,
            _test_logsoftmax_dim_3,
            _test_logsoftmax_backward,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


@flow.unittest.skip_unless_1n1d()
class TestLogSigmoidModule(flow.unittest.TestCase):
    @autotest()
    def test_logsigmoid_module_with_random_data(test_case):
        m = torch.nn.LogSigmoid()
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_pytorch_tensor().to(device)
        y = m(x)
        return y


def _test_softplus(test_case, device):
    m = flow.nn.Softplus()
    arr = np.random.randn(2, 3, 4, 5)
    np_out = numpy_softplus(arr, 1.0, 20)
    x = flow.Tensor(arr, device=flow.device(device))
    of_out = m(x)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))


def _test_softplus_beta(test_case, device):
    m = flow.nn.Softplus(beta=1.11)
    arr = np.random.randn(2, 3, 4, 5)
    np_out = numpy_softplus(arr, 1.11, 20)
    x = flow.Tensor(arr, device=flow.device(device))
    of_out = m(x)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))


def _test_softplus_threshold(test_case, device):
    m = flow.nn.Softplus(beta=1.11, threshold=1.55)
    arr = np.random.randn(2, 3, 4, 5)
    np_out = np.where(
        arr * 1.11 > 1.55, arr, 1.0 / 1.11 * np.log(1.0 + np.exp(1.11 * arr))
    )
    np_out = numpy_softplus(arr, 1.11, 1.55)
    x = flow.Tensor(arr, device=flow.device(device))
    of_out = m(x)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))


def _test_softplus_backward(test_case, device):
    m = flow.nn.Softplus()
    arr = np.array([1.0, 2.0, 21.0, 20.0, 4.0])
    x = flow.Tensor(arr, device=flow.device(device), requires_grad=True)
    of_out = m(x)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [0.7310585786300049, 0.8807970779778824, 1.0, 1.0, 0.9820137900379085]
    test_case.assertTrue(np.allclose(x.grad.numpy(), np_grad, 1e-05, 1e-05))


@flow.unittest.skip_unless_1n1d()
class TestSoftplusModule(flow.unittest.TestCase):
    def test_softplus(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_softplus,
            _test_softplus_beta,
            _test_softplus_threshold,
            _test_softplus_backward,
        ]
        arg_dict["device"] = ["cpu"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @unittest.skip("pytorch softplus backward has bug")
    @autotest()
    def test_softplus_module_with_random_data(test_case):
        m = torch.nn.Softplus(beta=random() | nothing(), threshold=random() | nothing())
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_pytorch_tensor().to(device)
        y = m(x)
        return y


@flow.unittest.skip_unless_1n1d()
class TestHardswishModule(flow.unittest.TestCase):
    @autotest()
    def test_hardswish_module_with_random_data(test_case):
        m = torch.nn.Hardswish()
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_pytorch_tensor().to(device)
        y = m(x)
        return y


def _np_hardtanh_grad(x):
    return np.where(x <= -2.0, 0.0, np.where(x >= 2.3, 0.0, 1.0))


def _test_hardtanh_impl(test_case, shape, device):
    m = flow.nn.Hardtanh()
    arr = np.random.randn(*shape)
    np_out = np.maximum(-1, np.minimum(1, arr))
    x = flow.Tensor(arr, device=flow.device(device))
    of_out = m(x)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    m = flow.nn.Hardtanh(min_val=-2.0, max_val=2.3)
    arr = np.random.randn(*shape)
    np_out = np.maximum(-2.0, np.minimum(2.3, arr))
    x = flow.Tensor(arr, device=flow.device(device), requires_grad=True)
    of_out = m(x)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
    of_out = of_out.sum()
    of_out.backward()
    test_case.assertTrue(
        np.allclose(x.grad.numpy(), _np_hardtanh_grad(np_out), 1e-05, 1e-05)
    )


@flow.unittest.skip_unless_1n1d()
class TestHardtanhModule(flow.unittest.TestCase):
    def test_hardtanh(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 3), (2, 3, 4), (2, 4, 5, 6)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_hardtanh_impl(test_case, *arg)


@flow.unittest.skip_unless_1n1d()
class TestLeakyReLUModule(flow.unittest.TestCase):
    @autotest()
    def test_leakyrelu_module_with_random_data(test_case):
        m = torch.nn.LeakyReLU(negative_slope=random() | nothing())
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_pytorch_tensor().to(device)
        y = m(x)
        return y


@flow.unittest.skip_unless_1n1d()
class TestMishModule(flow.unittest.TestCase):
    @autotest(n=5)
    def test_mish_module_with_random_data(test_case):
        m = torch.nn.Mish()
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_pytorch_tensor().to(device)
        y = m(x)
        return y


@flow.unittest.skip_unless_1n1d()
class TestSiluModule(flow.unittest.TestCase):
    @autotest(n=5)
    def test_silu_module_with_random_data(test_case):
        m = torch.nn.SiLU()
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_pytorch_tensor().to(device)
        y = m(x)
        return y


@flow.unittest.skip_unless_1n1d()
class TestSeluModule(flow.unittest.TestCase):
    @autotest(n=5)
    def test_selu_module_with_random_data(test_case):
        m = torch.nn.SELU()
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_pytorch_tensor().to(device)
        y = m(x)
        return y


def _np_softsign(x):
    return x / (1.0 + np.abs(x))


def _np_softsign_grad(x):
    return 1.0 / (np.square(1.0 + np.abs(x)))


def _test_softsign_impl(test_case, shape, device):
    m = flow.nn.Softsign()
    np_input = np.random.randn(*shape)
    np_out = _np_softsign(np_input)
    of_input = flow.Tensor(
        np_input, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    of_out = m(of_input)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-3, 1e-3))

    of_out = of_out.sum()
    of_out.backward()
    test_case.assertTrue(
        np.allclose(of_input.grad.numpy(), _np_softsign_grad(np_input), 1e-3, 1e-3)
    )


@unittest.skip("still have error in ci test")
class TestSoftsignModule(flow.unittest.TestCase):
    @autotest(n=5)
    def test_softsign_module_with_random_data(test_case):
        m = torch.nn.Softsign()
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_pytorch_tensor().to(device)
        y = m(x)
        return y

    def test_softsign(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_softsign_impl]
        arg_dict["shape"] = [(3, 3), (2, 3, 3)]

        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
