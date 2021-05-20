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
from test_util import GenArgList
from collections import OrderedDict
import numpy as np
import oneflow.experimental as flow


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestReLUModule(flow.unittest.TestCase):
    def test_relu(test_case):
        m = flow.nn.ReLU()
        arr = np.random.randn(2, 3, 4, 5)

        np_out = np.maximum(0, arr)
        x = flow.Tensor(arr)
        of_out = m(x)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, rtol=1e-05))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestReLU6Module(flow.unittest.TestCase):
    def test_relu6(test_case):
        m = flow.nn.ReLU6()
        arr = np.random.randn(2, 3, 4, 5)

        np_out = np.minimum(np.maximum(0, arr), 6.0)
        x = flow.Tensor(arr)
        of_out = m(x)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, rtol=1e-05))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestTanhModule(flow.unittest.TestCase):
    def _test_body_tanh(test_case, input_arr):
        x = flow.Tensor(input_arr)

        tanh = flow.nn.Tanh()
        y = tanh(x)
        z = np.tanh(input_arr)

        test_case.assertTrue(np.allclose(y.numpy(), z, rtol=1e-4, atol=1e-4))

    def _test_ones_body_tanh(self, shape):
        x = np.ones(shape, dtype=np.float32)
        self._test_body_tanh(x)

    def _test_random_body_tanh(self, shape):
        x = np.random.random(shape).astype(np.float32)
        self._test_body_tanh(x)

    def test_ones_input_tanh(self):
        self._test_ones_body_tanh((1))
        self._test_ones_body_tanh((1, 10))
        self._test_ones_body_tanh((2, 10, 2))
        self._test_ones_body_tanh((2, 5, 2, 2))

    def test_random_input_tanh(self):
        self._test_random_body_tanh((1))
        self._test_random_body_tanh((1, 10))
        self._test_random_body_tanh((2, 10, 2))
        self._test_random_body_tanh((2, 5, 2, 2))

    def _test_body_tanh_v2(test_case, input_arr):
        x = flow.Tensor(input_arr)

        y = flow.tanh(x)
        z = np.tanh(input_arr)

        test_case.assertTrue(np.allclose(y.numpy(), z, rtol=1e-4, atol=1e-4))

    def _test_body_tanh_v3(test_case, input_arr):
        x = flow.Tensor(input_arr)

        y = x.tanh()
        z = np.tanh(input_arr)

        test_case.assertTrue(np.allclose(y.numpy(), z, rtol=1e-4, atol=1e-4))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestELUModule(flow.unittest.TestCase):
    def test_elu(test_case):
        m = flow.nn.ELU()
        arr = np.random.randn(2, 3, 4, 5)
        np_out = np.where(arr > 0, arr, 1.0 * (np.exp(arr) - 1))
        x = flow.Tensor(arr)
        of_out = m(x)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, rtol=1e-4, atol=1e-4))

    def test_elu_alpha(test_case):
        m = flow.nn.ELU(alpha=1.2)
        arr = np.random.randn(2, 3, 4, 5)
        np_out = np.where(arr > 0, arr, 1.2 * (np.exp(arr) - 1))
        x = flow.Tensor(arr)
        of_out = m(x)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, rtol=1e-4, atol=1e-4))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestGeLU(flow.unittest.TestCase):
    def test_gelu_v1(test_case):
        input_arr = np.array([-0.5, 0, 0.5]).astype(np.float32)
        x = flow.Tensor(input_arr)

        gelu = flow.nn.GELU()
        y = gelu(x)
        z = np.array([-0.15426877, 0.0, 0.34573123])

        test_case.assertTrue(np.allclose(y.numpy(), z, rtol=1e-4, atol=1e-4))

    def test_gelu_v2(test_case):
        input_arr = np.array([-0.5, 0, 0.5]).astype(np.float32)
        x = flow.Tensor(input_arr)

        y = flow.gelu(x)
        z = np.array([-0.15426877, 0.0, 0.34573123])

        test_case.assertTrue(np.allclose(y.numpy(), z, rtol=1e-4, atol=1e-4))

    def test_gelu_v3(test_case):
        input_arr = np.array([-0.5, 0, 0.5]).astype(np.float32)
        x = flow.Tensor(input_arr)

        y = x.gelu()

        z = np.array([-0.15426877, 0.0, 0.34573123])

        test_case.assertTrue(np.allclose(y.numpy(), z, rtol=1e-4, atol=1e-4))


def numpy_sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def numpy_sigmoid_grad(inputs, grads):
    x = np.exp(-inputs)
    delta = x / (1 + x) ** 2
    return delta * grads


def numpy_softmax(x, axis):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def numpy_logsoftmax(x, dim):
    e_x = np.exp(x - np.max(x, axis=dim, keepdims=True))
    return np.log(e_x / e_x.sum(axis=dim, keepdims=True))


def _test_sigmoid(test_case, device):
    m = flow.nn.Sigmoid()
    input_arr = np.random.randn(2, 3, 4, 5)
    x = flow.Tensor(input_arr, device=flow.device(device))

    y = m(x)
    y2 = flow.sigmoid(x)
    y3 = x.sigmoid()
    output = numpy_sigmoid(input_arr)

    test_case.assertTrue(np.allclose(y.numpy(), output, rtol=1e-05))
    test_case.assertTrue(np.allclose(y2.numpy(), output, rtol=1e-05))
    test_case.assertTrue(np.allclose(y3.numpy(), output, rtol=1e-05))


def _test_sigmoid_backward(test_case, device):
    input_arr = np.random.randn(2, 3, 4, 5)
    x = flow.Tensor(input_arr, device=flow.device(device), requires_grad=True)
    x_grad = numpy_sigmoid_grad(input_arr, np.ones(input_arr.shape))
    m = flow.nn.Sigmoid()
    y = m(x).sum()
    y.backward()
    test_case.assertTrue(np.allclose(x.grad.numpy(), x_grad, rtol=1e-05))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestSigmoid(flow.unittest.TestCase):
    def test_sigmoid(test_case):
        arg_dict = OrderedDict()
        arg_dict["fun"] = [
            _test_sigmoid,
            _test_sigmoid_backward,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


def _test_softmax(test_case, device):
    axis = 0
    m = flow.nn.Softmax(dim=axis)
    arr = np.random.randn(2, 3, 4, 5)
    x = flow.Tensor(arr, device=flow.device(device))
    y = m(x)
    output = numpy_softmax(arr, axis)
    test_case.assertTrue(np.allclose(y.numpy(), output, rtol=1e-05))


def _test_softmax_dim_1(test_case, device):
    axis = 1
    m = flow.nn.Softmax(dim=axis)
    arr = np.random.randn(9, 7, 8, 16)
    x = flow.Tensor(arr, device=flow.device(device))
    y = m(x)
    output = numpy_softmax(arr, axis)
    test_case.assertTrue(np.allclose(y.numpy(), output, rtol=1e-05))


def _test_softmax_dim_2(test_case, device):
    axis = 2
    m = flow.nn.Softmax(dim=axis)
    arr = np.random.randn(2, 5, 6, 3)
    x = flow.Tensor(arr, device=flow.device(device))
    y = m(x)
    output = numpy_softmax(arr, axis)
    test_case.assertTrue(np.allclose(y.numpy(), output, rtol=1e-05))


def _test_softmax_dim_3(test_case, device):
    axis = 3
    m = flow.nn.Softmax(dim=axis)
    arr = np.random.randn(1, 3, 4, 7)
    x = flow.Tensor(arr, device=flow.device(device))
    y = m(x)
    output = numpy_softmax(arr, axis)
    test_case.assertTrue(np.allclose(y.numpy(), output, rtol=1e-05))

    axis2 = -1
    m2 = flow.nn.Softmax(dim=axis)
    y2 = m(x)
    output2 = numpy_softmax(arr, axis)
    test_case.assertTrue(np.allclose(y2.numpy(), output2, rtol=1e-05))


softmax_input_arr = np.array(
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


def _test_softmax_backward(test_case, device):
    x_grad = np.array(
        [
            [
                [
                    [
                        0.00000000e00,
                        0.00000000e00,
                        -2.21495572e-16,
                        9.77881196e-17,
                        -1.05306593e-17,
                    ],
                    [
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        1.32341829e-17,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        -2.21495572e-16,
                        -1.05306593e-17,
                        9.77881196e-17,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        -1.05306593e-17,
                        -2.11513946e-16,
                        -2.11513946e-16,
                        0.00000000e00,
                    ],
                ],
                [
                    [
                        -5.49032632e-19,
                        0.00000000e00,
                        1.32341829e-17,
                        9.77881196e-17,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        -2.11513946e-16,
                        -1.05306593e-17,
                        0.00000000e00,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        -1.05306593e-17,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        0.00000000e00,
                        -1.05306593e-17,
                        0.00000000e00,
                        -1.48611144e-18,
                    ],
                ],
                [
                    [
                        9.77881196e-17,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        1.32341829e-17,
                    ],
                    [
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        -2.20558493e-16,
                    ],
                    [
                        0.00000000e00,
                        -2.21495572e-16,
                        0.00000000e00,
                        0.00000000e00,
                        -1.05306593e-17,
                    ],
                    [
                        0.00000000e00,
                        1.32341829e-17,
                        -5.49032632e-19,
                        -1.05306593e-17,
                        0.00000000e00,
                    ],
                ],
            ],
            [
                [
                    [
                        0.00000000e00,
                        0.00000000e00,
                        -5.49032632e-19,
                        1.32341829e-17,
                        -2.11513946e-16,
                    ],
                    [
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        9.77881196e-17,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        -5.49032632e-19,
                        -2.11513946e-16,
                        1.32341829e-17,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        -2.11513946e-16,
                        -1.05306593e-17,
                        -1.05306593e-17,
                        0.00000000e00,
                    ],
                ],
                [
                    [
                        -2.21495572e-16,
                        0.00000000e00,
                        9.77881196e-17,
                        1.32341829e-17,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        -1.05306593e-17,
                        -2.11513946e-16,
                        0.00000000e00,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        -2.11513946e-16,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        0.00000000e00,
                        -2.11513946e-16,
                        0.00000000e00,
                        -2.20558493e-16,
                    ],
                ],
                [
                    [
                        1.32341829e-17,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        9.77881196e-17,
                    ],
                    [
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        -1.48611144e-18,
                    ],
                    [
                        0.00000000e00,
                        -5.49032632e-19,
                        0.00000000e00,
                        0.00000000e00,
                        -2.11513946e-16,
                    ],
                    [
                        0.00000000e00,
                        9.77881196e-17,
                        -2.21495572e-16,
                        -2.11513946e-16,
                        0.00000000e00,
                    ],
                ],
            ],
        ]
    )

    axis = 0
    m = flow.nn.Softmax(dim=axis)
    x = flow.Tensor(
        input_arr, requires_grad=True, device=flow.device(device), dtype=flow.float64
    )
    y = m(x).sum()
    y.backward()
    test_case.assertTrue(np.allclose(x.grad.numpy(), x_grad, 1e-5, 1e-5))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestHardsigmoidModule(flow.unittest.TestCase):
    def test_hardsigmoid(test_case):
        m = flow.nn.Hardsigmoid()
        arr = np.random.randn(2, 3, 4, 5)
        np_out = np.maximum(0, np.minimum(1, (arr + 3) / 6))
        x = flow.Tensor(arr)
        of_out = m(x)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, rtol=1e-05))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestSoftmax(flow.unittest.TestCase):
    def test_softmax(test_case):
        arg_dict = OrderedDict()
        arg_dict["fun"] = [
            _test_softmax,
            _test_softmax_dim_1,
            _test_softmax_dim_2,
            _test_softmax_dim_3,
            _test_softmax_backward,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


def _test_logsoftmax(test_case, device):
    dim = 1
    m = flow.nn.LogSoftmax(dim)
    input_arr = np.random.randn(4, 7)
    x = flow.Tensor(input_arr, device=flow.device(device))
    y = m(x)
    output = numpy_logsoftmax(input_arr, dim)
    test_case.assertTrue(np.allclose(y.numpy(), output, rtol=1e-05))


def _test_logsoftmax_dim_2(test_case, device):
    dim = 2
    m = flow.nn.LogSoftmax(dim)
    input_arr = np.random.randn(3, 4, 5)
    x = flow.Tensor(input_arr, device=flow.device(device))
    y = m(x)
    output = numpy_logsoftmax(input_arr, dim)
    test_case.assertTrue(np.allclose(y.numpy(), output, rtol=1e-05))


def _test_logsoftmax_dim_3(test_case, device):
    dim = 3
    m = flow.nn.LogSoftmax(dim)
    input_arr = np.random.randn(8, 9, 7, 3)
    x = flow.Tensor(input_arr, device=flow.device(device))
    y = m(x)
    output = numpy_logsoftmax(input_arr, dim)
    test_case.assertTrue(np.allclose(y.numpy(), output, rtol=1e-05))


def _test_logsoftmax_backward(test_case, device):
    axis = 0
    m = flow.nn.LogSoftmax(axis)
    x = flow.Tensor(
        softmax_input_arr,
        requires_grad=True,
        device=flow.device(device),
        dtype=flow.float64,
    )
    x_grad = np.array(
        [
            [
                [
                    [0.46211716, 0.96402758, -0.99505475, -0.76159416, 0.90514825],
                    [0.96402758, -0.96402758, -0.46211716, 0.76159416, 0.46211716],
                    [0.46211716, -0.99505475, 0.90514825, -0.76159416, 0.99932930],
                    [0.00000000, 0.90514825, -0.90514825, -0.90514825, -0.96402758],
                ],
                [
                    [0.99505475, 0.96402758, 0.76159416, -0.76159416, 0.46211716],
                    [0.00000000, -0.90514825, 0.90514825, -0.46211716, -0.46211716],
                    [0.46211716, -0.96402758, 0.00000000, 0.90514825, 0.46211716],
                    [0.96402758, 0.46211716, 0.90514825, 0.99817790, 0.98661430],
                ],
                [
                    [-0.76159416, 0.96402758, -0.46211716, 0.00000000, 0.76159416],
                    [-0.96402758, -0.96402758, 0.46211716, 0.46211716, -0.98661430],
                    [0.46211716, -0.99505475, 0.96402758, 0.46211716, 0.90514825],
                    [-0.96402758, 0.76159416, 0.99505475, 0.90514825, -0.96402758],
                ],
            ],
            [
                [
                    [-0.46211716, -0.96402758, 0.99505475, 0.76159416, -0.90514825],
                    [-0.96402758, 0.96402758, 0.46211716, -0.76159416, -0.46211716],
                    [-0.46211716, 0.99505475, -0.90514825, 0.76159416, -0.99932930],
                    [0.00000000, -0.90514825, 0.90514825, 0.90514825, 0.96402758],
                ],
                [
                    [-0.99505475, -0.96402758, -0.76159416, 0.76159416, -0.46211716],
                    [0.00000000, 0.90514825, -0.90514825, 0.46211716, 0.46211716],
                    [-0.46211716, 0.96402758, 0.00000000, -0.90514825, -0.46211716],
                    [-0.96402758, -0.46211716, -0.90514825, -0.99817790, -0.98661430],
                ],
                [
                    [0.76159416, -0.96402758, 0.46211716, 0.00000000, -0.76159416],
                    [0.96402758, 0.96402758, -0.46211716, -0.46211716, 0.98661430],
                    [-0.46211716, 0.99505475, -0.96402758, -0.46211716, -0.90514825],
                    [0.96402758, -0.76159416, -0.99505475, -0.90514825, 0.96402758],
                ],
            ],
        ]
    )
    y = m(x)
    z = y.sum()
    z.backward()
    test_case.assertTrue(np.allclose(x.grad.numpy(), x_grad, 1e-5, 1e-5))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
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


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestLogSigmoidModule(flow.unittest.TestCase):
    def test_logsigmoid(test_case):
        m = flow.nn.LogSigmoid()
        arr = np.random.randn(2, 3, 4, 5)
        np_out = np.log(1.0 / (1.0 + np.exp(-arr)))
        x = flow.Tensor(arr)
        of_out = m(x)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, rtol=1e-05))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestSoftplusModule(flow.unittest.TestCase):
    def test_softplus(test_case):
        m = flow.nn.Softplus()
        arr = np.random.randn(2, 3, 4, 5)
        np_out = np.where(arr > 20, arr, np.log(1.0 + np.exp(1.0 * arr)))
        x = flow.Tensor(arr)
        of_out = m(x)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, rtol=1e-05))

    def test_softplus_beta(test_case):
        m = flow.nn.Softplus(beta=1.11)
        arr = np.random.randn(2, 3, 4, 5)
        np_out = np.where(
            arr * 1.11 > 20, arr, 1.0 / 1.11 * np.log(1.0 + np.exp(1.11 * arr))
        )
        x = flow.Tensor(arr)
        of_out = m(x)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, rtol=1e-05))

    def test_softplus_threshold(test_case):
        m = flow.nn.Softplus(beta=1.11, threshold=1.55)
        arr = np.random.randn(2, 3, 4, 5)
        np_out = np.where(
            arr * 1.11 > 1.55, arr, 1.0 / 1.11 * np.log(1.0 + np.exp(1.11 * arr))
        )
        x = flow.Tensor(arr)
        of_out = m(x)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, rtol=1e-05))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestHardswishModule(flow.unittest.TestCase):
    def test_hardswish(test_case):
        m = flow.nn.Hardswish()
        arr = np.random.randn(2, 3, 4, 5)
        f = arr + 3
        relu6 = np.where(np.where(f < 0, 0, f) > 6, 6, np.where(f < 0, 0, f))
        np_out = arr * relu6 / 6
        x = flow.Tensor(arr)
        of_out = m(x)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, rtol=1e-05))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestHardtanhModule(flow.unittest.TestCase):
    def test_hardtanh(test_case):
        m = flow.nn.Hardtanh()
        arr = np.random.randn(2, 3, 4, 5)
        np_out = np.maximum(-1, np.minimum(1, arr))
        x = flow.Tensor(arr)
        of_out = m(x)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, rtol=1e-05))

    def test_hardtanh_min_max(test_case):
        m = flow.nn.Hardtanh(min_val=-2.0, max_val=2.3)
        arr = np.random.randn(2, 3, 4, 5)
        np_out = np.maximum(-2.0, np.minimum(2.3, arr))
        x = flow.Tensor(arr)
        of_out = m(x)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, rtol=1e-05))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestLeakyReLUModule(flow.unittest.TestCase):
    def test_leaky_relu(test_case):
        negative_slope = 0.2
        m = flow.nn.LeakyReLU(negative_slope=negative_slope)
        arr = np.random.randn(2, 3, 4, 5)

        np_out = np.maximum(0, arr) + negative_slope * np.minimum(0, arr)
        x = flow.Tensor(arr)
        of_out = m(x)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, rtol=1e-05))


if __name__ == "__main__":
    unittest.main()
