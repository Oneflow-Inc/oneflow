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
from scipy import special

import oneflow.experimental as flow
from test_util import GenArgList


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
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))


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
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))


def _test_tanh_nn_impl(test_case, shape, device):
    np_input = np.random.randn(*shape)
    of_input = flow.Tensor(
        np_input, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )

    tanh = flow.nn.Tanh()
    of_out = tanh(of_input)
    np_out = np.tanh(np_input)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

    of_out = of_out.sum()
    of_out.backward()
    test_case.assertTrue(
        np.allclose(of_input.grad.numpy(), 1.0 - np_out * np_out, 1e-4, 1e-4)
    )


def _test_tanh_function_impl(test_case, shape, device):
    np_input = np.random.randn(*shape)
    of_input = flow.Tensor(
        np_input, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )

    of_out = flow.tanh(of_input)
    np_out = np.tanh(np_input)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

    of_out = of_out.sum()
    of_out.backward()
    test_case.assertTrue(
        np.allclose(of_input.grad.numpy(), 1.0 - np_out * np_out, 1e-4, 1e-4)
    )


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestTanh(flow.unittest.TestCase):
    def test_tanh(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 3), (2, 4, 5, 6)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_tanh_nn_impl(test_case, *arg)
            _test_tanh_function_impl(test_case, *arg)


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


def _np_gelu(x):
    return 0.5 * x * (1 + special.erf(x / np.sqrt(2)))


def _test_gelu_impl(test_case, device):
    np_input = np.array([1.0, -1.0, 2.3]).astype(np.float32)
    of_input = flow.Tensor(
        np_input, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )

    gelu = flow.nn.GELU()
    of_out = gelu(of_input)
    np_out = _np_gelu(np_input)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

    of_out = of_out.sum()
    of_out.backward()
    np_grad = [1.0833154916763306, -0.08331547677516937, 1.0544281005859375]
    test_case.assertTrue(np.allclose(of_input.grad.numpy(), np_grad, 1e-4, 1e-4))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestGelu(flow.unittest.TestCase):
    def test_gelu(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_gelu_impl(test_case, *arg)


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

    test_case.assertTrue(np.allclose(y.numpy(), output, 1e-4, 1e-4))
    test_case.assertTrue(np.allclose(y2.numpy(), output, 1e-4, 1e-4))
    test_case.assertTrue(np.allclose(y3.numpy(), output, 1e-4, 1e-4))


def _test_sigmoid_backward(test_case, device):
    input_arr = np.random.randn(2, 3, 4, 5)
    x = flow.Tensor(input_arr, device=flow.device(device), requires_grad=True)
    x_grad = numpy_sigmoid_grad(input_arr, np.ones(input_arr.shape))
    m = flow.nn.Sigmoid()
    y = m(x).sum()
    y.backward()
    test_case.assertTrue(np.allclose(x.grad.numpy(), x_grad, 1e-4, 1e-4))


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
    test_case.assertTrue(np.allclose(y.numpy(), output, 1e-4, 1e-4))


def _test_softmax_dim_1(test_case, device):
    axis = 1
    m = flow.nn.Softmax(dim=axis)
    arr = np.random.randn(9, 7, 8, 16)
    x = flow.Tensor(arr, device=flow.device(device))
    y = m(x)
    output = numpy_softmax(arr, axis)
    test_case.assertTrue(np.allclose(y.numpy(), output, 1e-4, 1e-4))


def _test_softmax_dim_2(test_case, device):
    axis = 2
    m = flow.nn.Softmax(dim=axis)
    arr = np.random.randn(2, 5, 6, 3)
    x = flow.Tensor(arr, device=flow.device(device))
    y = m(x)
    output = numpy_softmax(arr, axis)
    test_case.assertTrue(np.allclose(y.numpy(), output, 1e-4, 1e-4))


def _test_softmax_dim_3(test_case, device):
    axis = 3
    m = flow.nn.Softmax(dim=axis)
    arr = np.random.randn(1, 3, 4, 7)
    x = flow.Tensor(arr, device=flow.device(device))
    y = m(x)
    output = numpy_softmax(arr, axis)
    test_case.assertTrue(np.allclose(y.numpy(), output, 1e-4, 1e-4))

    axis2 = -1
    m2 = flow.nn.Softmax(dim=axis)
    y2 = m(x)
    output2 = numpy_softmax(arr, axis)
    test_case.assertTrue(np.allclose(y2.numpy(), output2, 1e-4, 1e-4))


def _test_softmax_backward(test_case, device):
    # Grad of softmax should equal to zero.
    # See:https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
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
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))


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
    test_case.assertTrue(np.allclose(y.numpy(), output, 1e-4, 1e-4))


def _test_logsoftmax_dim_2(test_case, device):
    dim = 2
    m = flow.nn.LogSoftmax(dim)
    input_arr = np.random.randn(3, 4, 5)
    x = flow.Tensor(input_arr, device=flow.device(device))
    y = m(x)
    output = numpy_logsoftmax(input_arr, dim)
    test_case.assertTrue(np.allclose(y.numpy(), output, 1e-4, 1e-4))


def _test_logsoftmax_dim_3(test_case, device):
    dim = 3
    m = flow.nn.LogSoftmax(dim)
    input_arr = np.random.randn(8, 9, 7, 3)
    x = flow.Tensor(input_arr, device=flow.device(device))
    y = m(x)
    output = numpy_logsoftmax(input_arr, dim)
    test_case.assertTrue(np.allclose(y.numpy(), output, 1e-4, 1e-4))


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
        input_arr, requires_grad=True, device=flow.device(device), dtype=flow.float64,
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
    y = m(x).sum()
    y.backward()
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
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))


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
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

    def test_softplus_beta(test_case):
        m = flow.nn.Softplus(beta=1.11)
        arr = np.random.randn(2, 3, 4, 5)
        np_out = np.where(
            arr * 1.11 > 20, arr, 1.0 / 1.11 * np.log(1.0 + np.exp(1.11 * arr))
        )
        x = flow.Tensor(arr)
        of_out = m(x)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

    def test_softplus_threshold(test_case):
        m = flow.nn.Softplus(beta=1.11, threshold=1.55)
        arr = np.random.randn(2, 3, 4, 5)
        np_out = np.where(
            arr * 1.11 > 1.55, arr, 1.0 / 1.11 * np.log(1.0 + np.exp(1.11 * arr))
        )
        x = flow.Tensor(arr)
        of_out = m(x)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))


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
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

def _np_hardtanh_grad(x):
    return np.where(x <= -2.0, 0.0, np.where(x >= 2.3 , 0.0, 1.0))


def _test_hardtanh_impl(test_case, shape, device):
    m = flow.nn.Hardtanh()
    arr = np.random.randn(*shape)
    np_out = np.maximum(-1, np.minimum(1, arr))
    x = flow.Tensor(arr, device=flow.device(device))
    of_out = m(x)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

    m = flow.nn.Hardtanh(min_val=-2.0, max_val=2.3)
    arr = np.random.randn(*shape)
    np_out = np.maximum(-2.0, np.minimum(2.3, arr))
    x = flow.Tensor(arr, device=flow.device(device), requires_grad=True)
    of_out = m(x)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

    of_out = of_out.sum()
    of_out.backward()
    test_case.assertTrue(np.allclose(x.grad.numpy(), _np_hardtanh_grad(np_out), 1e-4, 1e-4))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestHardtanhModule(flow.unittest.TestCase):
    def test_hardtanh(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 3), (2, 4, 5, 6)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_hardtanh_impl(test_case, *arg)


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
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))


if __name__ == "__main__":
    unittest.main()
