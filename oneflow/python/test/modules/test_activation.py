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


def numpy_softmax(x, axis):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def numpy_logsoftmax(x, dim):
    e_x = np.exp(x - np.max(x, axis=dim, keepdims=True))
    return np.log(e_x / e_x.sum(axis=dim, keepdims=True))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestSigmoidModule(flow.unittest.TestCase):
    def test_sigmoid(test_case):
        m = flow.nn.Sigmoid()
        input_arr = np.random.randn(2, 3, 4, 5)
        x = flow.Tensor(input_arr)

        y = m(x)
        y2 = flow.sigmoid(x)
        y3 = x.sigmoid()
        output = numpy_sigmoid(input_arr)

        test_case.assertTrue(np.allclose(y.numpy(), output, rtol=1e-05))
        test_case.assertTrue(np.allclose(y2.numpy(), output, rtol=1e-05))
        test_case.assertTrue(np.allclose(y3.numpy(), output, rtol=1e-05))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestSoftmaxModule(flow.unittest.TestCase):
    def test_softmax(test_case):
        axis = 0
        m = flow.nn.Softmax(dim=axis)
        arr = np.random.randn(2, 3, 4, 5)
        x = flow.Tensor(arr)
        y = m(x)
        output = numpy_softmax(arr, axis)
        test_case.assertTrue(np.allclose(y.numpy(), output, rtol=1e-05))

    def test_softmax_dim_1(test_case):
        axis = 1
        m = flow.nn.Softmax(dim=axis)
        arr = np.random.randn(9, 7, 8, 16)
        x = flow.Tensor(arr)
        y = m(x)
        output = numpy_softmax(arr, axis)
        test_case.assertTrue(np.allclose(y.numpy(), output, rtol=1e-05))

    def test_softmax_dim_2(test_case):
        axis = 2
        m = flow.nn.Softmax(dim=axis)
        arr = np.random.randn(2, 5, 6, 3)
        x = flow.Tensor(arr)
        y = m(x)
        output = numpy_softmax(arr, axis)
        test_case.assertTrue(np.allclose(y.numpy(), output, rtol=1e-05))

    def test_softmax_dim_3(test_case):
        axis = 3
        m = flow.nn.Softmax(dim=axis)
        arr = np.random.randn(1, 3, 4, 7)
        x = flow.Tensor(arr)
        y = m(x)
        output = numpy_softmax(arr, axis)
        test_case.assertTrue(np.allclose(y.numpy(), output, rtol=1e-05))

        axis2 = -1
        m2 = flow.nn.Softmax(dim=axis)
        y2 = m(x)
        output2 = numpy_softmax(arr, axis)
        test_case.assertTrue(np.allclose(y2.numpy(), output2, rtol=1e-05))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestLogSoftmaxModule(flow.unittest.TestCase):
    def test_logsoftmax(test_case):
        dim = 1
        m = flow.nn.LogSoftmax(dim)
        input_arr = np.random.randn(4, 7)
        x = flow.Tensor(input_arr)
        y = m(x)
        output = numpy_logsoftmax(input_arr, dim)
        test_case.assertTrue(np.allclose(y.numpy(), output, rtol=1e-05))

    def test_logsoftmax_dim_2(test_case):
        dim = 2
        m = flow.nn.LogSoftmax(dim)
        input_arr = np.random.randn(3, 4, 5)
        x = flow.Tensor(input_arr)
        y = m(x)
        output = numpy_logsoftmax(input_arr, dim)
        test_case.assertTrue(np.allclose(y.numpy(), output, rtol=1e-05))

    def test_logsoftmax_dim_3(test_case):
        dim = 3
        m = flow.nn.LogSoftmax(dim)
        input_arr = np.random.randn(8, 9, 7, 3)
        x = flow.Tensor(input_arr)
        y = m(x)
        output = numpy_logsoftmax(input_arr, dim)
        test_case.assertTrue(np.allclose(y.numpy(), output, rtol=1e-05))


if __name__ == "__main__":
    unittest.main()
