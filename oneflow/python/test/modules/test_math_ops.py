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
import oneflow.experimental as flow
import unittest
import numpy as np
from collections import OrderedDict

from test_util import GenArgList


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestVariance(flow.unittest.TestCase):
    def test_variance(test_case):
        input_arr = np.random.randn(2, 3, 4, 5)
        of_out = flow.Tensor(input_arr).var(1, True)
        np_out = np.var(input_arr, 1, keepdims=True)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))

    def test_variance_v2(test_case):
        input_arr = np.random.randn(4, 1, 3, 2)
        of_out = flow.var(flow.Tensor(input_arr), 2, False)
        np_out = np.var(input_arr, 2, keepdims=False)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestSin(flow.unittest.TestCase):
    def test_sin(test_case):
        input = flow.Tensor(np.random.randn(2, 6, 5, 3))
        of_out = flow.sin(input)
        np_out = np.sin(input.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))

    def test_sin_tensor_function(test_case):
        input = flow.Tensor(np.random.randn(8, 11, 9, 7))
        of_out = input.sin()
        np_out = np.sin(input.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestCos(flow.unittest.TestCase):
    def test_cos(test_case):
        input = flow.Tensor(np.random.randn(1, 3, 6), dtype=flow.float32)
        of_out = flow.cos(input)
        np_out = np.cos(input.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))
        test_case.assertTrue(np.allclose(input.cos().numpy(), np_out, 1e-5, 1e-5))

    def test_cos_tensor_function(test_case):
        arr = np.random.randn(4, 5, 6, 7)
        input = flow.Tensor(arr, dtype=flow.float32)
        np_out = np.cos(arr)
        of_out = input.cos()
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestLog(flow.unittest.TestCase):
    def test_log(test_case):
        input = flow.Tensor(np.random.randn(2, 3, 4, 5), dtype=flow.float32)
        of_out = flow.log(input)
        np_out = np.log(input.numpy())
        test_case.assertTrue(
            np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5, equal_nan=True)
        )
        test_case.assertTrue(np.allclose(input.log().numpy(), np_out, equal_nan=True))

    def test_log_nan_value(test_case):
        arr = np.array([-0.7168, -0.5471, -0.8933, -1.4428, -0.1190])
        input = flow.Tensor(arr, dtype=flow.float32)
        np_out = np.full((5,), np.nan)
        of_out = flow.log(input)
        test_case.assertTrue(
            np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5, equal_nan=True)
        )


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestStd(flow.unittest.TestCase):
    def test_std(test_case):
        np_arr = np.random.randn(2, 3, 4, 5)
        input = flow.Tensor(np_arr)
        of_out = flow.std(input, dim=2)
        np_out = np.std(np_arr, axis=2)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5,))

    def test_std_tensor_function(test_case):
        np_arr = np.random.randn(9, 8, 7, 6)
        input = flow.Tensor(np_arr)
        of_out = input.std(dim=1, keepdim=False)
        np_out = np.std(np_arr, axis=1)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))

    def test_std_negative_dim(test_case):
        np_arr = np.random.randn(4, 2, 3, 5)
        input = flow.Tensor(np_arr)
        of_out = input.std(dim=(-2, -1, -3), keepdim=False)
        np_out = np.std(np_arr, axis=(-2, -1, -3))
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestSqrt(flow.unittest.TestCase):
    def test_sqrt(test_case):
        input_arr = np.random.randn(3, 2, 5, 7)
        np_out = np.sqrt(input_arr)
        x = flow.Tensor(input_arr)
        of_out = flow.sqrt(input=x)
        test_case.assertTrue(
            np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5, equal_nan=True)
        )

    def test_sqrt_tensor_function(test_case):
        input_arr = np.random.randn(1, 6, 3, 8)
        np_out = np.sqrt(input_arr)
        x = flow.Tensor(input_arr)
        of_out = x.sqrt()
        test_case.assertTrue(
            np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5, equal_nan=True)
        )


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestRsqrt(flow.unittest.TestCase):
    def test_rsqrt(test_case):
        input_arr = np.random.randn(2, 3, 4, 5)
        np_out = 1 / np.sqrt(input_arr)
        x = flow.Tensor(input_arr)
        of_out = x.rsqrt()
        test_case.assertTrue(
            np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5, equal_nan=True)
        )

    def test_rsqrt_tensor_function(test_case):
        input_arr = np.random.randn(3, 2, 5, 7)
        np_out = 1 / np.sqrt(input_arr)
        x = flow.Tensor(input_arr)
        of_out = flow.rsqrt(input=x)
        test_case.assertTrue(
            np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5, equal_nan=True)
        )


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestSquare(flow.unittest.TestCase):
    def test_square(test_case):
        input_arr = np.random.randn(9, 4, 5, 6)
        np_out = np.square(input_arr)
        x = flow.Tensor(input_arr)
        of_out = flow.square(x)
        test_case.assertTrue(
            np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5, equal_nan=True)
        )

    def test_square_tensor_function(test_case):
        input_arr = np.random.randn(2, 7, 7, 3)
        np_out = np.square(input_arr)
        x = flow.Tensor(input_arr)
        of_out = x.square()
        test_case.assertTrue(
            np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5, equal_nan=True)
        )


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestPow(flow.unittest.TestCase):
    def test_pow(test_case):
        input = flow.Tensor(np.array([1, 2, 3, 4, 5, 6]), dtype=flow.float32)
        of_out = flow.pow(input, 2.1)
        np_out = np.power(input.numpy(), 2.1)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))

    def test_pow_tensor_function(test_case):
        input = flow.Tensor(np.array([1, 2, 3, 4, 5, 6]), dtype=flow.float32)
        of_out = input.pow(2.1)
        np_out = np.power(input.numpy(), 2.1)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))


def _test_abs_forward(test_case, device):
    input = flow.Tensor(np.random.randn(2, 3).astype(np.float32))
    of_out = flow.abs(input)
    np_out = np.abs(input.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))
    test_case.assertTrue(np.allclose(input.abs().numpy(), np_out, 1e-5, 1e-5))


def _test_abs_tensor_function_forward(test_case, device):
    x = np.random.randn(2, 3).astype(np.float32)
    input = flow.Tensor(x, dtype=flow.float32)
    np_out = np.abs(x)
    of_out = input.abs()
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))


def _test_abs_backward(test_case, device):
    np_input = np.random.randn(2, 3).astype(np.float32)
    input = flow.Tensor(np_input, dtype=flow.float32, requires_grad=True)
    of_out = flow.abs(input).sum()
    of_out.backward()
    np_grad = np.where(np_input > 0, 1, -1)
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-5, 1e-5))


def _test_abs_tensor_function_backward(test_case, device):
    np_input = np.random.randn(2, 3).astype(np.float32)
    input = flow.Tensor(np_input, dtype=flow.float32, requires_grad=True)
    of_out = input.abs().sum()
    of_out.backward()
    np_grad = np.where(np_input > 0, 1, -1)
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-5, 1e-5))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestAbs(flow.unittest.TestCase):
    def test_cosh(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_abs_forward,
            _test_abs_tensor_function_forward,
            _test_abs_backward,
            _test_abs_tensor_function_backward,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
