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
import oneflow as flow


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


if __name__ == "__main__":
    unittest.main()
