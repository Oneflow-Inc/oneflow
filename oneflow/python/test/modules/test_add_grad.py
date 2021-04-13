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
import oneflow as flow
from oneflow.python.nn.parameter import Parameter
import unittest
import numpy as np


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in eager mode",
)
class TestModule(flow.unittest.TestCase):
    def test_add_case1(test_case):
        add = flow.Add()
        x = Parameter(flow.Tensor(np.random.randn(2, 3)))
        y = Parameter(flow.Tensor(np.random.randn(2, 3)))
        of_out = add(x, y)
        grad = flow.Tensor(np.ones((2, 3), dtype=np.float32))
        of_out.backward(grad)
        test_case.assertTrue(np.allclose(x.grad.numpy(), grad.numpy(), 1e-4, 1e-4))
        test_case.assertTrue(np.allclose(y.grad.numpy(), grad.numpy(), 1e-4, 1e-4))

    def test_add_case2(test_case):
        add = flow.Add()
        x = Parameter(flow.Tensor(np.random.randn(2, 3)))
        y = 1
        of_out = add(x, y)
        grad = flow.Tensor(np.ones((2, 3), dtype=np.float32))
        of_out.backward(grad)
        test_case.assertTrue(np.allclose(x.grad.numpy(), grad.numpy(), 1e-4, 1e-4))

    def test_add_case3(test_case):
        add = flow.Add()
        x = 1
        y = Parameter(flow.Tensor(np.random.randn(2, 3)))
        of_out = add(x, y)
        grad = flow.Tensor(np.ones((2, 3), dtype=np.float32))
        of_out.backward(grad)
        test_case.assertTrue(np.allclose(y.grad.numpy(), grad.numpy(), 1e-4, 1e-4))

    def test_add_case4(test_case):
        # test __add__
        x = Parameter(flow.Tensor(np.random.randn(2, 3)))
        y = x + 1
        grad = flow.Tensor(np.ones((2, 3), dtype=np.float32))
        y.backward(grad)
        test_case.assertTrue(np.allclose(x.grad.numpy(), grad.numpy(), 1e-4, 1e-4))


if __name__ == "__main__":
    unittest.main()
