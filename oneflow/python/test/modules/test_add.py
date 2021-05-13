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


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestAddModule(flow.unittest.TestCase):
    def test_add(test_case):
        x = flow.Tensor(np.random.randn(2, 3))
        y = flow.Tensor(np.random.randn(2, 3))
        of_out = flow.add(x, y)
        np_out = np.add(x.numpy(), y.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

        x = 5
        y = flow.Tensor(np.random.randn(2, 3))
        of_out = flow.add(x, y)
        np_out = np.add(x, y.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

        x = flow.Tensor(np.random.randn(2, 3))
        y = 5
        of_out = flow.add(x, y)
        np_out = np.add(x.numpy(), y)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

        x = flow.Tensor(np.random.randn(2, 3))
        y = flow.Tensor(np.array([5.0]))
        of_out = flow.add(x, y)
        np_out = np.add(x.numpy(), y.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

        x = flow.Tensor(np.random.randn(1, 1))
        y = flow.Tensor(np.random.randn(2, 3))
        of_out = flow.add(x, y)
        np_out = np.add(x.numpy(), y.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

    def test_add_cuda(test_case):
        x = flow.Tensor(np.random.randn(2, 3))
        y = flow.Tensor(np.random.randn(2, 3))
        x.to("cuda")
        y.to("cuda")
        of_out = flow.add(x, y)
        np_out = np.add(x.numpy(), y.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

        x = 5
        y = flow.Tensor(np.random.randn(2, 3))
        y.to("cuda")
        of_out = flow.add(x, y)
        np_out = np.add(x, y.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

        x = flow.Tensor(np.random.randn(2, 3))
        y = 5
        x.to("cuda")
        of_out = flow.add(x, y)
        np_out = np.add(x.numpy(), y)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

        x = flow.Tensor(np.random.randn(2, 3))
        y = flow.Tensor(np.array([5.0]))
        x.to("cuda")
        y.to("cuda")
        of_out = flow.add(x, y)
        np_out = np.add(x.numpy(), y.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

        x = flow.Tensor(np.random.randn(1, 1))
        y = flow.Tensor(np.random.randn(2, 3))
        x.to("cuda")
        y.to("cuda")
        of_out = flow.add(x, y)
        np_out = np.add(x.numpy(), y.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))


import time


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestAddCudaBigTensorModule(flow.unittest.TestCase):
    def test_add_cuda(test_case):
        x = flow.Tensor(np.random.randn(10000, 10000))
        y = flow.Tensor(np.random.randn(10000, 10000))
        x.to("cuda")
        y.to("cuda")
        of_out = flow.add(x, y)
        np_out = np.add(x.numpy(), y.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestAddCpuBigTensorModule(flow.unittest.TestCase):
    def test_add_cuda(test_case):
        x = flow.Tensor(np.random.randn(10000, 10000))
        y = flow.Tensor(np.random.randn(10000, 10000))
        of_out = flow.add(x, y)
        np_out = np.add(x.numpy(), y.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))


if __name__ == "__main__":
    unittest.main()
