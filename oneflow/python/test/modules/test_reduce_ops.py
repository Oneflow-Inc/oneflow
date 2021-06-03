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
class TestSumModule(flow.unittest.TestCase):
    def test_sum(test_case):

        input = flow.Tensor(
            np.random.randn(4, 5, 6), dtype=flow.float32, requires_grad=True
        )
        of_out = flow.sum(input, dim=(2, 1))
        np_out = np.sum(input.numpy(), axis=(2, 1))
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

        of_out = of_out.sum()
        of_out.backward()
        print(input.grad.numpy())


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestMinModule(flow.unittest.TestCase):
    def test_min(test_case):

        input = flow.Tensor(
            np.random.randn(4, 5, 6), dtype=flow.float32, requires_grad=True
        )
        of_out = flow.min(input, dim=(2, 1))
        np_out = np.min(input.numpy(), axis=(2, 1))
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

        of_out = of_out.sum()
        of_out.backward()
        print(input.grad.numpy())


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestMaxModule(flow.unittest.TestCase):
    def test_max(test_case):

        input = flow.Tensor(
            np.random.randn(4, 5, 6), dtype=flow.float32, requires_grad=True
        )
        of_out = flow.max(input, dim=(2, 1))
        np_out = np.max(input.numpy(), axis=(2, 1))
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

        of_out = of_out.sum()
        of_out.backward()
        print(input.grad.numpy())


if __name__ == "__main__":
    unittest.main()
