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

import oneflow as flow
import oneflow.unittest
import numpy as np

from oneflow.test_utils.automated_test_util import *


def _get_indexes(device):
    return (
        constant(
            torch.tensor(np.array([[0, 1], [1, 0]]), dtype=torch.int64, device=device)
        ),
        constant(
            torch.tensor(np.array([[1, 0], [0, 1]]), dtype=torch.int64, device=device)
        ),
        constant(
            torch.tensor(np.array([[1, 0], [1, 0]]), dtype=torch.int64, device=device)
        ),
        constant(
            torch.tensor(np.array([[0, 1], [0, 1]]), dtype=torch.int64, device=device)
        ),
    )


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestTensor(flow.unittest.TestCase):
    @autotest(n=10)
    def test_scatter_random_data(test_case):
        device = random_device()
        input = random_tensor(ndim=2, dim0=2, dim1=2).to(device)
        src = oneof(3.14, random_tensor(ndim=2, dim0=2, dim1=2).to(device))
        inplace = oneof(True, False)
        dim = oneof(0, 1, -1)
        if inplace:
            y = input + 1
            y.scatter_(dim, oneof(*_get_indexes(device)), src)
            return y
        return input.scatter(dim, oneof(*_get_indexes(device)), src)

    @autotest(
        n=10, auto_backward=False
    )  # peihong: pytorch dose not support backward when reduce is add or multiply
    def test_scatter_add_or_multiply_random_data(test_case):
        device = random_device()
        input = random_tensor(ndim=2, dim0=2, dim1=2).to(device)
        src = random_tensor(ndim=2, dim0=2, dim1=2).to(device)
        inplace = oneof(True, False)
        reduce = oneof("add", "multiply")
        dim = oneof(0, 1)
        if inplace:
            y = input + 1
            y.scatter_(
                dim, oneof(*_get_indexes(device)), src, reduce=reduce,
            )
            return y
        return input.scatter(dim, oneof(*_get_indexes(device)), src, reduce=reduce)

    def test_tensor_element_size_api(test_case):
        x = flow.ones(2, 1, dtype=flow.float)
        test_case.assertEqual(x.element_size(), 4)

    @autotest(n=1)
    def test_tensor_matmul(test_case):
        device = random_device()
        dim0 = random(low=2, high=10).to(int)
        dim1 = random(low=3, high=20).to(int)
        dim2 = random(low=2, high=11).to(int)
        a = random_tensor(ndim=2, dim0=dim0, dim1=dim1).to(device)
        b = random_tensor(ndim=2, dim0=dim1, dim1=dim2).to(device)
        return a.matmul(b)

    @autotest(n=1)
    def test_tensor_softplus(test_case):
        device = random_device()
        np_input = np.random.randn(2, 3)
        of_input = flow.tensor(
            np_input, dtype=flow.float32, device=flow.device(device), requires_grad=True
        )
        np_x_grad = np.exp(np_input) / (1 + np.exp(np_input))
        of_out = flow.softplus(of_input)
        np_out = np.log(1 + np.exp(np_input))
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))
        of_out = of_out.sum()
        of_out.backward()
        test_case.assertTrue(np.allclose(of_input.grad.numpy(), np_x_grad, 0.0001, 0.0001))




if __name__ == "__main__":
    unittest.main()
