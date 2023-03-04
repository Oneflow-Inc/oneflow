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

    def test_tensor_new(test_case):
        dtype = random_dtype(["pod"])
        device = random_device()
        x = random_tensor(ndim=3).to(dtype).to(device)
        of_result = x.oneflow.new()
        th_result = x.pytorch.new()
        test_case.assertTrue(list(of_result.shape) == list(th_result.shape))
        test_case.assertTrue(
            of_result.numpy().dtype == th_result.detach().cpu().numpy().dtype
        )
        test_case.assertTrue(of_result.device.type == th_result.device.type)

        y = random_tensor(ndim=3).to(dtype).to(device)
        of_result = x.oneflow.new(y.oneflow)
        th_result = x.pytorch.new(y.pytorch)
        test_case.assertTrue(list(of_result.shape) == list(th_result.shape))
        test_case.assertTrue(
            of_result.numpy().dtype == th_result.detach().cpu().numpy().dtype
        )
        test_case.assertTrue(of_result.device.type == th_result.device.type)

        np_data = np.random.randn(3, 3)
        of_result = x.oneflow.new(np_data)
        th_result = x.pytorch.new(np_data)
        test_case.assertTrue(list(of_result.shape) == list(th_result.shape))
        test_case.assertTrue(
            of_result.numpy().dtype == th_result.detach().cpu().numpy().dtype
        )
        test_case.assertTrue(of_result.device.type == th_result.device.type)

        of_result = x.oneflow.new([1, 2, 3])
        th_result = x.pytorch.new([1, 2, 3])
        test_case.assertTrue(list(of_result.shape) == list(th_result.shape))
        test_case.assertTrue(
            of_result.numpy().dtype == th_result.detach().cpu().numpy().dtype
        )
        test_case.assertTrue(of_result.device.type == th_result.device.type)

    @autotest(n=3)
    def test_baddbmm(test_case):
        device = random_device()
        batch_dim = random().to(int)
        dim1 = random().to(int)
        dim2 = random().to(int)
        dim3 = random().to(int)
        x = random_tensor(
            ndim=3, dim0=oneof(batch_dim, 1).value(), dim1=dim1, dim2=dim3
        ).to(device)
        batch1 = random_tensor(ndim=3, dim0=batch_dim, dim1=dim1, dim2=dim2).to(device)
        batch2 = random_tensor(ndim=3, dim0=batch_dim, dim1=dim2, dim2=dim3).to(device)
        alpha = random_or_nothing(-1, 1).to(float)
        beta = random_or_nothing(-1, 1).to(float)
        return x.baddbmm(batch1, batch2, alpha=alpha, beta=beta)


if __name__ == "__main__":
    unittest.main()
