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
import random as random_util
import torch as torch_original

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

    @autotest(n=10)
    def test_to_memory_format(test_case):
        def check_equal(flow_result, torch_result):
            test_case.assertEqual(list(flow_result.shape), list(torch_result.shape))
            test_case.assertEqual(
                list(flow_result.stride()), list(torch_result.stride())
            )
            test_case.assertEqual(
                flow_result.is_contiguous(), torch_result.is_contiguous()
            )
            test_case.assertEqual(
                flow_result.is_contiguous(memory_format=flow.channels_last),
                torch_result.is_contiguous(memory_format=torch.channels_last),
            )
            test_case.assertTrue(
                np.allclose(
                    flow_result.detach().cpu().numpy(),
                    torch_result.detach().cpu().numpy(),
                    1e-06,
                    1e-06,
                )
            )

        device = random_device()
        dtype = random_dtype(["pod", "half"])
        x = (
            random_tensor(
                ndim=4,
                dim0=random(1, 6).to(int),
                dim1=random(1, 6).to(int),
                dim2=random(1, 6).to(int),
                dim3=random(1, 6).to(int),
            )
            .to(device)
            .to(dtype)
        )

        oneflow_x = x.oneflow
        pytorch_x = x.pytorch

        oneflow_x = oneflow_x.to(memory_format=flow.contiguous_format)
        pytorch_x = pytorch_x.to(memory_format=torch_original.contiguous_format)
        check_equal(oneflow_x, pytorch_x)

        oneflow_x = oneflow_x.to(memory_format=flow.channels_last)
        pytorch_x = pytorch_x.to(memory_format=torch_original.channels_last)
        check_equal(oneflow_x, pytorch_x)

        oneflow_x = oneflow_x.to(memory_format=flow.contiguous_format)
        pytorch_x = pytorch_x.to(memory_format=torch_original.contiguous_format)
        check_equal(oneflow_x, pytorch_x)


if __name__ == "__main__":
    unittest.main()
