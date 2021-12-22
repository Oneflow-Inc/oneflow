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
import random
import unittest
from collections import OrderedDict

import torch
import numpy as np
import oneflow as flow
import oneflow.unittest

def test_cumsum_forward(test_case, np_arr, device):
    for i in range(0, np_arr.ndim):
        torch_tensor = torch.tensor(np_arr, device=device)
        torch_rlt = torch.cumsum(torch_tensor, i)

        of_tensor = flow.tensor(np_arr, device=device)
        of_rlt = flow.cumsum(of_tensor, i)

        if device == "cpu":
            test_case.assertTrue(np.allclose(torch_rlt.numpy(), of_rlt.numpy()))
        elif device == "cuda":
            test_case.assertTrue(np.allclose(torch_rlt.cpu().numpy(), of_rlt.cpu().numpy()))

def test_cumsum_backward(test_case, np_arr, device):
    for i in range(0, np_arr.ndim):
        torch_tensor = torch.tensor(np_arr, device=device, requires_grad=True)
        torch_rlt = torch.cumsum(torch_tensor, i)
        torch_sum = torch_rlt.sum()
        torch_sum.backward()

        of_tensor = flow.tensor(np_arr, device=device, requires_grad=True)
        of_rlt = flow.cumsum(of_tensor, i)
        of_sum = of_rlt.sum()
        of_sum.backward()

        if device == "cpu":
            test_case.assertTrue(np.allclose(torch_tensor.grad.numpy(), of_tensor.grad.numpy()))
        elif device == "cuda":
            test_case.assertTrue(np.allclose(torch_tensor.grad.cpu().numpy(), of_tensor.grad.cpu().numpy()))

@flow.unittest.skip_unless_1n1d()
class TestCumsum(flow.unittest.TestCase):
    def test_cumsum_forward_cpu(test_case):
        for i in range(0, 20):
            np_arr = np.random.randn(3)
            test_cumsum_forward(test_case, np_arr, "cpu")
        for i in range(0, 20):
            np_arr = np.random.randn(3, 3)
            test_cumsum_forward(test_case, np_arr, "cpu")
        for i in range(0, 20):
            np_arr = np.random.randn(3, 3, 3)
            test_cumsum_forward(test_case, np_arr, "cpu")
        for i in range(0, 20):
            np_arr = np.random.randn(3, 3, 3, 3)
            test_cumsum_forward(test_case, np_arr, "cpu")
        for i in range(0, 20):
            np_arr = np.random.randn(3, 3, 3, 3, 3)
            test_cumsum_forward(test_case, np_arr, "cpu")
        for i in range(0, 20):
            np_arr = np.random.randn(3, 3, 3, 3, 3, 3)
            test_cumsum_forward(test_case, np_arr, "cpu")

    def test_cumsum_forward_gpu(test_case):
        for i in range(0, 20):
            np_arr = np.random.randn(3)
            test_cumsum_forward(test_case, np_arr, "cuda")
        for i in range(0, 20):
            np_arr = np.random.randn(3, 3)
            test_cumsum_forward(test_case, np_arr, "cuda")
        for i in range(0, 20):
            np_arr = np.random.randn(3, 3, 3)
            test_cumsum_forward(test_case, np_arr, "cuda")
        for i in range(0, 20):
            np_arr = np.random.randn(3, 3, 3, 3)
            test_cumsum_forward(test_case, np_arr, "cuda")
        for i in range(0, 20):
            np_arr = np.random.randn(3, 3, 3, 3, 3)
            test_cumsum_forward(test_case, np_arr, "cuda")
        for i in range(0, 20):
            np_arr = np.random.randn(3, 3, 3, 3, 3, 3)
            test_cumsum_forward(test_case, np_arr, "cuda")

    def test_cumsum_backward_cpu(test_case):
        for i in range(0, 20):
            np_arr = np.random.randn(3)
            test_cumsum_backward(test_case, np_arr, "cpu")
        for i in range(0, 20):
            np_arr = np.random.randn(3, 3)
            test_cumsum_backward(test_case, np_arr, "cpu")
        for i in range(0, 20):
            np_arr = np.random.randn(3, 3, 3)
            test_cumsum_backward(test_case, np_arr, "cpu")
        for i in range(0, 20):
            np_arr = np.random.randn(3, 3, 3, 3)
            test_cumsum_backward(test_case, np_arr, "cpu")
        for i in range(0, 20):
            np_arr = np.random.randn(3, 3, 3, 3, 3)
            test_cumsum_backward(test_case, np_arr, "cpu")
        for i in range(0, 20):
            np_arr = np.random.randn(3, 3, 3, 3, 3, 3)
            test_cumsum_backward(test_case, np_arr, "cpu")

    def test_cumsum_backward_gpu(test_case):
        for i in range(0, 20):
            np_arr = np.random.randn(3)
            test_cumsum_backward(test_case, np_arr, "cuda")
        for i in range(0, 20):
            np_arr = np.random.randn(3, 3)
            test_cumsum_backward(test_case, np_arr, "cuda")
        for i in range(0, 20):
            np_arr = np.random.randn(3, 3, 3)
            test_cumsum_backward(test_case, np_arr, "cuda")
        for i in range(0, 20):
            np_arr = np.random.randn(3, 3, 3, 3)
            test_cumsum_backward(test_case, np_arr, "cuda")
        for i in range(0, 20):
            np_arr = np.random.randn(3, 3, 3, 3, 3)
            test_cumsum_backward(test_case, np_arr, "cuda")
        for i in range(0, 20):
            np_arr = np.random.randn(3, 3, 3, 3, 3, 3)
            test_cumsum_backward(test_case, np_arr, "cuda")

if __name__ == "__main__":
    unittest.main()
