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
from collections import OrderedDict

import numpy as np
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


def _test_addmm(test_case, shape, alpha, beta, device):
    mat1 = np.random.randn(*shape)
    mat2 = np.random.randn(*shape)
    input = np.random.randn(*shape)
    mat1_tensor = flow.tensor(mat1, dtype=flow.float32, device=flow.device(device))
    mat2_tensor = flow.tensor(mat2, dtype=flow.float32, device=flow.device(device))
    input_tensor = flow.tensor(input, dtype=flow.float32, device=flow.device(device))
    of_out = flow.addmm(input_tensor, mat1_tensor, mat2_tensor, alpha, beta)
    np_out = np.add(beta * input, alpha * np.matmul(mat1, mat2))
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))


def _test_addmm_backward(test_case, shape, alpha, beta, device):
    mat1 = np.random.randn(*shape)
    mat2 = np.random.randn(*shape)
    input = np.random.randn(*shape)
    mat1_tensor = flow.tensor(mat1, dtype=flow.float32, device=flow.device(device))
    mat2_tensor = flow.tensor(mat2, dtype=flow.float32, device=flow.device(device))
    input_tensor = flow.tensor(
        input, dtype=flow.float32, requires_grad=True, device=flow.device(device)
    )
    of_out = flow.addmm(input_tensor, mat1_tensor, mat2_tensor, alpha, beta).sum()
    of_out.backward()
    np_grad_out = np.ones_like(input) * beta
    test_case.assertTrue(
        np.allclose(input_tensor.grad.numpy(), np_grad_out, 1e-05, 1e-05)
    )


@flow.unittest.skip_unless_1n1d()
class TestAddmm(flow.unittest.TestCase):
    def test_addmm(test_case):
        arg_dict = OrderedDict()
        arg_dict["function_test"] = [_test_addmm, _test_addmm_backward]
        arg_dict["shape"] = [(3, 3)]
        arg_dict["alpha"] = [4, 1.2, -3.7]
        arg_dict["beta"] = [1.5, 4, -2]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @autotest(n=5, rtol=1e-2, atol=1e-3)
    def test_addmm_flow_with_random_data(test_case):
        device = random_device()
        input = random_tensor(ndim=2, dim0=2, dim1=3).to(device)
        mat1 = random_tensor(ndim=2, dim0=2, dim1=4).to(device)
        mat2 = random_tensor(ndim=2, dim0=4, dim1=3).to(device)
        y = torch.addmm(
            input,
            mat1,
            mat2,
            beta=random().to(float) | nothing(),
            alpha=random().to(float) | nothing(),
        )
        return y

    @autotest(n=5, rtol=1e-2, atol=1e-3)
    def test_addmm_broadcast_flow_with_random_data(test_case):
        device = random_device()
        input = random_tensor(ndim=2, dim0=1, dim1=1).to(device)
        mat1 = random_tensor(ndim=2, dim0=2, dim1=4).to(device)
        mat2 = random_tensor(ndim=2, dim0=4, dim1=3).to(device)
        y = torch.addmm(
            input,
            mat1,
            mat2,
            beta=random().to(float) | nothing(),
            alpha=random().to(float) | nothing(),
        )
        return y

    @profile(torch.addmm)
    def profile_addmm(test_case):
        input = torch.ones(2, 3)
        mat1 = torch.ones(2, 3)
        mat2 = torch.ones(3, 3)
        torch.addmm(input, mat1, mat2)
        torch.addmm(input, mat1, mat2, alpha=1, beta=2)


if __name__ == "__main__":
    unittest.main()
