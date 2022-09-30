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

from random import shuffle
import numpy as np

from oneflow.test_utils.automated_test_util import *
from oneflow.test_utils.test_util import GenArgList
import oneflow.unittest
import oneflow as flow


@flow.unittest.skip_unless_1n1d()
class TestContiguous(flow.unittest.TestCase):
    @autotest(n=5)
    def test_transpose_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=4).to(device)
        y = torch.transpose(x, dim0=random(1, 3).to(int), dim1=random(1, 3).to(int))
        z = y.contiguous()
        return z

    @autotest(n=5, auto_backward=False)
    def test_transpose_with_bool_data(test_case):
        device = random_device()
        x = random_tensor(ndim=4, requires_grad=False).to(device).to(torch.bool)
        y = torch.transpose(x, dim0=random(1, 3).to(int), dim1=random(1, 3).to(int))
        z = y.contiguous()
        return z

    @autotest(n=5, auto_backward=False)
    def test_transpose_with_int_data(test_case):
        device = random_device()
        x = random_tensor(ndim=4, requires_grad=False).to(device).to(torch.int)
        y = torch.transpose(x, dim0=random(1, 3).to(int), dim1=random(1, 3).to(int))
        z = y.contiguous()
        return z

    @autotest(n=5, auto_backward=False)
    def test_contiguous_with_half_data(test_case):
        device = random_device()
        x = random_tensor(ndim=4, requires_grad=False).to(device).to(torch.float16)
        y = torch.transpose(x, dim0=random(1, 3).to(int), dim1=random(1, 3).to(int))
        z = y.contiguous()
        return z

    @autotest(n=10, check_graph=True)
    def test_permute2d_tensor_with_random_data(test_case):
        device = random_device()
        ndim = 2
        permute_list = [0, 1]
        shuffle(permute_list)
        x = random_tensor(
            ndim=ndim, dim0=random(1, 32).to(int), dim1=random(1, 59).to(int),
        ).to(device)
        y = x.permute(permute_list)
        z = y.contiguous()
        return z

    @autotest(n=10, check_graph=True)
    def test_permute3d_tensor_with_random_data(test_case):
        device = random_device()
        ndim = 3
        permute_list = [0, 1, 2]
        shuffle(permute_list)
        x = random_tensor(
            ndim=ndim,
            dim0=random(1, 7).to(int),
            dim1=random(1, 15).to(int),
            dim2=random(1, 9).to(int),
        ).to(device)
        y = x.permute(permute_list)
        z = y.contiguous()
        return z

    @autotest(n=10, check_graph=True)
    def test_permute4d_tensor_with_random_data(test_case):
        device = random_device()
        ndim = 4
        permute_list = [0, 1, 2, 3]
        shuffle(permute_list)
        x = random_tensor(
            ndim=ndim,
            dim0=random(1, 7).to(int),
            dim1=random(1, 15).to(int),
            dim2=random(1, 9).to(int),
            dim3=random(1, 19).to(int),
        ).to(device)
        y = x.permute(permute_list)
        z = y.contiguous()
        return z

    @profile(torch.Tensor.contiguous)
    def profile_contiguous(test_case):
        x = torch.ones(32, 3, 128, 128)
        x.contiguous()


def _test_inplace_contiguous(test_case, device):
    arr = np.random.randn(4, 5, 6, 7).astype(np.float32)
    input = flow.tensor(arr, device=device)
    x = input.permute(0, 3, 2, 1)  # x is non-contiguous tensor
    test_case.assertTrue(x.is_contiguous() == False)
    # y1 is normal version of tensor contiguous
    y1 = x.contiguous()
    # y2 is inplace version of tensor contiguous
    y2 = x.contiguous_()
    test_case.assertTrue(np.array_equal(y1.cpu().numpy(), y2.cpu().numpy()))
    test_case.assertTrue(id(x) != id(y1))
    test_case.assertTrue(id(x) == id(y2))
    test_case.assertTrue(x.is_contiguous() == True)
    test_case.assertTrue(y1.is_contiguous() == True)
    test_case.assertTrue(y2.is_contiguous() == True)


@flow.unittest.skip_unless_1n1d()
class TestInplaceContiguous(flow.unittest.TestCase):
    def test_inplace_contiguous(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_inplace_contiguous,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
