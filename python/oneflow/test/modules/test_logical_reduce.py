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

from oneflow.test_utils.automated_test_util import *
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
class TestLogicalReduce(flow.unittest.TestCase):
    @autotest(auto_backward=False)
    def test_sum_with_random_data(test_case):
        device = random_device()
        dim = random(1, 4).to(int)
        x = random_tensor(ndim=4, dtype=float, requires_grad=False).to(device)
        return torch.sum(x, dim)

    @autotest(auto_backward=False)
    def test_mean_with_random_data(test_case):
        device = random_device()
        dim = random(1, 4).to(int)
        x = random_tensor(ndim=4, dtype=float, requires_grad=False).to(device)
        return torch.mean(x, dim)

    @autotest(auto_backward=False)
    def test_all_with_random_data(test_case):
        device = random_device()
        dim = random(1, 4).to(int)
        x = random_tensor(ndim=4, dtype=float, requires_grad=False).to(device)
        return torch.all(x, dim)

    @autotest(auto_backward=False)
    def test_any_with_random_data(test_case):
        device = random_device()
        dim = random(1, 4).to(int)
        x = random_tensor(ndim=4, dtype=float, requires_grad=False).to(device)
        return torch.any(x, dim)

    @autotest(auto_backward=False)
    def test_prod_with_random_data(test_case):
        device = random_device()
        dim = random(1, 4).to(int)
        x = random_tensor(ndim=4, dtype=float, requires_grad=False).to(device)
        return torch.prod(x, dim)

    @autotest(auto_backward=False)
    def test_sum_keepdim_with_random_data(test_case):
        device = random_device()
        dim = random(1, 4).to(int)
        x = random_tensor(ndim=4, dtype=float, requires_grad=False).to(device)
        return torch.sum(x, dim, keepdim=True)

    @autotest(auto_backward=False)
    def test_mean_keepdim_with_random_data(test_case):
        device = random_device()
        dim = random(1, 4).to(int)
        x = random_tensor(ndim=4, dtype=float, requires_grad=False).to(device)
        return torch.mean(x, dim, keepdim=True)

    @autotest(auto_backward=False)
    def test_all_keepdim_with_random_data(test_case):
        device = random_device()
        dim = random(1, 4).to(int)
        x = random_tensor(ndim=4, dtype=float, requires_grad=False).to(device)
        return torch.all(x, dim, keepdim=True)

    @autotest(auto_backward=False)
    def test_any_keepdim_with_random_data(test_case):
        device = random_device()
        dim = random(1, 4).to(int)
        x = random_tensor(ndim=4, dtype=float, requires_grad=False).to(device)
        return torch.any(x, dim, keepdim=True)

    @autotest(auto_backward=False)
    def test_prod_keepdim_with_random_data(test_case):
        device = random_device()
        dim = random(1, 4).to(int)
        x = random_tensor(ndim=4, dtype=float, requires_grad=False).to(device)
        return torch.prod(x, dim, keepdim=True)

    @autotest(auto_backward=False)
    def test_scalar_reduce_sum_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=4, dtype=float, requires_grad=False).to(device)
        return torch.sum(x)

    @autotest(auto_backward=False)
    def test_scalar_reduce_mean_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=4, dtype=float, requires_grad=False).to(device)
        return torch.mean(x)

    @autotest(auto_backward=False)
    def test_scalar_reduce_all_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=4, dtype=float, requires_grad=False).to(device)
        return torch.all(x)

    @autotest(auto_backward=False)
    def test_scalar_reduce_any_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=4, dtype=float, requires_grad=False).to(device)
        return torch.any(x)

    @autotest(auto_backward=False)
    def test_scalar_reduce_prod_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=4, dtype=float, requires_grad=False).to(device)
        return torch.prod(x)

    @autotest(auto_backward=False)
    def test_all_bool_input_with_random_data(test_case):
        device = random_device()
        dim = random(1, 4).to(int)
        x = random_tensor(ndim=4, dtype=float, requires_grad=False).to(
            device, dtype=torch.bool
        )
        return torch.all(x, dim)

    @autotest(auto_backward=False)
    def test_any_bool_input_with_random_data(test_case):
        device = random_device()
        dim = random(1, 4).to(int)
        x = random_tensor(ndim=4, dtype=float, requires_grad=False).to(
            device, dtype=torch.bool
        )
        return torch.any(x, dim)


if __name__ == "__main__":
    unittest.main()
