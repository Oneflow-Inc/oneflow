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
from test_util import GenArgList

import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
class TestLogicalReduce(flow.unittest.TestCase):
    @autotest(auto_backward=False)
    def test_all_with_random_data(test_case):
        device = random_device()
        dim = random(1, 4).to(int)
        x = random_tensor(ndim=4, dtype=float, requires_grad=False).to(device)
        return torch.all(x, dim)

    @autotest(auto_backward=False)
    def test_all_bool_input_with_random_data(test_case):
        device = random_device()
        dim = random(1, 4).to(int)
        x = random_tensor(ndim=4, dtype=float, requires_grad=False).to(
            device, dtype=torch.bool
        )
        return torch.all(x, dim)

    @autotest(auto_backward=False, check_graph=True)
    def test_any_with_random_data(test_case):
        device = random_device()
        dim = random(1, 4).to(int)
        x = random_tensor(ndim=4, dtype=float, requires_grad=False).to(device)
        return torch.any(x, dim)

    @autotest(auto_backward=False)
    def test_any_bool_input_with_random_data(test_case):
        device = random_device()
        dim = random(1, 4).to(int)
        x = random_tensor(ndim=4, dtype=float, requires_grad=False).to(
            device, dtype=torch.bool
        )
        return torch.any(x, dim)

    @autotest(auto_backward=False, check_graph=True)
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
    def test_matrix_row_all_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=2, dtype=float, requires_grad=False).to(device)
        return torch.all(x, 1)

    @autotest(auto_backward=False)
    def test_matrix_row_any_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=2, dtype=float, requires_grad=False).to(device)
        return torch.any(x, 1)

    @autotest(auto_backward=False)
    def test_matrix_col_all_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=2, dtype=float, requires_grad=False).to(device)
        return torch.all(x, 0)

    @autotest(auto_backward=False)
    def test_matrix_col_any_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=2, dtype=float, requires_grad=False).to(device)
        return torch.any(x, 0)

    @autotest(auto_backward=False)
    def test_all_keepdim_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=4, dtype=float, requires_grad=False).to(device)
        return torch.all(x, 1, keepdim=True)

    @autotest(auto_backward=False)
    def test_any_keepdim_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=4, dtype=float, requires_grad=False).to(device)
        return torch.any(x, 1, keepdim=True)


if __name__ == "__main__":
    unittest.main()
