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


@flow.unittest.skip_unless_1n1d()
class TestScatterOpsModule(flow.unittest.TestCase):
    @autotest(n=5)
    def test_scatter_random_data_at_dim_0(test_case):
        device = random_device()
        input = random_tensor(ndim=2, dim0=2, dim1=2).to(device)
        src = random_tensor(ndim=2, dim0=2, dim1=2).to(device)
        index = constant(
            torch.tensor(np.array([[0, 1], [1, 0]]), dtype=torch.int64, device=device)
        )
        y = torch.scatter(input, 0, index, src)
        return y

    @autotest(n=5)
    def test_scatter_random_data_at_dim_1(test_case):
        device = random_device()
        input = random_tensor(ndim=2, dim0=2, dim1=2).to(device)
        src = random_tensor(ndim=2, dim0=2, dim1=2).to(device)
        index = constant(
            torch.tensor(np.array([[1, 0], [0, 1]]), dtype=torch.int64, device=device)
        )
        y = torch.scatter(input, 1, index, src)
        return y

    @autotest(n=5)
    def test_scatter_scalar_random_data_at_dim0(test_case):
        device = random_device()
        input = random_tensor(ndim=2, dim0=2, dim1=2).to(device)
        src = random_tensor(ndim=2, dim0=2, dim1=2).to(device)
        index = constant(
            torch.tensor(np.array([[0, 1], [1, 0]]), dtype=torch.int64, device=device)
        )
        y = torch.scatter(input, 0, index, 3.14)
        return y

    @autotest(n=5)
    def test_scatter_scalar_random_data_at_dim1(test_case):
        device = random_device()
        input = random_tensor(ndim=2, dim0=2, dim1=2).to(device)
        src = random_tensor(ndim=2, dim0=2, dim1=2).to(device)
        index = constant(
            torch.tensor(np.array([[1, 0], [0, 1]]), dtype=torch.int64, device=device)
        )
        y = torch.scatter(input, 1, index, 3.14)
        return y

    @autotest(n=5)
    def test_scatter_add_random_data_at_dim0(test_case):
        device = random_device()
        input = random_tensor(ndim=2, dim0=2, dim1=2).to(device)
        src = random_tensor(ndim=2, dim0=2, dim1=2).to(device)
        index = constant(
            torch.tensor(np.array([[1, 0], [0, 1]]), dtype=torch.int64, device=device)
        )
        y = torch.scatter_add(input, 0, index, src)
        return y

    @autotest(n=5)
    def test_scatter_add_random_data_at_dim1(test_case):
        device = random_device()
        input = random_tensor(ndim=2, dim0=2, dim1=2).to(device)
        src = random_tensor(ndim=2, dim0=2, dim1=2).to(device)
        index = constant(
            torch.tensor(np.array([[0, 1], [1, 0]]), dtype=torch.int64, device=device)
        )
        y = torch.scatter_add(input, 1, index, src)
        return y


if __name__ == "__main__":
    unittest.main()
