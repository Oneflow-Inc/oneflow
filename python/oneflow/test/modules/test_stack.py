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

from oneflow.test_utils.automated_test_util import *


@flow.unittest.skip_unless_1n1d()
class TestStackModule(flow.unittest.TestCase):
    @autotest(check_graph=True)
    def test_stack_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=4, dim1=3, dim2=4, dim3=5).to(device)
        y = random_tensor(ndim=4, dim1=3, dim2=4, dim3=5).to(device)
        out = torch.stack((x, y), dim=random(low=-5, high=5).to(int))
        return out

    @autotest(auto_backward=False, check_graph=True)
    def test_stack_bool_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=4, dim1=3, dim2=4, dim3=5).to(
            device=device, dtype=torch.bool
        )
        y = random_tensor(ndim=4, dim1=3, dim2=4, dim3=5).to(
            device=device, dtype=torch.bool
        )
        out = torch.stack((x, y), dim=random(low=1, high=4).to(int))
        return out

    @autotest(check_graph=True)
    def test_column_stack_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=1, dim0=10).to(device)
        y = random_tensor(ndim=2, dim0=10, dim1=5).to(device)
        z = random_tensor(ndim=2, dim0=10, dim1=5).to(device)
        out = torch.column_stack((x, y, z))
        return out

    def test_column_stack_with_0dim_data(test_case):
        device = random_device()
        x = random_tensor(ndim=0).to(device)
        y = random_tensor(ndim=1, dim0=1).to(device)
        out = torch.column_stack((x, y))
        return out

    @autotest(check_graph=True)
    def test_row_stack_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=1, dim0=10).to(device)
        y = random_tensor(ndim=2, dim0=5, dim1=10).to(device)
        z = random_tensor(ndim=2, dim0=5, dim1=10).to(device)
        out = torch.row_stack((x, y, z))
        return out

    def test_row_stack_with_0dim_data(test_case):
        device = random_device()
        x = random_tensor(ndim=0).to(device)
        y = random_tensor(ndim=1, dim0=1).to(device)
        out = torch.row_stack((x, y))
        return out

    @autotest(check_graph=True)
    def test_hstack_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=1, dim0=5).to(device)
        y = random_tensor(ndim=1, dim0=5).to(device)
        out = torch.hstack((x, y))
        return out

    @autotest(check_graph=True)
    def test_hstack_with_0dim_data(test_case):
        device = random_device()
        x = random_tensor(ndim=0).to(device)
        y = random_tensor(ndim=0).to(device)
        # test 1-dim simultaneouslsimultaneouslyy
        z = random_tensor(ndim=1, dim0=1).to(device)
        out = torch.hstack((x, y, z))
        return out

    @autotest(check_graph=True)
    def test_vstack_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=2, dim0=3, dim1=4).to(device)
        y = random_tensor(ndim=1, dim0=4).to(device)
        z = random_tensor(ndim=2, dim0=3, dim1=4).to(device)
        out = torch.vstack((x, y, z))
        return out

    @autotest(check_graph=True)
    def test_vstack_with_0dim_data(test_case):
        device = random_device()
        x = random_tensor(ndim=0).to(device)
        y = random_tensor(ndim=0).to(device)
        out = torch.vstack((x, y))
        return out

    @autotest(check_graph=True)
    def test_dstack_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=2, dim0=1, dim1=4).to(device)
        y = random_tensor(ndim=3, dim0=1, dim1=4, dim2=1).to(device)
        z = random_tensor(ndim=1, dim0=4).to(device)
        out = torch.dstack((x, y, z))
        return out

    @autotest(check_graph=True)
    def test_dstack_with_0dim_data(test_case):
        device = random_device()
        x = random_tensor(ndim=0).to(device)
        y = random_tensor(ndim=0).to(device)
        z = random_tensor(ndim=0).to(device)
        out = torch.dstack((x, y, z))

    @autotest(auto_backward=True, check_graph=True)
    def test_stack_kMaxInputCount_inputs(test_case):
        kMaxInputCount = 128 + 1
        stack_list = [
            random_tensor(ndim=2, dim0=3, dim1=4) for _ in range(kMaxInputCount)
        ]
        out = torch.stack(stack_list, 0)
        return out


if __name__ == "__main__":
    unittest.main()
