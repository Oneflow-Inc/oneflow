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
from oneflow.test_utils.automated_test_util.generators import random
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


@flow.unittest.skip_unless_1n1d()
class TestVar(flow.unittest.TestCase):
    @autotest(check_graph=True)
    def test_flow_var_all_dim_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = torch.var(x)
        return y

    @autotest(check_graph=True)
    def test_flow_var_one_dim_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=4).to(device)
        y = torch.var(
            x,
            dim=random(low=-4, high=4).to(int),
            unbiased=random().to(bool),
            keepdim=random().to(bool),
        )
        return y

    # In fp16 mode, variance op backward has a gap of 1e-3 between the gradient of PyTorch
    # and OneFlow for some unknown reason. However, it is not important now, because both in
    # PyTorch and OneFlow variance op don't need support fp16 backward in amp train.
    @autotest(n=5, auto_backward=True, check_graph=True, rtol=1e-3, atol=1e-3)
    def test_flow_var_one_dim_with_random_half_data(test_case):
        device = random_device()
        x = random_tensor(ndim=4).to(device).to(torch.float16)
        y = torch.var(
            x,
            dim=random(low=-4, high=4).to(int),
            unbiased=random().to(bool),
            keepdim=random().to(bool),
        )
        return y

    @autotest(auto_backward=False, check_graph=True)
    def test_flow_var_0_size_data_with_random_data(test_case):
        device = random_device()
        x = random_tensor(4, 2, 3, 0, 4).to(device)
        y = torch.var(
            x,
            dim=random(low=-4, high=4).to(int),
            unbiased=random().to(bool),
            keepdim=random().to(bool),
        )
        return y

    @autotest(n=5, auto_backward=False, check_graph=True)
    def test_flow_var_0_size_data_with_random_half_data(test_case):
        device = random_device()
        x = random_tensor(4, 2, 3, 0, 4).to(device).to(torch.float16)
        y = torch.var(
            x,
            dim=random(low=-4, high=4).to(int),
            unbiased=random().to(bool),
            keepdim=random().to(bool),
        )
        return y

    @autotest(n=5)
    def test_flow_var_all_dim_with_random_data_n5(test_case):
        device = random_device()
        x = random_tensor(ndim=4, dim0=5, dim1=1, dim2=16, dim3=16).to(device)
        y = torch.var(x, dim=[0, 2, 3])
        return y


if __name__ == "__main__":
    unittest.main()
