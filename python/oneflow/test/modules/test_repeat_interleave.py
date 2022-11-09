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

import numpy as np
import oneflow as flow
import oneflow.unittest
import torch as torch_original

from oneflow.test_utils.automated_test_util import *


@flow.unittest.skip_unless_1n1d()
class TestRepeatInterLeave(flow.unittest.TestCase):
    @autotest(n=5)
    def test_flow_int_repeat_interleave_dim_none(test_case):
        x = random_tensor(ndim=2, dim0=1, dim1=2)
        y = torch.repeat_interleave(x, 2)
        return y

    @autotest(n=5)
    def test_flow_int_repeat_interleave_with_dim(test_case):
        x = random_tensor(ndim=3, dim0=2, dim1=2, dim2=3)
        dim = random(low=0, high=2).to(int)
        y = torch.repeat_interleave(x, 2, dim)
        return y

    @autotest(n=5)
    def test_flow_tensor_repeat_interleave_dim(test_case):
        x = random_tensor(ndim=3, dim0=2, dim1=2, dim2=3)
        y = random_tensor(ndim=1, dim0=2, dtype=int, low=0, high=4)
        z = torch.repeat_interleave(x, y, 1)
        return z

    @autotest(n=5)
    def test_flow_tensor_repeat_interleave_dim_with_output_size(test_case):
        x = random_tensor(ndim=3, dim0=2, dim1=2, dim2=3)
        y = random_tensor(ndim=1, dim0=2, dtype=int, low=0, high=4)
        z = torch.repeat_interleave(x, y, 1, output_size=2)
        return z

    def test_flow_tensor_repeat_interleave_0size_tensor(test_case):
        np_arr = np.array(
            [
                [[0.8548, 0.0436, 0.7977], [0.1919, 0.4191, 0.2186]],
                [[0.4741, 0.8896, 0.6859], [0.5223, 0.7803, 0.1134]],
            ]
        )
        x_torch = torch_original.tensor(np_arr)
        x_torch.requires_grad = True
        y_torch = torch_original.tensor([0, 0])
        z_torch = torch_original.repeat_interleave(x_torch, y_torch, 1)
        z_torch.sum().backward()

        x_flow = flow.tensor(np_arr)
        x_flow.requires_grad = True
        y_flow = flow.tensor([0, 0])
        z_flow = flow.repeat_interleave(x_flow, y_flow, 1)
        z_flow.sum().backward()
        test_case.assertTrue(np.array_equal(x_torch.grad.numpy(), x_flow.grad.numpy()))


if __name__ == "__main__":
    unittest.main()
