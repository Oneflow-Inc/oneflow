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
import torch as ori_torch

import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *


@flow.unittest.skip_unless_1n1d()
class Test_Copy_module(flow.unittest.TestCase):
    def test_copy_broadcast_tensor(test_case):
        torch_base_grid = ori_torch.zeros(1, 2, 2, 3)
        flow_base_grid = flow.zeros(1, 2, 2, 3)
        torch_x_grid = ori_torch.ones(2)
        flow_x_grid = flow.ones(2)
        torch_base_grid[..., 0].copy_(torch_x_grid)
        flow_base_grid[..., 0].copy_(flow_x_grid)
        test_case.assertTrue(np.allclose(torch_base_grid.numpy(), flow_base_grid.numpy()))
    
    def test_non_contiguous_tensor_copy(test_case):
        torch_tensor = torch.arange(24, dtype=torch.float32).reshape(1,2,3,4)
        flow_tensor = flow.arange(24, dtype=flow.float32).reshape(1,2,3,4)
        torch_copy = torch.tensor([3.1415])
        flow_copy = flow.tensor([3.1415])
        torch_tensor[:,1:2,1:2,::2].copy_(torch_copy)
        flow_tensor[:,1:2,1:2,::2].copy_(flow_copy)
        test_case.assertTrue(np.allclose(flow_tensor.numpy(), torch_tensor.numpy()))


if __name__ == "__main__":
    unittest.main()
