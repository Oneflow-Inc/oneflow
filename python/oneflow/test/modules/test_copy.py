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
        # TODO: copy op not support non-contiguous input tensor
        flow_base_grid[..., 0].contiguous().copy_(flow_x_grid)
        test_case.assertTrue(np.allclose(torch_base_grid.size(), flow_base_grid.size()))


if __name__ == "__main__":
    unittest.main()
