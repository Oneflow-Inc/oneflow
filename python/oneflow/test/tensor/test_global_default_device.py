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

import os
import unittest
import oneflow as flow
import oneflow.unittest
from oneflow import nn


class CustomModule(nn.Module):
    def __init__(self, foo, bar, device=None):
        super().__init__()
        # ==== Case 1: Module creates parameters directly. ====
        self.param1 = nn.Parameter(flow.empty((foo, bar)))
        self.register_parameter("param2", nn.Parameter(flow.empty(bar)))
        with flow.no_grad():
            nn.init.kaiming_uniform_(self.param1)
            nn.init.uniform_(self.param2)
        # ==== Case 2: Module creates submodules. ====
        self.fc = nn.Linear(bar, 5)
        self.linears = nn.Sequential(nn.Linear(5, 5), nn.Linear(5, 1))
        # ==== Case 3: Module creates buffers. ====
        self.register_buffer("some_buffer", flow.ones(7))


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestTensor(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n1d()
    def test_creating_tensor_by_global_default_device(test_case):
        default_device = flow.get_global_default_device()
        test_case.assertEqual(default_device, flow.device("cpu"))

        def test_func(test_case, default_device="cpu"):
            flow.set_global_default_device(flow.device(default_device))
            default_device = flow.get_global_default_device()
            test_case.assertEqual(default_device, flow.device(default_device))
            x1 = flow.zeros(2, 3)
            x2 = flow.empty(2, 3)
            x3 = flow.ones(2, 3)
            x4 = flow.Tensor(2, 3)
            x5 = flow.Tensor([3, 4])
            x6 = flow.tensor([3, 4])
            test_case.assertEqual(x1.device, flow.device(default_device))
            test_case.assertEqual(x2.device, flow.device(default_device))
            test_case.assertEqual(x3.device, flow.device(default_device))
            test_case.assertEqual(x4.device, flow.device(default_device))
            test_case.assertEqual(x5.device, flow.device(default_device))
            test_case.assertEqual(x6.device, flow.device(default_device))

        test_func(test_case, "meta")
        test_func(test_case, "cuda")
        flow.set_global_default_device(flow.device("cpu"))

    @flow.unittest.skip_unless_1n1d()
    def test_creating_tensor_with_different_priority(test_case):
        flow.set_global_default_device(flow.device("meta"))
        x1 = flow.tensor([3, 4])
        x2 = flow.tensor([3, 4], device="cpu")
        test_case.assertEqual(x1.device, flow.device("meta"))
        test_case.assertEqual(x2.device, flow.device("cpu"))
        flow.set_global_default_device(flow.device("cpu"))

    @flow.unittest.skip_unless_1n1d()
    def test_skip_init_by_global_default_device(test_case):
        goal_device = "cpu"
        x = flow.nn.utils.skip_init(CustomModule, 4, 3)
        y = CustomModule(4, 3, device=goal_device)
        test_case.assertEqual(x.param1.dtype, y.param1.dtype)
        test_case.assertEqual(x.param1.shape, y.param1.shape)
        test_case.assertEqual(x.param1.requires_grad, y.param1.requires_grad)
        test_case.assertEqual(x.param1.device, flow.device(goal_device))
        test_case.assertEqual(x.param2.dtype, y.param2.dtype)
        test_case.assertEqual(x.param2.shape, y.param2.shape)
        test_case.assertEqual(x.param2.requires_grad, y.param2.requires_grad)
        test_case.assertEqual(x.param2.device, flow.device(goal_device))
        test_case.assertEqual(x.fc.weight.dtype, y.fc.weight.dtype)
        test_case.assertEqual(x.fc.weight.shape, y.fc.weight.shape)
        test_case.assertEqual(x.fc.weight.requires_grad, y.fc.weight.requires_grad)
        test_case.assertEqual(x.fc.weight.device, flow.device(goal_device))
        test_case.assertEqual(x.linears[0].weight.dtype, y.linears[0].weight.dtype)
        test_case.assertEqual(x.linears[0].weight.shape, y.linears[0].weight.shape)
        test_case.assertEqual(
            x.linears[0].weight.requires_grad, y.linears[0].weight.requires_grad
        )
        test_case.assertEqual(x.linears[0].weight.device, flow.device(goal_device))
        test_case.assertEqual(x.linears[1].weight.dtype, y.linears[1].weight.dtype)
        test_case.assertEqual(x.linears[1].weight.shape, y.linears[1].weight.shape)
        test_case.assertEqual(
            x.linears[1].weight.requires_grad, y.linears[1].weight.requires_grad
        )
        test_case.assertEqual(x.linears[1].weight.device, flow.device(goal_device))
        test_case.assertEqual(x.some_buffer.dtype, y.some_buffer.dtype)
        test_case.assertEqual(x.some_buffer.shape, y.some_buffer.shape)
        test_case.assertEqual(x.some_buffer.requires_grad, y.some_buffer.requires_grad)
        test_case.assertEqual(x.some_buffer.device, flow.device(goal_device))


if __name__ == "__main__":
    unittest.main()
