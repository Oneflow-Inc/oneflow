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
        self.param1 = nn.Parameter(flow.empty((foo, bar), device=device))
        self.register_parameter("param2", nn.Parameter(flow.empty(bar, device=device)))
        with flow.no_grad():
            nn.init.kaiming_uniform_(self.param1)
            nn.init.uniform_(self.param2)
        # ==== Case 2: Module creates submodules. ====
        self.fc = nn.Linear(bar, 5, device=device)
        self.linears = nn.Sequential(
            nn.Linear(5, 5, device=device), nn.Linear(5, 1, device=device)
        )
        # ==== Case 3: Module creates buffers. ====
        self.register_buffer("some_buffer", flow.ones(7, device=device))


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestMetaTensor(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n1d()
    def test_meta_tensor_local_mode_without_data(test_case):
        x = flow.Tensor(3, 2, device="meta")
        y = flow.Tensor(3, 2, device="cpu")
        test_case.assertEqual(x.dtype, y.dtype)
        test_case.assertEqual(x.shape, y.shape)
        test_case.assertEqual(x.device, flow.device("meta"))

    @flow.unittest.skip_unless_1n1d()
    def test_meta_tensor_local_mode_with_data(test_case):
        x = flow.Tensor([3, 2], device="meta")
        y = flow.Tensor([3, 2], device="cpu")
        test_case.assertEqual(x.dtype, y.dtype)
        test_case.assertEqual(x.shape, y.shape)
        test_case.assertEqual(x.device, flow.device("meta"))

    @flow.unittest.skip_unless_1n1d()
    def test_meta_tensor_func_local_mode_without_data(test_case):
        x = flow.tensor([3, 2], device="meta")
        y = flow.tensor([3, 2], device="cpu")
        test_case.assertEqual(x.dtype, y.dtype)
        test_case.assertEqual(x.shape, y.shape)
        test_case.assertEqual(x.device, flow.device("meta"))

    @flow.unittest.skip_unless_1n1d()
    def test_meta_tensor_func_local_mode_with_data(test_case):
        x = flow.tensor([3, 2], device="meta")
        y = flow.tensor([3, 2], device="cpu")
        test_case.assertEqual(x.dtype, y.dtype)
        test_case.assertEqual(x.shape, y.shape)
        test_case.assertEqual(x.device, flow.device("meta"))

    @flow.unittest.skip_unless_1n1d()
    def test_meta_tensor_local_mode_ones(test_case):
        x = flow.ones(3, 2, device="meta")
        y = flow.ones([3, 2], device="cpu")
        test_case.assertEqual(x.dtype, y.dtype)
        test_case.assertEqual(x.shape, y.shape)
        test_case.assertEqual(x.device, flow.device("meta"))

    @flow.unittest.skip_unless_1n1d()
    def test_meta_tensor_local_mode_linear(test_case):
        x = flow.nn.Linear(3, 2, device="meta")
        y = flow.nn.Linear(3, 2, device="cpu")
        test_case.assertEqual(x.weight.dtype, y.weight.dtype)
        test_case.assertEqual(x.weight.shape, y.weight.shape)
        test_case.assertEqual(x.weight.requires_grad, y.weight.requires_grad)
        test_case.assertEqual(x.weight.device, flow.device("meta"))

    @flow.unittest.skip_unless_1n1d()
    def test_skip_init_function(test_case):
        x = flow.nn.utils.skip_init(flow.nn.Linear, 4, 3)
        y = flow.nn.Linear(4, 3, device="cpu")
        test_case.assertEqual(x.weight.dtype, y.weight.dtype)
        test_case.assertEqual(x.weight.shape, y.weight.shape)
        test_case.assertEqual(x.weight.requires_grad, y.weight.requires_grad)
        test_case.assertEqual(x.weight.device, flow.device("cpu"))

    @flow.unittest.skip_unless_1n1d()
    def test_skip_init_function_custom_module(test_case):
        x = flow.nn.utils.skip_init(CustomModule, 4, 3)
        y = CustomModule(4, 3, device="cpu")
        test_case.assertEqual(x.param1.dtype, y.param1.dtype)
        test_case.assertEqual(x.param1.shape, y.param1.shape)
        test_case.assertEqual(x.param1.requires_grad, y.param1.requires_grad)
        test_case.assertEqual(x.param1.device, flow.device("cpu"))
        test_case.assertEqual(x.param2.dtype, y.param2.dtype)
        test_case.assertEqual(x.param2.shape, y.param2.shape)
        test_case.assertEqual(x.param2.requires_grad, y.param2.requires_grad)
        test_case.assertEqual(x.param2.device, flow.device("cpu"))
        test_case.assertEqual(x.fc.weight.dtype, y.fc.weight.dtype)
        test_case.assertEqual(x.fc.weight.shape, y.fc.weight.shape)
        test_case.assertEqual(x.fc.weight.requires_grad, y.fc.weight.requires_grad)
        test_case.assertEqual(x.fc.weight.device, flow.device("cpu"))

    @flow.unittest.skip_unless_1n1d()
    def test_meta_tensor_local_mode_clone(test_case):
        x = flow.tensor([3, 2], device="meta")
        y = x.clone()
        test_case.assertEqual(x.dtype, y.dtype)
        test_case.assertEqual(x.shape, y.shape)
        test_case.assertEqual(x.device, y.device)

    @flow.unittest.skip_unless_1n1d()
    def test_meta_tensor_global_mode_without_data(test_case):
        P1 = flow.placement(type="meta", ranks=[0])
        P2 = flow.placement(type="cpu", ranks=[0])
        sbp = flow.sbp.broadcast
        x = flow.Tensor(3, 2, placement=P1, sbp=sbp)
        y = flow.Tensor(3, 2, placement=P2, sbp=sbp)
        test_case.assertEqual(x.dtype, y.dtype)
        test_case.assertEqual(x.shape, y.shape)
        test_case.assertEqual(x.sbp, y.sbp)
        test_case.assertEqual(x.placement.type, "meta")
        test_case.assertEqual(x.to_local().dtype, y.to_local().dtype)
        test_case.assertEqual(x.to_local().shape, y.to_local().shape)
        test_case.assertEqual(x.to_local().device.type, "meta")

    @flow.unittest.skip_unless_1n1d()
    def test_meta_tensor_global_mode_with_data(test_case):
        P1 = flow.placement(type="meta", ranks=[0])
        P2 = flow.placement(type="cpu", ranks=[0])
        sbp = flow.sbp.broadcast
        x = flow.Tensor([3, 2], placement=P1, sbp=sbp)
        y = flow.Tensor([3, 2], placement=P2, sbp=sbp)
        test_case.assertEqual(x.dtype, y.dtype)
        test_case.assertEqual(x.shape, y.shape)
        test_case.assertEqual(x.sbp, y.sbp)
        test_case.assertEqual(x.placement.type, "meta")
        test_case.assertEqual(x.to_local().dtype, y.to_local().dtype)
        test_case.assertEqual(x.to_local().shape, y.to_local().shape)
        test_case.assertEqual(x.to_local().device.type, "meta")

    @flow.unittest.skip_unless_1n1d()
    def test_meta_tensor_func_global_mode_without_data(test_case):
        P1 = flow.placement(type="meta", ranks=[0])
        P2 = flow.placement(type="cpu", ranks=[0])
        sbp = flow.sbp.broadcast
        x = flow.tensor([3, 2], placement=P1, sbp=sbp)
        y = flow.tensor([3, 2], placement=P2, sbp=sbp)
        test_case.assertEqual(x.dtype, y.dtype)
        test_case.assertEqual(x.shape, y.shape)
        test_case.assertEqual(x.sbp, y.sbp)
        test_case.assertEqual(x.placement.type, "meta")
        test_case.assertEqual(x.to_local().dtype, y.to_local().dtype)
        test_case.assertEqual(x.to_local().shape, y.to_local().shape)
        test_case.assertEqual(x.to_local().device.type, "meta")

    @flow.unittest.skip_unless_1n1d()
    def test_meta_tensor_global_mode_clone(test_case):
        P = flow.placement(type="meta", ranks=[0])
        sbp = flow.sbp.broadcast
        x = flow.tensor([3, 2], placement=P, sbp=sbp)
        y = x.clone()
        test_case.assertEqual(x.dtype, y.dtype)
        test_case.assertEqual(x.shape, y.shape)
        test_case.assertEqual(x.sbp, y.sbp)
        test_case.assertEqual(x.placement, y.placement)

    @flow.unittest.skip_unless_1n1d()
    def test_meta_tensor_calculate(test_case):
        x1 = flow.tensor([3, 2], device="meta")
        y1 = x1 + 1
        P = flow.placement(type="meta", ranks=[0])
        sbp = flow.sbp.broadcast
        x2 = flow.tensor([3, 2], placement=P, sbp=sbp)
        y2 = x2 + 1
        test_case.assertEqual(y1.device.type, "meta")
        test_case.assertEqual(y2.placement.type, "meta")


if __name__ == "__main__":
    unittest.main()
