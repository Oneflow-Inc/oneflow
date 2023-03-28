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


if __name__ == "__main__":
  unittest.main()