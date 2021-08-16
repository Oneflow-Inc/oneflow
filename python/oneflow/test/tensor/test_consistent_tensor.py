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
from automated_test_util import *

import oneflow as flow
import oneflow.unittest

@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestTensor(flow.unittest.TestCase):

    def test_creating_consistent_tensor(test_case):
        placement = flow.placement("cuda", {0: 0})
        sbp = flow.sbp.broadcast
        shape = (2, 3)

        # Shape -> ConsistentTensor
        x = flow.Tensor(*shape, placement=placement, sbp=sbp)
        test_case.assertTrue(x.is_consistent)

        # LocalTensor -> ConsistentTensor
        x = flow.Tensor(*shape, device="cpu")
        test_case.assertTrue(x.is_local)
        y = flow.Tensor(x, placement=placement, sbp=sbp)
        test_case.assertTrue(y.is_consistent)

        # ConsistentTensor -> ConsistentTensor
        z = flow.Tensor(y, placement=placement, sbp=sbp)
        test_case.assertTrue(z.is_consistent)

        # TODO: ndarray -> ConsistentTensor

    def test_construct_local_from_consistent_tensor(test_case):
        placement = flow.placement("cuda", {0: 0})
        sbp = flow.sbp.broadcast
        shape = (2, 3)
        x = flow.Tensor(*shape, placement=placement, sbp=sbp)
        test_case.assertTrue(x.is_consistent)
        # ConsistentTensor -> LocalTensor
        y = flow.Tensor(x)
        test_case.assertTrue(y.is_local)
        y = flow.Tensor(x, device="cuda")
        test_case.assertTrue(y.is_local)

        print("test---------->")

if __name__ == "__main__":
    unittest.main()
