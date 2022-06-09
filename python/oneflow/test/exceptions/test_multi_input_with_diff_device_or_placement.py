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
class TestModule(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n1d()
    def test_multi_input_with_diff_device(test_case):
        # torch exception and messge:
        #
        #   RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
        #
        x = flow.tensor([1, 2, 3, 4])
        y = flow.tensor([2, 4, 6, 8], device="cuda")
        with test_case.assertRaises(RuntimeError) as ctx:
            z = flow.add(x, y)
        test_case.assertTrue(
            "Expected all tensors to be on the same device, but found at least two devices"
            in str(ctx.exception)
        )

    @flow.unittest.skip_unless_1n2d()
    def test_multi_input_with_diff_placement(test_case):
        x = flow.tensor(
            [1, 2, 3, 4], placement=flow.placement("cuda", [0]), sbp=flow.sbp.broadcast
        )
        y = flow.tensor(
            [2, 4, 6, 8], placement=flow.placement("cuda", [1]), sbp=flow.sbp.broadcast
        )
        with test_case.assertRaises(RuntimeError) as ctx:
            z = flow.add(x, y)
        test_case.assertTrue(
            "Expected all tensors to be on the same placement, but found at least two placements"
            in str(ctx.exception)
        )


if __name__ == "__main__":
    unittest.main()
