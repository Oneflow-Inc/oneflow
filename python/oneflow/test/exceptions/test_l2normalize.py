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


class TestL2NormalizeError(flow.unittest.TestCase):
    def test_l2normalize_axis_error1(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow.ones((3, 3), dtype=flow.float32)
            out = flow._C.normalize(x, dim=3, use_l2_norm_kernel=True)
        test_case.assertTrue(
            "Check failed: (axis_) <= (final_dim) (3 vs 1) Axis should <2 but axis is 3 now."
            in str(ctx.exception)
        )

    def test_l2normalize_axis_error2(test_case):
        with test_case.assertRaises(Exception) as ctx:
            x = flow.ones((3, 3), dtype=flow.float32)
            out = flow._C.normalize(x, dim=-3, use_l2_norm_kernel=True)
        test_case.assertTrue(
            "Check failed: (axis_) >= (0) (-1 vs 0) Axis should >=0 but axis is -1 now."
            in str(ctx.exception)
        )


if __name__ == "__main__":
    unittest.main()
