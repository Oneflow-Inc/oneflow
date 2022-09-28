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


class TestPlacement(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n2d()
    def test_inconsistent_placement(test_case):
        x = flow.randn(2, 3)
        if flow.env.get_rank() == 0:
            placement = flow.placement("cpu", [0, 1])
        else:
            placement = flow.placement("cpu", [0])
        sbp = flow.sbp.split(1)
        with test_case.assertRaises(RuntimeError) as ctx:
            x_global = x.to_global(placement=placement, sbp=sbp)
        test_case.assertTrue("Inconsistent parallel description" in str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
