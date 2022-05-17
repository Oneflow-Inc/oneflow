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
import re
import unittest
import oneflow as flow
import oneflow.unittest
import oneflow.nn.functional as F


@flow.unittest.skip_unless_1n1d()
class TestPlacement(flow.unittest.TestCase):
    def test_placement_type(test_case):
        with test_case.assertRaises(RuntimeError) as exp:
            flow.placement(type="xpu", ranks=[0])
        test_case.assertTrue(
            re.match(
                "Expected one of (.*) device type at start of device string: xpu",
                str(exp.exception),
            )
            is not None
        )

    def test_placement_rank(test_case):
        with test_case.assertRaises(RuntimeError) as exp:
            flow.placement(type="cpu", ranks=[])
        test_case.assertTrue(
            "placement ranks shoule be an array of long int" in str(exp.exception)
        )

        with test_case.assertRaises(RuntimeError) as exp:
            placement = flow.placement(type="cpu", ranks=[1000])
            flow.Tensor(2, 3, placement=placement, sbp=flow.sbp.broadcast)
        test_case.assertTrue(
            "Placement is invalid because rank must be less than world size!"
            in str(exp.exception)
        )


if __name__ == "__main__":
    unittest.main()
