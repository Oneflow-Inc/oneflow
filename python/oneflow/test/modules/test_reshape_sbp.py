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
import oneflow.unittest
import oneflow as flow


@flow.unittest.skip_unless_1n2d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestReshapeSbp(flow.unittest.TestCase):
    def test_reshape_sbp(test_case):
        input = flow.rand(
            9, 9, 8, placement=flow.placement("cuda", [0, 1]), sbp=flow.sbp.split(0)
        )

        output = input.view(81, 8)
        test_case.assertTrue(output.sbp[0] != flow.sbp.split(0))


if __name__ == "__main__":
    unittest.main()
