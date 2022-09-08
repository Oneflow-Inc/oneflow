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
class TestDevice(flow.unittest.TestCase):
    def test_device_type(test_case):
        with test_case.assertRaises(RuntimeError) as exp:
            flow.device("xpu")
        test_case.assertTrue(
            re.match(
                "Expected one of (.*) device type at start of device string: xpu",
                str(exp.exception),
            )
            is not None
        )

    def test_device_index(test_case):
        # TODO(hjchen2): throw runtime error if cuda reports error
        #     with test_case.assertRaises(RuntimeError) as exp:
        #         device = flow.device("cuda:1000")
        #         flow.Tensor(2, 3).to(device=device)
        #     test_case.assertTrue("CUDA error: invalid device ordinal" in str(exp.exception))
        pass


if __name__ == "__main__":
    unittest.main()
