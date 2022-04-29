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
from oneflow.test_utils.automated_test_util import *
import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
class TestModuleToHalf(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_module_to_half(test_case):
        input = flow.randn(10, 10).to(flow.float16).cuda()
        model = flow.nn.Linear(10, 20).half().cuda()
        output = model(input)
        test_case.assertEqual(output.dtype, flow.float16)


if __name__ == "__main__":
    unittest.main()
