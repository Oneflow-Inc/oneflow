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
@flow.unittest.skip_unless_1n2d()
class TestEnv(flow.unittest.TestCase):
    def test_get_device_count(test_case):
        test_case.assertEqual(flow.cuda.device_count(), 2)

    def test_current_device_idx(test_case):
        test_case.assertEqual(flow.cuda.current_device(), flow.env.get_rank())

    def test_cuda_is_available(test_case):
        test_case.assertEqual(flow.cuda.is_available(), True)


if __name__ == "__main__":
    unittest.main()
