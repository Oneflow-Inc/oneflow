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


class TestPlacement(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_placement_all_cuda(test_case):
        placement = flow.placement.all("cuda")
        test_case.assertEqual(placement.type, "cuda")
        # assertEqual fails to compare lists
        test_case.assertTrue(
            list(placement.ranks) == list(range(flow.env.get_world_size()))
        )

    def test_placement_all_cpu(test_case):
        placement = flow.placement.all("cpu")
        test_case.assertEqual(placement.type, "cpu")
        # assertEqual fails to compare lists
        test_case.assertTrue(
            list(placement.ranks) == list(range(flow.env.get_world_size()))
        )


if __name__ == "__main__":
    unittest.main()
