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


class TestDistributedEnvVars(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n1d()
    def test_default(test_case):
        test_case.assertFalse("MASTER_ADDR" in os.environ)
        test_case.assertFalse("MASTER_PORT" in os.environ)
        test_case.assertFalse("WORLD_SIZE" in os.environ)
        test_case.assertFalse("RANK" in os.environ)
        test_case.assertFalse("LOCAL_RANK" in os.environ)
        test_case.assertEqual(flow.distributed.get_world_size(), 1)
        test_case.assertEqual(flow.distributed.get_rank(), 0)
        test_case.assertEqual(flow.distributed.get_local_rank(), 0)

    @flow.unittest.skip_unless_1n2d()
    def test_1n2d(test_case):
        test_case.assertEqual(os.environ["MASTER_ADDR"], "127.0.0.1")
        test_case.assertEqual(os.environ["WORLD_SIZE"], "2")
        test_case.assertTrue(os.environ["RANK"] in ["0", "1"])
        test_case.assertTrue(os.environ["LOCAL_RANK"] in ["0", "1"])


if __name__ == "__main__":
    unittest.main()
