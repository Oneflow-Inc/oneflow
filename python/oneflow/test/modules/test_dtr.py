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
import sys
import re
import unittest

import numpy as np

import oneflow as flow
import oneflow.unittest


# @unittest.skipUnless(os.getenv("OF_DTR"), "only test while DTR is on")
class TestDTR(flow.unittest.TestCase):
    flow.enable_dtr(True, "20KB", 0, "eq")

    def test_dtr_enabled(test_case):
        test_case.assertTrue(flow.check_dtr())

    def test_dtr_work_on_simple_case(test_case):
        x1 = flow.ones(1024).to('cuda') # x1 = 1, total memory: 1024 * 4 = 4096 bytes = 4KB
        x2 = x1 + 3                     # x2 = 4, total memory: 8KB
        x3 = x1 * x2                    # x3 = 4, total memory: 12KB
        x4 = x1 - x3                    # x4 = -3, total memory: 16KB
        x5 = x4.square()                # x5 = 9, total memory: 20KB
        x6 = x1 + x3                    # x6 = 5, evict a tensor
        x7 = x1 + x3                    # x7 = 5, evict a tensor

        # wait for the operations to finish
        flow._oneflow_internal.eager.multi_client.Sync()
        # check if there are 2 tensors are evicted
        not_in_memory_num = sum(0 if x._is_in_memory else 1 for x in [x1, x2, x3, x4, x5, x6, x7])
        test_case.assertEqual(not_in_memory_num, 2)

        # trigger recomputation and check the result
        y = x1 + x2 + x3 + x4 + x5 + x6 + x7

        test_case.assertTrue(np.array_equal(y.numpy(), 25 * np.ones(y.shape)))

    # def test_dtr_threshold(test_case):
    #     regex = re.compile(r"(\d+(?:\.\d+)?)\s*([kmg]?b)", re.IGNORECASE)
    #     magnitude = ["b", "kb", "mb", "gb"]
    #     out = regex.findall(TestDTR.THRES)
    #     test_case.assertEqual(len(out), 1)
        

if __name__ == "__main__":
    unittest.main()
