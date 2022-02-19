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
    BS = 80
    THRES = "3500mb"
    ITER = 40
    OF_DTR = True
    DEBUG_LEVEL = 0
    HEURISTIC = "eq"

    flow.enable_dtr(OF_DTR, THRES, DEBUG_LEVEL, HEURISTIC)

    def test_dtr_enabled(test_case):
        test_case.assertTrue(flow.check_dtr())

    def test_dtr_work_on_simple_case(test_case):
        x = 6
        y = x + 6 + 6 + 6 + 6 + 6
        test_case.assertEqual(y, 36)

    def test_dtr_threshold(test_case):
        regex = re.compile(r"(\d+(?:\.\d+)?)\s*([kmg]?b)", re.IGNORECASE)
        magnitude = ["b", "kb", "mb", "gb"]
        out = regex.findall(TestDTR.THRES)
        test_case.assertEqual(len(out), 1)
        

if __name__ == "__main__":
    # python3 python/oneflow/test/modules/test_dtr.py 80 8000mb 40 True
    if len(sys.argv) > 1:
        TestDTR.OF_DTR = bool(sys.argv.pop())
        TestDTR.ITER = sys.argv.pop()
        TestDTR.THRES = sys.argv.pop()
        TestDTR.BS = sys.argv.pop()
    unittest.main()
