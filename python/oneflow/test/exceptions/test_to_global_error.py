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
from collections import OrderedDict

import os
import numpy as np
import time
import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


@flow.unittest.skip_unless_1n2d()
class TestToGlobalError(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_tensor_to_consistent(self):
        with self.assertRaises(Exception) as context:
            data = flow.rand(2, dtype=flow.float32)
            placement = flow.placement.all("cuda")
            sbp = flow.sbp.split(0)
            global_data = data.to_consistent(placement=placement, sbp=sbp)

        self.assertTrue(
            ".to_consistent has been removed, please use .to_global instead"
            in str(context.exception)
        )

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_tensor_is_global(self):
        with self.assertRaises(Exception) as context:
            data = flow.rand(2, dtype=flow.float32)
            print(data.is_consistent())

        self.assertTrue(
            ".is_consistent has been removed, please use .is_global instead"
            in str(context.exception)
        )

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_module_to_consistent(self):
        with self.assertRaises(Exception) as context:
            m = flow.nn.Conv2d(1, 1, 1)
            placement = flow.placement.all("cuda")
            sbp = flow.sbp.split(0)
            m.to_consistent(placement=placement, sbp=sbp)

        self.assertTrue(
            ".to_consistent has been removed, please use .to_global instead"
            in str(context.exception)
        )


if __name__ == "__main__":
    unittest.main()
