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
# RUN: python3 %s | FileCheck %s

import os
import unittest
import numpy as np
os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
import oneflow as flow
import oneflow.unittest

@flow.unittest.skip_unless_1n1d()
class TestCastOpOptimalPass(flow.unittest.TestCase):
    def test_fused_op(test_case):
        a = flow.tensor([2, 3], dtype=flow.float64)
        eager_b = flow.cast(a, dtype=flow.float64)
        class CastOpOptimalPass(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.cast = flow.cast

            def build(self, x):
                return self.cast(x, dtype=flow.float64)      
                
        lazy_b = CastOpOptimalPass()(a)
        test_case.assertEqual(eager_b.dtype, lazy_b.dtype)

if __name__ == "__main__":
    unittest.main()
