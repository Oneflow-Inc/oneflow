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
# RUN: python3 -m oneflow.test_utils.throttle --with-cuda=%with_cuda python3 %s | FileCheck %s
# CHECK-NOT: oneflow.cast

import os
import unittest
import numpy as np

os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
os.environ["ONEFLOW_MLIR_FUSE_FORWARD_OPS"] = "1"
os.environ["ONEFLOW_MLIR_STDOUT"] = "1"
os.environ["ONEFLOW_MLIR_ENABLE_TIMING"] = "1"
os.environ["ONEFLOW_MLIR_PRINT_STATS"] = "1"
os.environ["ONEFLOW_MLIR_ENABLE_IR_PRINTING"] = "1"
import oneflow as flow
import oneflow.unittest


def _cast_optimal_pass(test_case, dtype):
    a = flow.tensor([2, 3], dtype=dtype)
    eager_b = flow.cast(a, dtype=dtype)

    class CastOpOptimalPass(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.cast = flow.cast

        def build(self, x):
            return self.cast(x, dtype=dtype)

    lazy_b = CastOpOptimalPass()(a)
    test_case.assertEqual(eager_b.dtype, lazy_b.dtype)


@flow.unittest.skip_unless_1n1d()
class TestCastOpOptimalPass(flow.unittest.TestCase):
    def test_case_optimal_pass(test_case):
        for dtype in [flow.float32, flow.float64, flow.int32, flow.int64]:
            _cast_optimal_pass(test_case, dtype)


if __name__ == "__main__":
    unittest.main()
