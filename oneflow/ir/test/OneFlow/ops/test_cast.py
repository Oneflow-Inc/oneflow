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
# RUN: python3 %s
import os
import unittest
import numpy as np

os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = '1'
os.environ["ONEFLOW_MLIR_ENABLE_CODEGEN_FUSERS"] = '1'

import oneflow as flow
import oneflow.unittest

@flow.unittest.skip_unless_1n1d()
class TestCastMLIR(oneflow.unittest.TestCase):
    def test_cast_graph(test_case):
        data1 = np.random.randn(20, 30)
        a = flow.tensor(data1, dtype=flow.float32)

        y_eager = flow.cast(a, dtype=flow.int8)

        class CastGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()

            def build(self, a):
                return flow.cast(a, dtype=flow.int8)

        cast_g = CastGraph()
        y_lazy = cast_g(a)

        # for i in range(100):
        #     y_lazy = conv2d_g(x)

        test_case.assertTrue(np.array_equal(y_eager.numpy(), y_lazy.numpy()))


if __name__ == "__main__":
    unittest.main()
