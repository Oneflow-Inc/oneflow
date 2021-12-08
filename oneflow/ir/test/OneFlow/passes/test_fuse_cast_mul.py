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
import unittest
import numpy as np

import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
class TestFuseCastMuldMLIR(oneflow.unittest.TestCase):
    def test_fused_cast_mul_graph(test_case):
        data = np.random.randn(1, 2, 3)
        x = flow.tensor(data, dtype=flow.float32)
        y_eager = flow.cast(x, flow.int32) * 2.0

        class FuseCastMulGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()

            def build(self, x):
                return flow.cast(x, flow.int32) * 2.0

        cast_mul = FuseCastMulGraph()
        y_lazy = cast_mul(x)
        test_case.assertTrue(np.array_equal(y_eager.numpy(), y_lazy.numpy()))


if __name__ == "__main__":
    unittest.main()
