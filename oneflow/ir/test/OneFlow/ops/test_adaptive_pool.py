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

os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
os.environ["ONEFLOW_MLIR_ENABLE_CODEGEN_FUSERS"] = "1"

import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
class TestAdaptivePool1dMLIR(oneflow.unittest.TestCase):
    def test_adaptive_pool1d_graph(test_case):
        data = np.random.randn(1, 2, 3)
        x = flow.tensor(data, dtype=flow.float32)

        AdaptivePool1d = flow.nn.AdaptiveAvgPool1d(output_size=(1))
        y_eager = AdaptivePool1d(x)

        class AdaptivePool1dGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.cc_adaptive_pool = AdaptivePool1d

            def build(self, x):
                return self.cc_adaptive_pool(x)

        adaptive_pool_1d_g = AdaptivePool1dGraph()
        y_lazy = adaptive_pool_1d_g(x)
        test_case.assertTrue(np.array_equal(y_eager.numpy(), y_lazy.numpy()))


if __name__ == "__main__":
    unittest.main()
