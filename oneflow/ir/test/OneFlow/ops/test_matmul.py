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

import oneflow as flow
import oneflow.unittest

os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = '1'
os.environ["ONEFLOW_MLIR_ENABLE_CODEGEN_FUSERS"] = '1'

@flow.unittest.skip_unless_1n1d()
class TestMatMulMLIR(oneflow.unittest.TestCase):
    def test_matmul_graph(test_case):
        data1 = np.random.randn(20, 30)
        a = flow.tensor(data1, dtype=flow.float32)
        data2 = np.random.randn(30, 30)
        b = flow.tensor(data2, dtype=flow.float32)

        y_eager = flow._C.matmul(a, b)

        class MatMulGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()

            def build(self, a, b):
                return flow._C.matmul(a, b)

        matmul_g = MatMulGraph()
        y_lazy = matmul_g(a, b)
        
        # for i in range(100):
        #     y_lazy = conv2d_g(x)

        test_case.assertTrue(np.array_equal(y_eager.numpy(), y_lazy.numpy()))


if __name__ == "__main__":
    unittest.main()
