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
class TestFuseBNAddReLUCPUMLIR(oneflow.unittest.TestCase):
    def test_fused_bn_add_relu_graph(test_case):
        data = np.random.randn(2, 96, 96, 3)
        x = flow.tensor(data, dtype=flow.float32)
        bn = flow.nn.BatchNorm2d(96)
        relu = flow.nn.ReLU()
        y_eager = relu(bn(x) + 2.0) + 1.0

        class FuseBNAddReLUGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.bn = bn
                self.relu = relu

            def build(self, x):
                return self.relu(self.bn(x) + 2.0) + 1.0

        bn_add_relu = FuseBNAddReLUGraph()
        y_lazy = bn_add_relu(x)
        test_case.assertTrue(np.array_equal(y_eager.numpy(), y_lazy.numpy()))

@flow.unittest.skip_unless_1n1d()
class TestFuseBNAddReLUGPUMLIR(oneflow.unittest.TestCase):
    def test_fused_bn_add_relu_graph(test_case):
        data = np.random.randn(2, 96, 96, 3)
        x = flow.tensor(data, dtype=flow.float32).to("cuda")
        bn = flow.nn.BatchNorm2d(96)
        relu = flow.nn.ReLU()
        bn.to("cuda")
        relu.to("cuda")
        y_eager = relu(bn(x) + 2.0) + 1.0

        class FuseBNAddReLUGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.bn = bn
                self.relu = relu

            def build(self, x):
                return self.relu(self.bn(x) + 2.0) + 1.0

        bn_add_relu = FuseBNAddReLUGraph()
        y_lazy = bn_add_relu(x)
        test_case.assertTrue(np.array_equal(y_eager.detach().cpu().numpy(), y_lazy.numpy()))

if __name__ == "__main__":
    unittest.main()
