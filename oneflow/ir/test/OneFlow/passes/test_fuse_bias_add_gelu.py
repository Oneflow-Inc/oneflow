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
class TestFuseBiasAddGeLUCPUMLIR(oneflow.unittest.TestCase):
    def test_fused_bias_add_gelu_graph(test_case):
        data = np.random.randn(1, 2, 3)
        bias_data = np.random.randn(2)
        x = flow.tensor(data, dtype=flow.float32)
        bias = flow.tensor(bias_data, dtype=flow.float32)
        y_eager = flow.gelu(flow._C.bias_add(x, bias, axis=1))

        class FuseBiasAddGeLUGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()

            def build(self, x):
                return flow.gelu(flow._C.bias_add(x, bias, axis=1))

        bias_add_gelu = FuseBiasAddGeLUGraph()
        y_lazy = bias_add_gelu(x)
        test_case.assertTrue(np.array_equal(y_eager.numpy(), y_lazy.numpy()))


@flow.unittest.skip_unless_1n1d()
class TestFuseBiasAddGeLUGPUMLIR(oneflow.unittest.TestCase):
    def test_fused_bias_add_gelu_graph(test_case):
        data = np.random.randn(1, 2, 3)
        bias_data = np.random.randn(2)
        x = flow.tensor(data, dtype=flow.float32).to("gpu")
        bias = flow.tensor(bias_data, dtype=flow.float32).to("gpu")
        y_eager = flow.gelu(flow._C.bias_add(x, bias, axis=1))

        class FuseBiasAddGeLUGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()

            def build(self, x):
                return flow.gelu(flow._C.bias_add(x, bias, axis=1))

        bias_add_gelu = FuseBiasAddGeLUGraph()
        y_lazy = bias_add_gelu(x)
        test_case.assertTrue(np.array_equal(y_eager.detach().cpu().numpy(), y_lazy.numpy()))

if __name__ == "__main__":
    unittest.main()
