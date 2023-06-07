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
import os
import oneflow as flow
import oneflow.unittest
import numpy as np
from test_run_graph_by_vm import RunGraphByVmEnv, Graph
from test_graph_ofrecord_reader import OFRecordDataLoader


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestGlobalInterpreter(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n2d()
    def test_data_parallel_run_by_vm(test_case):
        with RunGraphByVmEnv():

            class DataParallelMul(flow.nn.Module):
                def __init__(self, placement) -> None:
                    super().__init__()
                    self.w = flow.randn(
                        5, 8, placement=placement, sbp=flow.sbp.broadcast
                    )

                def forward(self, x):
                    return flow.matmul(x, self.w)

            placement = flow.placement("cuda", [0, 1])

            m = DataParallelMul(placement).eval()
            g = Graph(m)

            input = flow.randn(4, 5, placement=placement, sbp=flow.sbp.split(0))
            graph_output = g(input)
            eager_output = m(input)

            test_case.assertTrue(graph_output.sbp == eager_output.sbp)
            test_case.assertTrue(graph_output.shape == eager_output.shape)
            test_case.assertTrue(graph_output.placement == eager_output.placement)
            test_case.assertTrue(np.allclose(graph_output, eager_output))

    @flow.unittest.skip_unless_1n2d()
    def test_module_parallel_run_by_vm(test_case):
        with RunGraphByVmEnv():

            class ModuleParallelMul(flow.nn.Module):
                def __init__(self, placement) -> None:
                    super().__init__()
                    self.w = flow.randn(
                        5, 8, placement=placement, sbp=flow.sbp.split(1)
                    )

                def forward(self, x):
                    return flow.matmul(x, self.w)

            placement = flow.placement("cuda", [0, 1])
            m = ModuleParallelMul(placement).eval()
            g = Graph(m)

            input = flow.randn(4, 5, placement=placement, sbp=flow.sbp.broadcast)
            graph_output = g(input)
            eager_output = m(input)

            test_case.assertTrue(graph_output.sbp == eager_output.sbp)
            test_case.assertTrue(graph_output.shape == eager_output.shape)
            test_case.assertTrue(graph_output.placement == eager_output.placement)
            test_case.assertTrue(np.allclose(graph_output, eager_output))

    @flow.unittest.skip_unless_1n2d()
    def test_boxing_data_parallel_run_by_vm(test_case):
        with RunGraphByVmEnv():
            flow.boxing.nccl.enable_use_compute_stream(False)

            class BoxingModuleParallelMul(flow.nn.Module):
                def __init__(self, placement) -> None:
                    super().__init__()
                    self.w1 = flow.randn(
                        5, 8, placement=placement, sbp=flow.sbp.split(1)
                    )
                    self.w2 = flow.randn(
                        8, 6, placement=placement, sbp=flow.sbp.split(1)
                    )

                def forward(self, x):
                    x = flow.matmul(x, self.w1)
                    x = flow.matmul(x, self.w2)
                    return x

            placement = flow.placement("cuda", [0, 1])
            m = BoxingModuleParallelMul(placement).eval()
            g = Graph(m)

            input = flow.randn(4, 5, placement=placement, sbp=flow.sbp.broadcast)
            graph_output = g(input)
            eager_output = m(input)

            test_case.assertTrue(graph_output.sbp == eager_output.sbp)
            test_case.assertTrue(graph_output.shape == eager_output.shape)
            test_case.assertTrue(graph_output.placement == eager_output.placement)
            test_case.assertTrue(np.allclose(graph_output, eager_output))

    @flow.unittest.skip_unless_1n1d()
    def test_empty_inputs(test_case):
        with RunGraphByVmEnv():

            class GraphReader(flow.nn.Graph):
                def __init__(self):
                    super().__init__()
                    self.my_reader = OFRecordDataLoader()

                def build(self):
                    return self.my_reader()

            reader_g = GraphReader()
            image, label = reader_g()


if __name__ == "__main__":
    unittest.main()
