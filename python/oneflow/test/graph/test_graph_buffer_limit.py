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

import os
import time
import unittest
import numpy as np

import oneflow as flow
import oneflow.unittest


def _test_graph_buffer_limit(test_case):
    class StageLayerModule(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = flow.nn.Linear(10, 8, False)
            self.linear2 = flow.nn.Linear(8, 10, False)
            flow.nn.init.constant_(self.linear1.weight, 0.023)
            flow.nn.init.constant_(self.linear2.weight, 1.23)

        def forward(self, x):
            out0 = self.linear1(x)
            out0 = out0 + 1.0
            out0 = out0 * 2.0
            out1 = self.linear2(out0)
            return out1

    P0 = flow.placement("cuda", ranks=[0])
    P1 = flow.placement("cuda", ranks=[1])
    PT = flow.placement("cuda", ranks=[0, 1])
    B = flow.sbp.broadcast

    class PipelineModule(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_0 = StageLayerModule()
            self.layer_1 = StageLayerModule()
            self.layer_0.to_global(P0, B)
            self.layer_1.to_global(P1, B)

        def forward(self, x):
            # stage 0
            in0 = x.to_global(P0, B)
            out0 = self.layer_0(in0)
            # stage 1
            in1 = out0.to_global(P1, B)
            out1 = self.layer_1(in1)
            return out1

    pp_m = PipelineModule()
    pp_m.eval()

    class PipelineGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.pp_m = pp_m

        def build(self, x):
            return self.pp_m(x)

    pp_g = PipelineGraph()

    for i in range(500):
        x = flow.randn(16, 10)
        x = x.to_global(P0, B)
        out = pp_g(x)
        # print(out.to_local().mean())


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n2d()
class TestGraphPipelineBufferLimit(oneflow.unittest.TestCase):
    def test_graph_buffer_limit(test_case):
        _test_graph_buffer_limit(test_case)


if __name__ == "__main__":
    unittest.main()
