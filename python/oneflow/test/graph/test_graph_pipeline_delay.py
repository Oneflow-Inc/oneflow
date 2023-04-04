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
from oneflow.nn.graph import GraphModule


def _test_graph_pipeline_delay_output(test_case):
    class StageLayerModule(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = flow.nn.Linear(10, 8, False)
            self.linear2 = flow.nn.Linear(8, 10)
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
    pp_m.train()
    of_sgd = flow.optim.SGD(pp_m.parameters(), lr=0.001)

    class PipelineGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.pp_m = pp_m
            self.pp_m.layer_0.to(GraphModule).stage_id = 0
            self.pp_m.layer_1.to(GraphModule).stage_id = 1
            self.config.set_gradient_accumulation_steps(4)
            self.add_optimizer(of_sgd)

        def build(self, x, y):
            pp_out = self.pp_m(x)
            loss = pp_out.mean()
            loss.backward()
            y = x + y
            free_out = y.to_global(P1, B)
            return loss, free_out

    pp_g = PipelineGraph()
    rank = flow.env.get_rank()
    for i in range(3):
        x = flow.randn(16, 10)
        y = flow.randn(16, 10)
        x = x.to_global(P0, B)
        y = y.to_global(P0, B)
        if rank == 1:
            time.sleep(2)
        loss_pack_4, free_out = pp_g(x, y)
        if rank == 1:
            # NOTE(chengcheng): Before Oneflow-Inc/oneflow#6221 fix src/dst tick order with input/output,
            #   this case use sleep in rank 1 will expose this BUG:
            #   free_out is output only on rank 1, but NOT control in rank 1 src/dst tick, so if manual sleep
            #   on rank 1, free out pull callback must exec before rank 1 src tick exec, so will meet BUG of
            #   output_kernel buffer status empty.
            #   After this PR fix, this test case ensure that src/dst tick and input/output cb exec order on
            #   each rank is as expected.
            time.sleep(2)
            print(
                "rank: ",
                rank,
                "packed loss with 4 micro-batch = ",
                loss_pack_4.to_local(),
            )
            print(
                "rank: ",
                rank,
                "packed image with 4 micro-batch = ",
                free_out.to_local(),
            )


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n2d()
class TestGraphPipelineDelayOutput(oneflow.unittest.TestCase):
    def test_graph_pipeline_delay_output(test_case):
        _test_graph_pipeline_delay_output(test_case)


if __name__ == "__main__":
    unittest.main()
