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
import unittest

import numpy as np

import oneflow
import oneflow as flow
import oneflow.unittest
import oneflow.sysconfig
from oneflow.nn.graph import GraphModule


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestGraphNeqDeviceProcessNum(flow.unittest.TestCase):
    def test_graph_process_num_greater_than_device(test_case):
        # NOTE(chengcheng): this test case is ONLY for 1n8d in 4d env.
        if not (flow.env.get_node_size() == 1 and flow.env.get_world_size() == 8):
            return
        if not oneflow.sysconfig.has_rpc_backend_grpc():
            return

        BATCH_SIZE = 64
        BROADCAST = [flow.sbp.broadcast]
        P0 = flow.placement("cpu", ranks=[0, 1, 2, 3])
        P1 = flow.placement("cpu", ranks=[4, 5, 6, 7])

        class Stage0Module(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.flatten = flow.nn.Flatten()
                self.linear0 = flow.nn.Linear(28 * 28, 512)
                self.relu0 = flow.nn.ReLU()

            def forward(self, x):
                out = self.flatten(x)
                out = self.linear0(out)
                out = self.relu0(out)
                return out

        class Stage1Module(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = flow.nn.Linear(512, 512)
                self.relu1 = flow.nn.ReLU()
                self.linear2 = flow.nn.Linear(512, 10)
                self.relu2 = flow.nn.ReLU()

            def forward(self, x):
                out = self.linear1(x)
                out = self.relu1(out)
                out = self.linear2(out)
                out = self.relu2(out)
                return out

        class PipelineModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.m_stage0 = Stage0Module()
                self.m_stage1 = Stage1Module()

                self.m_stage0.to_global(placement=P0, sbp=BROADCAST)
                self.m_stage1.to_global(placement=P1, sbp=BROADCAST)

            def forward(self, x):
                out_stage0 = self.m_stage0(x)
                in_stage1 = out_stage0.to_global(placement=P1, sbp=flow.sbp.split(0))
                out_stage1 = self.m_stage1(in_stage1)
                return out_stage1

        module_pipeline = PipelineModule()
        sgd = flow.optim.SGD(module_pipeline.parameters(), lr=0.001)

        class PipelineGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.module_pipeline = module_pipeline
                self.module_pipeline.m_stage0.to(GraphModule).set_stage(0)
                self.module_pipeline.m_stage1.to(GraphModule).set_stage(1)
                self.loss_fn = flow.nn.CrossEntropyLoss(reduction="none")
                self.config.set_gradient_accumulation_steps(2)
                self.add_optimizer(sgd)

            def build(self, x, y):
                out = self.module_pipeline(x)
                loss = self.loss_fn(out, y).sum()
                loss = loss.to_global(placement=P1, sbp=BROADCAST)
                loss.backward()
                return loss

        graph_pipeline = PipelineGraph()
        graph_pipeline.debug(1)

        x = flow.randn(BATCH_SIZE, 1, 28, 28)
        x = x.to_global(P0, sbp=flow.sbp.split(0))
        y = flow.randint(0, 10, (BATCH_SIZE, 1))
        y = y.to_global(P1, sbp=flow.sbp.split(0))

        for i in range(2):
            loss = graph_pipeline(x, y)
            print(">>>>>>>", flow.env.get_rank(), loss.to_local().numpy(), flush=True)


if __name__ == "__main__":
    unittest.main()
