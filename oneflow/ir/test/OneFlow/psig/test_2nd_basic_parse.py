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
# RUN: python3 -m oneflow.distributed.launch --nproc_per_node 2 %s | FileCheck %s
# CHECK: [#sbp.B, #sbp.S<0>]
# CHECK: [#sbp.B, #sbp.S<0>]

import oneflow as flow
import unittest
import oneflow.unittest
import os
from google.protobuf import text_format

os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"


def _test_nd_basic_parse(test_case):
    class ModuleToRun(flow.nn.Module):
        def __init__(self):
            super().__init__()
            P0 = flow.placement("cpu", ranks=[[0], [1]])
            a0_sbp = (flow.sbp.broadcast, flow.sbp.split(0))
            b0_sbp = (flow.sbp.broadcast, flow.sbp.split(0))

            self.A0 = flow.randn(4, 5, placement=P0, sbp=a0_sbp)
            self.B0 = flow.randn(5, 8, placement=P0, sbp=b0_sbp)

        def forward(self):
            return flow.matmul(self.A0, self.B0)

    net = ModuleToRun()

    class GraphToRun(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.net = net

        def build(self):
            return self.net()

    graph_to_run = GraphToRun()
    lazy_output = graph_to_run()

    serialized_job = graph_to_run._forward_job_proto.SerializeToString()
    mlir = flow._oneflow_internal.nn.graph.ConvertJobToIR(serialized_job)
    print(mlir)


@flow.unittest.skip_unless_1n1d()
class TestBasicParse(flow.unittest.TestCase):
    def test_nd_basic_parse(test_case):
        _test_nd_basic_parse(test_case)


if __name__ == "__main__":
    unittest.main()
