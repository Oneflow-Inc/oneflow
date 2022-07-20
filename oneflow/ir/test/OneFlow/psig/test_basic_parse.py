import oneflow as flow
import unittest
import oneflow.unittest

import os
from google.protobuf import text_format

# RUN: python3 %s | FileCheck %s
# CHECK: [#sbp.b]
# CHECK: [#sbp.s<0>]

os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
def _test_1nd_basic_parse(test_case):
    class ModuleToRun(flow.nn.Module):
        def __init__(self):
            super().__init__()
            P0 = flow.placement("cuda", ranks=[0])
            a0_sbp = (flow.sbp.broadcast)
            b0_sbp = (flow.sbp.split(0))
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

    serialized_job = str(text_format.MessageToString(graph_to_run._forward_job_proto))

    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as dirname:
        flow._oneflow_internal.nn.graph.SaveJobToIR(serialized_job, str(dirname))
        f_name = os.path.join(dirname, 'model.mlir')
        f = open(f_name)
        print(f.read())
        f.close()

@flow.unittest.skip_unless_1n1d()
class TestBasicParse(flow.unittest.TestCase):
    def test_1nd_basic_parse(test_case):
        _test_1nd_basic_parse(test_case)


if __name__ == "__main__":
    unittest.main()
