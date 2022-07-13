import oneflow as flow
import os
from google.protobuf import text_format

os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"

class ModuleToRun(flow.nn.Module):
    def __init__(self):
        super().__init__()
        P0 = flow.placement("cuda", ranks=[0])
        a0_sbp = (flow.sbp.broadcast)
        b0_sbp = flow.sbp.broadcast
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
flow._oneflow_internal.nn.graph.SaveJobToIR(serialized_job, str("/home/yuhao"))
pass
