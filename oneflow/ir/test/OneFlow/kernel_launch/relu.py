# RUN: python3 %s
import numpy as np
import os

os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
os.environ["ONEFLOW_MLIR_PREFER_NHWC"] = "1"

import oneflow as flow


class GraphToRun(flow.nn.Graph):
    def __init__(self):
        super().__init__()

    def build(self, x):
        return flow.relu(x)


x = flow.Tensor([1, -1])
graph_to_run = GraphToRun()
lazy_relu = graph_to_run(x)
assert flow.all(flow.equal(lazy_relu, flow.Tensor([1, 0])))
