import os

os.environ["ONEFLOW_RUN_GRAPH_BY_VM"] = "1"
os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
os.environ["ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION"] = "1"

import oneflow as flow
import numpy as np


class Graph(flow.nn.Graph):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def build(self, x):
        return self.m(x)


class M(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = flow.nn.Parameter(flow.randn(4))

    def forward(self, x):
        # these broadcast_sub and cast ops will be
        # eliminated by nn.Graph
        w1 = self.w - self.w - self.w
        x = x * w1.to(flow.float32)
        return x


def test_run_graph_by_vm(capsys):
    m = M().eval()
    g = Graph(m)

    input = flow.randn(4)
    graph_output = g(input)
    eager_output = m(input)
    assert graph_output.shape == (4,)
    assert np.allclose(graph_output, eager_output)

    input = flow.randn(3, 4)
    graph_output = g(input)
    eager_output = m(input)
    assert graph_output.shape == (3, 4)
    assert np.allclose(graph_output, eager_output)

    # Test the optimization in graph works.
    # broadcast_sub and cast ops are pruned.
    print(g)
    assert "broadcast_sub" not in capsys.readouterr().out
    assert "cast" not in capsys.readouterr().out
    assert "broadcast_mul" not in capsys.readouterr().out
