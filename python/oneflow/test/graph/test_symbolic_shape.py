import os

import oneflow as flow
import oneflow._oneflow_internal.lazy_mode as lazy_mode
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
        x = x.unsqueeze(0)
        if lazy_mode.is_enabled():
            # Shape inference works correctly even with the presence of
            # symbolic dimensions:
            assert x.shape == (1, flow.Dim.unknown(), 4)
        return x


def test_run_graph_by_vm(capsys):
    os.environ["ONEFLOW_RUN_GRAPH_BY_VM"] = "1"
    os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
    os.environ["ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION"] = "1"

    m = M().eval()
    g = Graph(m)
    input = flow.randn(2, 4)
    g._input_shape[input] = (flow.Dim.unknown(), 4)

    graph_output = g(input)
    eager_output = m(input)
    assert graph_output.shape == (1, 2, 4)
    assert np.allclose(graph_output, eager_output)

    input = flow.randn(3, 4)
    graph_output = g(input)
    eager_output = m(input)
    assert graph_output.shape == (1, 3, 4)
    assert np.allclose(graph_output, eager_output)

    # Test the optimization in graph works.
    # broadcast_sub and cast ops are pruned.
    print(g)
    assert "broadcast_sub" not in capsys.readouterr().out
    assert "cast" not in capsys.readouterr().out
    assert "broadcast_mul" not in capsys.readouterr().out

    os.environ["ONEFLOW_RUN_GRAPH_BY_VM"] = "0"
    os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "0"
    os.environ["ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION"] = "0"
