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
        y = x.sum(dim=1)
        if lazy_mode.is_enabled():
            # Shape inference works correctly even with the presence of
            # symbolic dimensions:
            assert x.shape == (1, flow.Dim.unknown(), 4)
            assert isinstance(x.shape[0], int)
            assert isinstance(x.shape[2], int)
            # y has a static shape even though x has a symbolic shape
            assert y.shape == (1, 4)
        return x


def test_graph_with_symbolic_shape(capsys):
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


def test_symbolic_shape_equality():
    assert flow.Dim.unknown() == flow.Dim.unknown()
    assert flow.Dim.unknown() != -1
    assert flow.Dim.unknown() != 1
    assert (1, flow.Dim.unknown(), 4) == (1, flow.Dim.unknown(), 4)
    assert flow.Size((1, flow.Dim.unknown(), 4)) == (1, flow.Dim.unknown(), 4)
    assert flow.Size((1, flow.Dim.unknown(), 4)) == flow.Size(
        (1, flow.Dim.unknown(), 4)
    )
    assert flow.Size((1, flow.Dim.unknown(), 4)) != flow.Size((1, 1, 4))
