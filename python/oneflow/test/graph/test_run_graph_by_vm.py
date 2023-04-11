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
    # TODO: with EnvVar(...):
    os.environ["ONEFLOW_RUN_GRAPH_BY_VM"] = "1"
    os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
    os.environ["ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION"] = "1"

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
    # print(g)
    # assert "broadcast_sub" not in capsys.readouterr().out
    # assert "cast" not in capsys.readouterr().out
    # assert "broadcast_mul" not in capsys.readouterr().out

    os.environ["ONEFLOW_RUN_GRAPH_BY_VM"] = "0"
    os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "0"
    os.environ["ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION"] = "0"


class DataParallelMul(flow.nn.Module):
    def __init__(self, placement) -> None:
        super().__init__()
        self.w = flow.randn(5, 8, placement=placement, sbp=flow.sbp.broadcast)

    def forward(self, x):
        return flow.matmul(x, self.w)


def test_data_parallel_run_by_vm():
    os.environ["ONEFLOW_RUN_GRAPH_BY_VM"] = "1"
    os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
    os.environ["ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION"] = "1"

    placement = flow.placement("cuda", [0, 1])

    m = DataParallelMul(placement).eval()
    g = Graph(m)
    
    input = flow.randn(4, 5, placement=placement, sbp=flow.sbp.split(0))
    graph_output = g(input)
    eager_output = m(input)

    assert graph_output.sbp == eager_output.sbp
    assert graph_output.shape == eager_output.shape
    assert graph_output.placement == eager_output.placement
    assert np.allclose(graph_output, eager_output)

    os.environ["ONEFLOW_RUN_GRAPH_BY_VM"] = "0"
    os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "0"
    os.environ["ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION"] = "0" 

test_data_parallel_run_by_vm()


class ModuleParallelMul(flow.nn.Module):
    def __init__(self, placement) -> None:
        super().__init__()
        self.w = flow.randn(5, 8, placement=placement, sbp=flow.sbp.split(1))

    def forward(self, x):
        return flow.matmul(x, self.w)


def test_module_parallel_run_by_vm():
    os.environ["ONEFLOW_RUN_GRAPH_BY_VM"] = "1"
    os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
    os.environ["ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION"] = "1"

    placement = flow.placement("cuda", [0, 1])
    m = ModuleParallelMul(placement).eval()
    g = Graph(m)
    
    input = flow.randn(4, 5, placement=placement, sbp=flow.sbp.broadcast)
    graph_output = g(input)
    eager_output = m(input)

    assert graph_output.sbp == eager_output.sbp
    assert graph_output.shape == eager_output.shape
    assert graph_output.placement == eager_output.placement
    assert np.allclose(graph_output, eager_output)

    os.environ["ONEFLOW_RUN_GRAPH_BY_VM"] = "0"
    os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "0"
    os.environ["ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION"] = "0"  

test_module_parallel_run_by_vm()


class BoxingModuleParallelMul(flow.nn.Module):
    def __init__(self, placement) -> None:
        super().__init__()
        self.w1 = flow.randn(5, 8, placement=placement, sbp=flow.sbp.split(1))
        self.w2 = flow.randn(8, 6, placement=placement, sbp=flow.sbp.split(1))

    def forward(self, x):
        x = flow.matmul(x, self.w1)
        x = flow.matmul(x, self.w2)
        return x


def test_boxing_data_parallel_run_by_vm():
    os.environ["ONEFLOW_RUN_GRAPH_BY_VM"] = "1"
    os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
    os.environ["ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION"] = "1"

    placement = flow.placement("cuda", [0, 1])
    m = BoxingModuleParallelMul(placement).eval()
    g = Graph(m)
    
    input = flow.randn(4, 5, placement=placement, sbp=flow.sbp.broadcast)
    graph_output = g(input)
    eager_output = m(input)

    assert graph_output.sbp == eager_output.sbp
    assert graph_output.shape == eager_output.shape
    assert graph_output.placement == eager_output.placement
    assert np.allclose(graph_output, eager_output)

    os.environ["ONEFLOW_RUN_GRAPH_BY_VM"] = "0"
    os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "0"
    os.environ["ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION"] = "0"  

test_boxing_data_parallel_run_by_vm()